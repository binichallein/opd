#!/usr/bin/env python3
"""Original Qwen student/teacher prompt acceptance, never a benchmark evaluation."""

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd_ext.math_protocol import completion_math_prompt, validate_qwen_historical_prompt
from diagnose_qwen4_initial_rollouts import save_json, select_sources, sha256, summarize
from diagnose_token_truncation import analyze_tokens


def prompt_variant(row, tokenizer, cell):
    if cell == 'historical':
        text = row['prompt_text']
        if tokenizer.encode(text, add_special_tokens=False) != row['prompt_token_ids']:
            raise ValueError('Archived prompt/tokenizer mismatch')
    elif cell == 'plain':
        text = row['question'] + '\n\nSolution:\n'
    elif cell == 'candidate':
        text = completion_math_prompt(row['question'])
    else:
        raise ValueError(cell)
    return text, tokenizer.encode(text, add_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, nargs='+', required=True)
    parser.add_argument('--role', choices=['student', 'teacher'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TOKENIZERS_PARALLELISM='false',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn')
    args.output.mkdir(parents=True, exist_ok=False)
    import prepare_qwen4_assets as assets
    protected = assets.verify_assets()
    model = assets.STUDENT if args.role == 'student' else assets.TEACHER
    rows = []
    for path in args.archives:
        if path.with_suffix(path.suffix + '.sha256').read_text().split()[0] != sha256(path):
            raise ValueError('Archive SHA mismatch')
        with gzip.open(path, 'rt') as f:
            rows.extend(json.loads(line) for line in f)
    sources = select_sources(rows)
    if len(sources) != 16:
        raise ValueError('This registered diagnostic requires 16 archived questions')
    for row in sources:
        row.pop('archived_response_token_ids')
    save_json(args.output / 'manifest.json', {
        'argv': sys.argv, 'pid': os.getpid(), 'created_at': datetime.now(timezone.utc).isoformat(),
        'model': str(model), 'role': args.role, 'assets': protected,
        'student_modelscope_revision': assets.REVISION,
        'teacher_identity': 'unchanged historical public GRPO checkpoint; hashes in assets',
        'archives': {str(p): sha256(p) for p in args.archives}, 'sources': sources,
        'code': {str(p): sha256(p) for p in [Path(__file__),
                 Path(__file__).with_name('diagnose_qwen4_initial_rollouts.py'),
                 Path(__file__).with_name('diagnose_token_truncation.py'),
                 Path(__file__).resolve().parents[1] / 'opd_ext/math_protocol.py']},
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'vllm']},
        'engine': {'dtype': 'bfloat16', 'tp': 1, 'memory': .6, 'eager': True, 'seed': 21,
                   'max_model_len': 18432, 'max_num_seqs': 32, 'max_num_batched_tokens': 18432,
                   'prefix_caching': True},
        'scope': '16 fixed training questions, original models; NOT benchmark scores',
        'seed_rule': 'sample j uses 21+j, n2, matched across models and prompt cells'})
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), tokenizer=str(model), dtype='bfloat16', tensor_parallel_size=1,
              gpu_memory_utilization=.6, max_model_len=18432, max_num_seqs=32,
              max_num_batched_tokens=18432, enforce_eager=True, seed=21, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    if tokenizer.eos_token_id != 151643 or tokenizer.bos_token_id is not None:
        raise ValueError('Unexpected Qwen completion EOS/BOS')
    for row in sources:
        validate_qwen_historical_prompt(tokenizer, row['question'], row['prompt_token_ids'])
    config = dict(temperature=1., top_p=.9, top_k=-1, max_tokens=16384,
                  n=1, ignore_eos=False, stop_token_ids=[], logprobs=0, detokenize=False)
    summaries = {}
    for cell in ('historical', 'plain', 'candidate'):
        folder = args.output / cell
        folder.mkdir()
        requests = {}
        for i, original in enumerate(sources):
            text, ids = prompt_variant(original, tokenizer, cell)
            if len(ids) > 2048:
                raise ValueError('Diagnostic prompt exceeds training context cap')
            if cell != 'historical' and any(x in ids for x in (151644, 151645, 151667, 151668)):
                raise ValueError('Chat/thinking controls in completion input')
            for j in range(2):
                rid = f'{cell}:{i}:{j}'
                requests[rid] = {**original, 'prompt_text': text, 'prompt_token_ids': ids,
                                 'request_id': rid, 'seed': 21+j, 'sample_index': j}
        save_json(folder / 'inputs.json', {'config': config, 'requests': list(requests.values())})
        for rid, row in requests.items():
            llm.llm_engine.add_request(rid, {'prompt_token_ids': row['prompt_token_ids']},
                                      SamplingParams(**config, seed=row['seed']))
        records = []
        with (folder / 'raw.jsonl').open('x', encoding='utf-8') as f:
            while llm.llm_engine.has_unfinished_requests():
                for output in llm.llm_engine.step():
                    if not output.finished:
                        continue
                    row = requests[output.request_id]
                    if list(output.prompt_token_ids) != row['prompt_token_ids']:
                        raise ValueError('Actual GPU input IDs mismatch')
                    sample = output.outputs[0]
                    ids = list(sample.token_ids)
                    if len(sample.logprobs) != len(ids):
                        raise ValueError('Missing sampled logprobs')
                    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    record = {**row, 'cell': cell, 'response_token_ids': ids, 'response_text': text,
                              'finish_reason': sample.finish_reason, 'stop_reason': sample.stop_reason,
                              'sampled_logprobs': [float(lp[t].logprob) for t, lp in zip(ids, sample.logprobs)],
                              'contains_boxed_marker': '\\boxed' in text,
                              'generated_think_tags': '<think>' in text or '</think>' in text,
                              **analyze_tokens(ids, 16384)}
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    records.append(record)
                    print(json.dumps({'cell': cell, 'done': len(records), 'length': len(ids),
                                      'index': row['index'], 'seed': row['seed'],
                                      'finish': sample.finish_reason}), flush=True)
        if len(records) != 32 or len({r['request_id'] for r in records}) != 32:
            raise ValueError('Incomplete or duplicate outputs')
        stats = {**summarize(records), 'raw_sha256': sha256(folder / 'raw.jsonl'),
                 'high_4gram_repeat_tails': sum(r['tail_repeated_4gram_fraction'] >= .9 for r in records),
                 'format_errors': sum(not r['contains_boxed_marker'] for r in records),
                 'think_tags': sum(r['generated_think_tags'] for r in records)}
        save_json(folder / 'summary.json', stats)
        summaries[cell] = stats
        print(json.dumps({'cell': cell, 'summary': stats}), flush=True)
    for name, digest in protected.items():
        if sha256(name) != digest:
            raise ValueError(f'Protected asset changed: {name}')
    save_json(args.output / 'summary.json', summaries)


if __name__ == '__main__':
    main()

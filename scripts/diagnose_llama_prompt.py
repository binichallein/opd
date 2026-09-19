#!/usr/bin/env python3
"""Inference-only native Llama prompt controls on archived training questions."""

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opd_ext.math_protocol import (LLAMA_STOP_IDS, historical_math_prompt, math_prompt,
                                   render_nonthinking, validate_historical_training_prompt)
from diagnose_qwen4_initial_rollouts import save_json, select_sources, sha256, summarize
from diagnose_token_truncation import analyze_tokens, without_think_requirement


def verify_files(expected):
    for name, digest in expected.items():
        if sha256(name) != digest:
            raise ValueError(f'Changed pinned asset: {name}')
    return expected


def completion_text(question):
    content = without_think_requirement(historical_math_prompt(question)).rstrip()
    return '<|begin_of_text|>' + content + '\n\nSolution:\n'


def prompt_variant(row, tokenizer, cell):
    if cell == 'historical':
        text = row['prompt_text']
        if tokenizer.encode(text, add_special_tokens=False) != row['prompt_token_ids']:
            raise ValueError('Tokenizer differs from archived input')
    elif cell == 'no_think':
        text = without_think_requirement(row['prompt_text'])
    elif cell == 'completion':
        text = completion_text(row['question'])
    elif cell == 'eval':
        text = render_nonthinking(tokenizer, math_prompt(row['question']))
    else:
        raise ValueError(cell)
    return text, tokenizer.encode(text, add_special_tokens=False)


def response_stats(ids, text, cap):
    stats = analyze_tokens(ids, cap)
    for key in ('first_eos_index', 'first_im_end_index', 'tokens_after_first_im_end'):
        stats.pop(key)
    first = next((i for i, token in enumerate(ids) if token in LLAMA_STOP_IDS), None)
    return {**stats, 'first_native_stop_index': first,
            'tokens_after_first_native_stop': len(ids) - first - 1 if first is not None else 0,
            'contains_boxed_marker': '\\boxed{' in text,
            'generated_think_tags': '<think>' in text or '</think>' in text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives', type=Path, nargs='+', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--cells', nargs='+', default=['historical', 'no_think', 'completion', 'eval'])
    args = parser.parse_args()
    if args.n < 1 or args.max_tokens < 1:
        raise ValueError('Positive sample count and token cap required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TOKENIZERS_PARALLELISM='false',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.model / 'asset_manifest.json').read_text())
    if manifest['provider'] != 'modelscope' or not manifest['repo'].endswith('Instruct'):
        raise ValueError('Expected original ModelScope Instruct assets')
    protected = verify_files(manifest['sha256'])
    archive_rows = []
    for path in args.archives:
        if path.with_suffix(path.suffix + '.sha256').read_text().split()[0] != sha256(path):
            raise ValueError('Archive SHA mismatch')
        with gzip.open(path, 'rt') as f:
            archive_rows.extend(json.loads(line) for line in f)
    sources = select_sources(archive_rows)
    # Later archived responses came from updated models; only their questions are reused.
    for row in sources:
        row.pop('archived_response_token_ids')
    save_json(args.output / 'manifest.json', {
        'argv': sys.argv, 'pid': os.getpid(), 'created_at': datetime.now(timezone.utc).isoformat(),
        'model': str(args.model), 'modelscope_revision': manifest['revision'], 'assets': protected,
        'archives': {str(p): sha256(p) for p in args.archives}, 'sources': sources,
        'code': {str(p): sha256(p) for p in [Path(__file__),
                 Path(__file__).with_name('diagnose_qwen4_initial_rollouts.py'),
                 Path(__file__).with_name('diagnose_token_truncation.py'),
                 Path(__file__).resolve().parents[1] / 'opd_ext/math_protocol.py']},
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'vllm']},
        'engine': {'dtype': 'bfloat16', 'tp': 1, 'memory': .6, 'eager': True,
                   'max_model_len': 18432, 'max_num_seqs': 32, 'max_num_batched_tokens': 18432,
                   'prefix_caching': True, 'seed': 21},
        'scope': 'Fixed training cohort, not benchmark scores; original models, no optimizer',
        'seed_rule': 'request sample j uses 21+j; matched across prompts/cells/models'})
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(args.model), tokenizer=str(args.model), dtype='bfloat16',
              tensor_parallel_size=1, gpu_memory_utilization=.6, max_model_len=18432,
              max_num_seqs=32, max_num_batched_tokens=18432, enforce_eager=True,
              seed=21, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    for row in sources:
        validate_historical_training_prompt(tokenizer, row['question'], row['prompt_token_ids'])
    summaries = {}
    config = dict(temperature=1., top_p=.9, top_k=-1, max_tokens=args.max_tokens,
                  n=1, ignore_eos=False, stop_token_ids=LLAMA_STOP_IDS,
                  logprobs=0, detokenize=False)
    for cell in args.cells:
        folder = args.output / cell
        folder.mkdir()
        requests = {}
        for i, original in enumerate(sources):
            text, ids = prompt_variant(original, tokenizer, cell)
            if len(ids) > 2048 or ids.count(128000) != 1 or ids[0] != 128000:
                raise ValueError('Prompt cap or native BOS invariant violated')
            if any(t in text for t in ('<|im_start|>', '<|im_end|>')):
                raise ValueError('Qwen ChatML leaked into Llama input')
            for j in range(args.n):
                rid = f'{cell}:{i}:{j}'
                requests[rid] = {**original, 'prompt_text': text, 'prompt_token_ids': ids,
                                 'request_id': rid, 'seed': 21 + j, 'sample_index': j}
        save_json(folder / 'inputs.json', {'config': config, 'requests': list(requests.values())})
        for rid, row in requests.items():
            llm.llm_engine.add_request(rid, {'prompt_token_ids': row['prompt_token_ids']},
                                      SamplingParams(**config, seed=row['seed']))
        records = []
        with (folder / 'raw.jsonl').open('x', encoding='utf-8') as f:
            while llm.llm_engine.has_unfinished_requests():
                for result in llm.llm_engine.step():
                    if not result.finished:
                        continue
                    row = requests[result.request_id]
                    if list(result.prompt_token_ids) != row['prompt_token_ids']:
                        raise ValueError('Engine prompt mismatch')
                    sample = result.outputs[0]
                    ids = list(sample.token_ids)
                    if len(sample.logprobs) != len(ids):
                        raise ValueError('Missing sampled logprobs')
                    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    record = {**row, 'cell': cell, 'response_token_ids': ids, 'response_text': text,
                              'finish_reason': sample.finish_reason, 'stop_reason': sample.stop_reason,
                              'sampled_logprobs': [float(lp[t].logprob) for t, lp in zip(ids, sample.logprobs)],
                              **response_stats(ids, text, args.max_tokens)}
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    records.append(record)
                    print(json.dumps({'cell': cell, 'done': len(records), 'total': len(requests),
                                      'index': row['index'], 'seed': row['seed'], 'length': len(ids),
                                      'finish': sample.finish_reason}), flush=True)
        if len(records) != len(requests) or len({r['request_id'] for r in records}) != len(requests):
            raise ValueError('Incomplete/duplicated records')
        stats = {**summarize(records), 'length_cap': args.max_tokens,
                 'high_4gram_repeat_tails': sum(r['tail_repeated_4gram_fraction'] >= .9 for r in records),
                 'boxed_marker_present': sum(r['contains_boxed_marker'] for r in records),
                 'think_tags_present': sum(r['generated_think_tags'] for r in records),
                 'raw_sha256': sha256(folder / 'raw.jsonl')}
        save_json(folder / 'summary.json', stats)
        summaries[cell] = stats
        print(json.dumps({'cell': cell, 'summary': stats}), flush=True)
    verify_files(protected)
    save_json(args.output / 'summary.json', summaries)


if __name__ == '__main__':
    main()

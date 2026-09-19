#!/usr/bin/env python3
"""Standalone initial-model inference; never loads or modifies trained checkpoints."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def select_sources(archive):
    selected = {}
    for row in archive:
        index = row['source_extra_info']['index']
        if index not in selected:
            selected[index] = {**row['source_extra_info'],
                               'prompt_token_ids': row['prompt_token_ids'],
                               'prompt_text': row['prompt_text'],
                               'archived_response_token_ids': row['response_token_ids']}
        elif selected[index]['prompt_token_ids'] != row['prompt_token_ids']:
            raise ValueError('Conflicting inputs for a source question')
    return list(selected.values())


def requests(rows, *, seed_mode, n):
    if seed_mode not in ('fixed', 'independent'):
        raise ValueError(seed_mode)
    return [{**row, 'request_id': f'{i}:{j}', 'sample_index': j,
             'seed': 21 if seed_mode == 'fixed' else 21 + j}
            for i, row in enumerate(rows) for j in range(n)]


def summarize(rows):
    return {'responses': len(rows),
            'unique_prompt_response_pairs': len({(r['index'], tuple(r['response_token_ids'])) for r in rows}),
            'length_stops': sum(r['finish_reason'] == 'length' for r in rows),
            'periodic_tails': sum(r['tail_period'] is not None for r in rows),
            'mean_length': statistics.mean(r['response_length'] for r in rows),
            'finish_reasons': dict(Counter(r['finish_reason'] for r in rows))}


def delete_chat_markers(ids):
    if ids.count(151644) != 2 or ids.count(151645) != 1:
        raise ValueError('Expected single-turn historical Qwen ChatML input')
    return [i for i in ids if i not in (151644, 151645)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--cells', nargs='+', default=['historical_fixed', 'historical_independent',
                                                     'no_think_independent', 'plain_independent', 'historical_greedy'])
    parser.add_argument('--n', type=int, default=8)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    args.output.mkdir(parents=True, exist_ok=False)
    from diagnose_token_truncation import analyze_tokens, without_think_requirement
    from vllm import LLM, SamplingParams

    with gzip.open(args.archive, 'rt') as f:
        source_rows = select_sources([json.loads(line) for line in f])
    save_json(args.output / 'manifest.json', {
        'argv': sys.argv, 'pid': os.getpid(), 'created_at': datetime.now(timezone.utc).isoformat(),
        'model': str(args.model), 'archive': str(args.archive), 'archive_sha256': sha256(args.archive),
        'script_sha256': sha256(__file__), 'helper_sha256': sha256(Path(__file__).with_name('diagnose_token_truncation.py')),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'vllm']},
        'scope': 'Inference-only diagnostic, not benchmark evaluation or a training result',
        'length_cap': args.max_tokens, 'eos': 'model EOS; no added stop IDs',
        'engine': 'standalone vLLM BF16 TP1 eager; no FSDP, weight synchronization or optimizer',
        'sources': source_rows})
    llm = LLM(model=str(args.model), tokenizer=str(args.model), dtype='bfloat16',
              tensor_parallel_size=1, gpu_memory_utilization=0.6, max_model_len=18432,
              max_num_seqs=32, max_num_batched_tokens=18432, enforce_eager=True,
              seed=21, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    summaries = {}
    for cell in args.cells:
        prompt_mode, sampling_mode = cell.rsplit('_', 1)
        rows = []
        for original in source_rows:
            row = dict(original)
            if prompt_mode == 'historical':
                if tokenizer.encode(row['prompt_text'], add_special_tokens=False) != row['prompt_token_ids']:
                    raise ValueError('Tokenizer does not preserve archived input')
            elif prompt_mode == 'no_think':
                row['prompt_text'] = without_think_requirement(original['prompt_text'])
                row['prompt_token_ids'] = tokenizer.encode(row['prompt_text'], add_special_tokens=False)
            elif prompt_mode == 'plain':
                row['prompt_text'] = original['question'] + '\n\nSolution:\n'
                row['prompt_token_ids'] = tokenizer.encode(row['prompt_text'], add_special_tokens=False)
            elif prompt_mode == 'no_markers':
                row['prompt_token_ids'] = delete_chat_markers(original['prompt_token_ids'])
                row['prompt_text'] = tokenizer.decode(row['prompt_token_ids'], skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
            else:
                raise ValueError(prompt_mode)
            rows.append(row)
        greedy = sampling_mode == 'greedy'
        reqs = requests(rows, seed_mode='fixed' if greedy else sampling_mode, n=1 if greedy else args.n)
        mapping = {r['request_id']: r for r in reqs}
        config = {'temperature': 0. if greedy else 1., 'top_p': 1. if greedy else .9,
                  'top_k': -1, 'max_tokens': args.max_tokens, 'n': 1, 'ignore_eos': False,
                  'logprobs': 0, 'detokenize': False}
        folder = args.output / cell
        folder.mkdir()
        save_json(folder / 'inputs.json', {'config': config, 'requests': reqs})
        for request in reqs:
            llm.llm_engine.add_request(request['request_id'], {'prompt_token_ids': request['prompt_token_ids']},
                                      SamplingParams(**config, seed=request['seed']))
        records = []
        with (folder / 'raw.jsonl').open('x', encoding='utf-8') as f:
            while llm.llm_engine.has_unfinished_requests():
                for result in llm.llm_engine.step():
                    if not result.finished:
                        continue
                    sample = result.outputs[0]
                    ids = list(sample.token_ids)
                    row = mapping[result.request_id]
                    old_ids = row['archived_response_token_ids']
                    common = 0
                    for a, b in zip(ids, old_ids):
                        if a != b:
                            break
                        common += 1
                    raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    record = {**row, 'cell': cell, 'response_token_ids': ids, 'response_text': raw,
                              'finish_reason': sample.finish_reason, 'stop_reason': sample.stop_reason,
                              'cumulative_logprob': sample.cumulative_logprob,
                              'sampled_logprobs': [float(lp[t].logprob) for t, lp in zip(ids, sample.logprobs)],
                              'archived_common_prefix_tokens': common,
                              **analyze_tokens(ids, args.max_tokens)}
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    records.append(record)
                    print(json.dumps({'cell': cell, 'completed': len(records), 'total': len(reqs),
                                      'index': row['index'], 'seed': row['seed'], 'length': len(ids),
                                      'tail_period': record['tail_period'], 'common_prefix': common}), flush=True)
        if len(records) != len(reqs) or len({r['request_id'] for r in records}) != len(reqs):
            raise ValueError('Incomplete or duplicated requests')
        stats = summarize(records)
        stats.update(raw_sha256=sha256(folder / 'raw.jsonl'), length_cap=args.max_tokens)
        save_json(folder / 'summary.json', stats)
        summaries[cell] = stats
        print(json.dumps({'cell': cell, 'summary': stats}), flush=True)
    save_json(args.output / 'summary.json', summaries)


if __name__ == '__main__':
    main()

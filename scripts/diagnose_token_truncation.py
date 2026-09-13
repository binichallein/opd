#!/usr/bin/env python3
"""Inference-only, matched-prompt truncation probe with lossless raw outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import statistics

ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
RUNTIME = ROOT / 'deployments/ec0a7a950540d7f4753c08b18bc26c544f6a7cde'
OLD = ROOT / 'runs/20260712v1_token_opd_replication_seed21_ml2/token_opd'
NEW = ROOT / 'runs/20260913v1_qwen06_pair_seed21_ml2/token_opd'
MODELS = {
    'q06_base': ROOT / 'models/Qwen3-0.6B-Base',
    'q17_base': Path('/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base'),
    'q06_step50': NEW / 'checkpoints/global_step_50/actor/huggingface',
    'q17_step50': OLD / 'checkpoints/global_step_50/actor/huggingface',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_records(path, records):
    with Path(path).open('x', encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
        os.fsync(handle.fileno())


def analyze_tokens(token_ids, max_tokens):
    import numpy as np

    ids = list(token_ids)
    tail = ids[-2048:]
    period, match = None, None
    array = np.asarray(tail)
    if len(tail) >= 256:
        for offset in range(1, min(128, len(tail) // 4) + 1):
            fraction = float(np.mean(array[offset:] == array[:-offset]))
            if fraction >= 0.99:
                period, match = offset, fraction
                break
    grams = [tuple(tail[i:i + 4]) for i in range(max(0, len(tail) - 3))]
    first_eos = ids.index(151643) if 151643 in ids else None
    first_end = ids.index(151645) if 151645 in ids else None
    return {
        'response_length': len(ids), 'length_at_cap': len(ids) >= max_tokens,
        'first_eos_index': first_eos, 'first_im_end_index': first_end,
        'tokens_after_first_im_end': len(ids) - first_end - 1 if first_end is not None else 0,
        'tail_period': period, 'tail_period_match': match,
        'tail_dominant_token_fraction': max(Counter(tail).values()) / len(tail) if tail else 0,
        'tail_repeated_4gram_fraction': 1 - len(set(grams)) / len(grams) if grams else 0,
    }


def build_record(row, token_ids, text, finish_reason, stop_reason, cumulative_logprob,
                 *, label, seed, max_tokens):
    return {
        **row, 'label': label, 'seed': seed, 'max_tokens': max_tokens,
        'response_token_ids': list(token_ids), 'response_raw': text,
        'finish_reason': finish_reason, 'stop_reason': stop_reason,
        'cumulative_logprob': cumulative_logprob,
        **analyze_tokens(token_ids, max_tokens),
    }


def without_think_requirement(text):
    phrase = 'conduct reasoning inside <think> and </think> and '
    if text.count(phrase) != 1:
        raise ValueError('Expected exactly one original think instruction')
    return text.replace(phrase, '')


def prepare(output):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from types import SimpleNamespace
    from verl.utils.dataset.multitask_rl_dataset import SequentialTaskSampler

    data = ROOT / 'data/math_opd_dapo17k_hf_full_eval4/train.parquet'
    parquet = pq.ParquetFile(data)
    if parquet.metadata.num_rows != 1791700 or 'task_type' in parquet.schema_arrow.names:
        raise ValueError('Unexpected historical data shape/task grouping')
    # The frozen dataset has no task_type column: the actual sampler uses one unknown group.
    dataset = SimpleNamespace(get_task_types=lambda: ['unknown'],
                              get_task_indices=lambda _: range(parquet.metadata.num_rows))
    sampler = SequentialTaskSampler(dataset, batch_size=4, seed=21, shuffle=True, drop_last=False)
    indices = list(map(int, itertools.islice(iter(sampler), 32)))
    selected = {}
    start = 0
    for group in range(parquet.metadata.num_row_groups):
        end = start + parquet.metadata.row_group(group).num_rows
        wanted = [i for i in indices if start <= i < end]
        if wanted:
            table = parquet.read_row_group(group, columns=['env_kwargs'])
            for index, row in zip(wanted, table.take([i - start for i in wanted]).to_pylist()):
                selected[index] = row['env_kwargs']
        start = end
    tokenizers = [AutoTokenizer.from_pretrained(str(MODELS[name]), local_files_only=True)
                  for name in ['q06_base', 'q17_base']]
    template_path = RUNTIME / 'external/revisiting_opd/agent_system/environments/prompts/math.py'
    template = runpy.run_path(str(template_path))['MATH_TEMPLATE']
    rows = []
    for order, index in enumerate(indices):
        item = selected[index]
        content = template.format(task_description=item['question'])
        texts = [t.apply_chat_template([{'role': 'user', 'content': content}],
                                      tokenize=False, add_generation_prompt=True) for t in tokenizers]
        ids = [t.encode(text, add_special_tokens=False) for t, text in zip(tokenizers, texts)]
        if texts[0] != texts[1] or ids[0] != ids[1]:
            raise ValueError('Student prompts are not token-identical')
        original_length = len(ids[0])
        tokens = ids[0] if original_length <= 2048 else ids[0][:1024] + ids[0][-1024:]
        rows.append({'index': index, 'train_step': order // 4 + 1, 'position_in_batch': order % 4,
                     'question': item['question'], 'answer': item['ground_truth'],
                     'training_user_content': content,
                     'prompt': texts[0], 'prompt_token_ids': tokens,
                     'prompt_original_length': original_length})
    matches = {}
    for run in [OLD, NEW]:
        known = [json.loads(line) for line in (run / 'diagnostics/scalars.jsonl').read_text().splitlines()]
        for step in [1, 5]:
            batch = rows[(step - 1) * 4:step * 4]
            digests = [hashlib.sha256(np.asarray(row['prompt_token_ids'], dtype='<i8').tobytes()).digest()
                       for row in batch for _ in range(8)]
            actual = hashlib.sha256(b''.join(sorted(digests))).hexdigest()
            expected = next(row['prompt_batch_sha256'] for row in known if row['step'] == step)
            if actual != expected:
                raise ValueError(f'Prompt reconstruction mismatch: {run} step{step}')
            matches[f'{run.parent.name}/step{step}'] = actual
    output.mkdir(parents=True, exist_ok=False)
    write_records(output / 'prompts.jsonl', rows)
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'scope': '32 first scheduled training prompts, inference-only; not original rollouts or benchmark eval',
                'script_sha256': sha256(__file__), 'data_sha256': sha256(data),
                'training_template_sha256': sha256(template_path),
                'prompts_sha256': sha256(output / 'prompts.jsonl'), 'prompt_reconstruction_matches': matches,
                'data_seed': 21, 'generation_seed': 21, 'temperature': 1.0, 'top_p': 0.9,
                'max_tokens': 16384, 'prompt_mode': 'training chat template, no enable_thinking override',
                'stop_mode': 'model EOS only (151643), matching training, not evaluation stop set',
                'models': {name: str(path) for name, path in MODELS.items()},
                'historical_gpu_bitwise_replay': False}
    write_records(output / 'manifest.jsonl', [manifest])
    print(json.dumps(manifest, indent=2), flush=True)


def generate(root, label, gpu, *, stop_mode='training', prompt_mode='training', seed=21, select=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from vllm import LLM, SamplingParams

    manifest = json.loads((root / 'manifest.jsonl').read_text())
    if sha256(root / 'prompts.jsonl') != manifest['prompts_sha256']:
        raise ValueError('Cohort identity changed')
    rows = [json.loads(line) for line in (root / 'prompts.jsonl').read_text().splitlines()]
    if select:
        rows = [row for row in rows if row['index'] in set(select)]
        if len(rows) != len(set(select)):
            raise ValueError('Requested row is missing')
    destination = root / f'{label}_{stop_mode}_{prompt_mode}_seed{seed}'
    destination.mkdir(exist_ok=False)
    model = MODELS[label]
    llm = LLM(model=str(model), tokenizer=str(MODELS['q06_base']), dtype='bfloat16',
              tensor_parallel_size=1, gpu_memory_utilization=0.6, max_model_len=18432,
              max_num_seqs=32, max_num_batched_tokens=18432, enforce_eager=True, seed=21,
              enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    stop_ids = [151643, 151645] if stop_mode == 'both' else None
    prompts = []
    for row in rows:
        if prompt_mode == 'plain':
            row = dict(row, prompt=row['question'] + '\n\nSolution:\n')
            row['prompt_token_ids'] = tokenizer.encode(row['prompt'], add_special_tokens=False)
        elif prompt_mode == 'eval':
            row = dict(row, prompt=tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['question']}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
            row['prompt_token_ids'] = tokenizer.encode(row['prompt'], add_special_tokens=False)
        elif prompt_mode == 'no_think':
            row = dict(row, prompt=tokenizer.apply_chat_template(
                [{'role': 'user', 'content': without_think_requirement(row['training_user_content'])}],
                tokenize=False, add_generation_prompt=True))
            row['prompt_token_ids'] = tokenizer.encode(row['prompt'], add_special_tokens=False)
        prompts.append(row)
    sampling = SamplingParams(temperature=1., top_p=.9, top_k=-1, max_tokens=16384,
                              seed=seed, n=1, ignore_eos=False, stop_token_ids=stop_ids,
                              logprobs=0, detokenize=False)
    write_records(destination / 'config.jsonl', [{
        'model': str(model), 'tokenizer': str(MODELS['q06_base']), 'label': label,
        'gpu': gpu, 'seed': seed, 'stop_mode': stop_mode, 'prompt_mode': prompt_mode,
        'sampling': str(sampling), 'n_prompts': len(prompts),
        'script_sha256': sha256(__file__), 'started_at': datetime.now(timezone.utc).isoformat(),
        'engine': 'vllm, bf16, eager, TP1, memory0.6; identical probe engine settings across models',
    }])
    # Use the engine's public request API so finished sequences are durable before the full batch ends.
    for i, row in enumerate(prompts):
        llm.llm_engine.add_request(str(i), {'prompt_token_ids': row['prompt_token_ids']}, sampling)
    records = []
    with (destination / 'raw.jsonl').open('x', encoding='utf-8') as handle:
        while llm.llm_engine.has_unfinished_requests():
            for result in llm.llm_engine.step():
                if not result.finished:
                    continue
                sample = result.outputs[0]
                row = prompts[int(result.request_id)]
                raw = tokenizer.decode(sample.token_ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
                record = build_record(row, sample.token_ids, raw, sample.finish_reason,
                                      sample.stop_reason, sample.cumulative_logprob,
                                      label=label, seed=seed, max_tokens=16384)
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
                records.append(record)
                print(json.dumps({'completed': len(records), 'total': len(prompts),
                                  'index': row['index'], 'length': len(sample.token_ids),
                                  'finish_reason': sample.finish_reason,
                                  'tail_period': record['tail_period']}), flush=True)
    if len(records) != len(prompts) or len({row['index'] for row in records}) != len(prompts):
        raise ValueError('Missing or duplicated completed responses')
    summary = {'label': label, 'n': len(records), 'seed': seed,
               'stop_mode': stop_mode, 'prompt_mode': prompt_mode,
               'finish_reasons': dict(Counter(row['finish_reason'] for row in records)),
               'length_at_cap_count': sum(row['length_at_cap'] for row in records),
               'length_at_cap_rate': statistics.mean(row['length_at_cap'] for row in records),
               'mean_length': statistics.mean(row['response_length'] for row in records),
               'tail_periodic_count': sum(row['tail_period'] is not None for row in records),
               'capped_tail_periodic_count': sum(row['length_at_cap'] and row['tail_period'] is not None for row in records),
               'im_end_before_cap_count': sum(row['first_im_end_index'] is not None and row['first_im_end_index'] < 16383 for row in records),
               'capped_with_earlier_im_end': sum(row['length_at_cap'] and row['first_im_end_index'] is not None and row['first_im_end_index'] < 16383 for row in records),
               'raw_sha256': sha256(destination / 'raw.jsonl'),
               'finished_at': datetime.now(timezone.utc).isoformat()}
    write_records(destination / 'summary.jsonl', [summary])
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'generate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--label', choices=MODELS)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--stop-mode', choices=['training', 'both'], default='training')
    parser.add_argument('--prompt-mode', choices=['training', 'plain', 'eval', 'no_think'], default='training')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--select', type=int, nargs='+')
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.root)
    else:
        if args.label is None or args.gpu is None:
            parser.error('generate requires --label and --gpu')
        generate(args.root, args.label, args.gpu, stop_mode=args.stop_mode,
                 prompt_mode=args.prompt_mode, seed=args.seed, select=args.select)


if __name__ == '__main__':
    main()

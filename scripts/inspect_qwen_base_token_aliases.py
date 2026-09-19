#!/usr/bin/env python3
"""Read-only weight and probability inspection, with optional HF GPU tail replay."""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys


def alias_indices(block, reference, tolerance):
    import torch
    return torch.where((block - reference).abs().amax(-1) <= tolerance)[0].tolist()


def distribution_stats(logits, aliases):
    import torch
    probs = logits.float().softmax(-1)
    values, indices = logits.float().sort(-1, descending=False)
    mask = values.softmax(-1).cumsum(-1) <= .1
    mask[:, -1] = False
    processed = values.masked_fill(mask, -torch.inf).scatter(-1, indices, values.masked_fill(mask, -torch.inf))
    nucleus = processed.softmax(-1)
    return {'alias_raw_mass': probs[0, aliases].sum().item(),
            'alias_nucleus_mass': nucleus[0, aliases].sum().item(),
            'entropy': -(probs * logits.float().log_softmax(-1)).sum().item(),
            'nucleus_size': int(torch.isfinite(processed).sum())}


def tail_probe_input(row):
    response = row['response_token_ids']
    if not response:
        raise ValueError('Missing response')
    return row['prompt_token_ids'] + response[:-1], response[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--tail-check', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and args.gpu is None:
        parser.error('CUDA inspection requires an explicit GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else str(args.gpu)
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from diagnose_qwen4_initial_rollouts import select_sources, delete_chat_markers
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((args.model / 'config.json').read_text())
    if not config.get('tie_word_embeddings'):
        raise ValueError('This inspection assumes tied input/output embeddings')
    index = args.model / 'model.safetensors.index.json'
    key = 'model.embed_tokens.weight'
    filename = json.loads(index.read_text())['weight_map'][key] if index.exists() else 'model.safetensors'
    aliases, exact = [], []
    with safe_open(args.model / filename, framework='pt', device='cpu') as f:
        embedding = f.get_slice(key)
        reference = embedding[151644:151645].float()
        for start in range(0, embedding.get_shape()[0], 4096):
            block = embedding[start:start + 4096].float()
            aliases.extend(start + i for i in alias_indices(block, reference, .0001))
            exact.extend(start + i for i in alias_indices(block, reference, 0.))
        differences = {str(i): (embedding[i:i+1].float() - reference).norm().item()
                       for i in [151643, 151644, 151645, 151667, 151668, 125576]}
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().to(args.device)
    with gzip.open(args.archive, 'rt') as f:
        archive = [json.loads(line) for line in f]
        sources = select_sources(archive)
    metadata = {'argv': sys.argv, 'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
                'device': args.device, 'gpu': args.gpu, 'dtype': 'bfloat16', 'attention': 'sdpa', 'threads': 8,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                'tolerance_max_abs': .0001, 'exact_alias_count': len(exact),
                'near_alias_count': len(aliases), 'near_alias_ids': aliases,
                'special_embedding_l2_differences': differences,
                'scope': 'Local weight geometry and first-token probability; not proof of initialization history'}
    with args.output.open('x', encoding='utf-8') as out:
        out.write(json.dumps({'metadata': metadata}) + '\n'); out.flush()
        for row in sources:
            prompts = {'historical': row['prompt_token_ids'],
                       'no_markers': delete_chat_markers(row['prompt_token_ids']),
                       'plain': tokenizer.encode(row['question'] + '\n\nSolution:\n', add_special_tokens=False)}
            for name, ids in prompts.items():
                with torch.inference_mode():
                    tokens = torch.tensor([ids], device=args.device)
                    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                   use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                probs = logits.softmax(-1)
                values, indices = probs.topk(5)
                original_token = row['archived_response_token_ids'][0]
                record = {'index': row['index'], 'prompt_mode': name, 'prompt_token_ids': ids,
                          **distribution_stats(logits, aliases),
                          'top5': [{'id': i, 'text': tokenizer.decode([i]), 'prob': p}
                                   for i, p in zip(indices[0].tolist(), values[0].tolist())],
                          'original_first_token_id': original_token,
                          'original_first_token_probability': probs[0, original_token].item()}
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                out.flush(); os.fsync(out.fileno())
                print(json.dumps({k: v for k, v in record.items() if k != 'prompt_token_ids'}, ensure_ascii=False), flush=True)
        if args.tail_check:
            seen = set()
            for row in archive:
                index = row['source_extra_info']['index']
                if row['finish_reason'] != 'length' or index in seen:
                    continue
                seen.add(index)
                ids, target = tail_probe_input(row)
                tokens = torch.tensor([ids], device=args.device)
                with torch.inference_mode():
                    logits = model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                   use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
                    probs = logits.softmax(-1)
                    continuation = model.generate(tokens, attention_mask=torch.ones_like(tokens),
                        do_sample=False, max_new_tokens=32, pad_token_id=tokenizer.eos_token_id)
                new_ids = continuation[0, len(ids):].tolist()
                record = {'check': 'historical_tail_hf_greedy', 'index': index, 'prefix_length': len(ids),
                          'original_target_id': target, 'target_probability': probs[0, target].item(),
                          'eos_probability': probs[0, tokenizer.eos_token_id].item(),
                          'generated_token_ids': new_ids, 'generated_text': tokenizer.decode(new_ids)}
                out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                out.flush(); os.fsync(out.fileno())
                print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

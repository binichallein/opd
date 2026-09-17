"""Versioned math prompt protocol and lossless on-policy trajectory archive."""

import gzip
import hashlib
import json
import os
from pathlib import Path

PROTOCOL = 'math_eval_nonthinking_v1'
DISABLED_SUFFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
ANSWER_INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'
STOP_TOKEN_IDS = [151643, 151645]
LLAMA_PROTOCOL = 'llama32_nonthinking_v1'
LLAMA_SUFFIX = '<|start_header_id|>assistant<|end_header_id|>\n\n'
LLAMA_STOP_IDS = [128001, 128008, 128009]
LLAMA_DATE = '18 Sep 2026'


def tokenizer_protocol(tokenizer):
    get_vocab = getattr(tokenizer, 'get_added_vocab', getattr(tokenizer, 'get_vocab', lambda: {}))
    vocab = get_vocab()
    if vocab.get('<|begin_of_text|>') == 128000:
        expected = {'<|end_of_text|>': 128001, '<|eom_id|>': 128008, '<|eot_id|>': 128009}
        if any(vocab.get(token) != value for token, value in expected.items()):
            raise ValueError('Incomplete Llama special-token mapping')
        return LLAMA_PROTOCOL
    return PROTOCOL


def valid_control_prefix(text, tokenizer, expected_protocol):
    actual = tokenizer_protocol(tokenizer)
    suffix = LLAMA_SUFFIX if actual == LLAMA_PROTOCOL else DISABLED_SUFFIX
    return actual == expected_protocol and text.endswith(suffix)


def evaluation_stop_ids(tokenizer):
    if tokenizer_protocol(tokenizer) == LLAMA_PROTOCOL:
        return list(LLAMA_STOP_IDS)
    stops = []
    for token in ('<|im_end|>', '<|endoftext|>'):
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if encoded:
                stops.append(encoded[0])
        except Exception:
            continue
    return stops


def evaluation_inputs(tokenizer, prompts):
    # Native Llama templates already include BOS; string tokenization may add it again.
    if tokenizer_protocol(tokenizer) == LLAMA_PROTOCOL:
        return [{'prompt_token_ids': tokenizer.encode(text, add_special_tokens=False)} for text in prompts]
    return prompts


def native_eval_record(output, expected_ids):
    if list(output.prompt_token_ids) != expected_ids:
        raise ValueError('Engine evaluation prompt IDs differ from the training protocol')
    sample = output.outputs[0]
    return {'prompt_token_ids': list(output.prompt_token_ids), 'response_token_ids': list(sample.token_ids),
            'num_generated_tokens': len(sample.token_ids), 'finish_reason': sample.finish_reason,
            'stop_reason': sample.stop_reason}


def math_prompt(question):
    prompt = question.strip()
    if not prompt:
        raise ValueError('Empty math question')
    if 'put your final answer' not in prompt:
        prompt += '\n\n' + ANSWER_INSTRUCTION
    return prompt


def render_nonthinking(tokenizer, content):
    kwargs = {'date_string': LLAMA_DATE} if tokenizer_protocol(tokenizer) == LLAMA_PROTOCOL else {}
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False, **kwargs,
    )


def generated_think_tags(ids):
    return any(token in (151667, 151668) for token in ids)


def generation_record(prompt_ids, sample, sampling, eos_token_id):
    fields = ('temperature', 'top_p', 'top_k', 'seed', 'max_tokens', 'n', 'ignore_eos', 'stop_token_ids')
    params = {key: getattr(sampling, key) for key in fields}
    params['stop_token_ids'] = list(params['stop_token_ids'] or [])
    return {
        'prompt_token_ids': list(map(int, prompt_ids)),
        'response_token_ids': list(map(int, sample.token_ids)),
        'finish_reason': sample.finish_reason, 'stop_reason': sample.stop_reason,
        'cumulative_logprob': sample.cumulative_logprob,
        'sampling': params, 'eos_token_id': eos_token_id,
    }


def length_mask(response, lengths):
    import torch
    if len(lengths) != response.shape[0] or any(n < 0 or n > response.shape[1] for n in lengths):
        raise ValueError('Invalid generation lengths')
    counts = torch.tensor(lengths, device=response.device)
    return (torch.arange(response.shape[1], device=response.device)[None, :] < counts[:, None]).long()


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported archive value: {type(value)}')


def save_rollouts(batch, tokenizer, directory, *, step, run_id, attempt_id, source_commit):
    protocol = tokenizer_protocol(tokenizer)
    if not attempt_id or Path(attempt_id).name != attempt_id or attempt_id in ('.', '..'):
        raise ValueError('A simple explicit attempt ID is required')
    folder = Path(directory) / attempt_id
    folder.mkdir(parents=True, exist_ok=True)
    step_dir = folder / f'step_{step:06d}'
    step_dir.mkdir(exist_ok=False)
    path = step_dir / 'raw.jsonl.gz'
    partial = path.with_suffix(path.suffix + '.partial')
    if path.exists():
        raise FileExistsError(path)
    metadata = batch.non_tensor_batch['generation_record']
    if len(metadata) != len(batch.batch['responses']):
        raise ValueError('Missing raw generations')
    stopped, thinking = 0, 0
    # An interrupted write is retained; a retry must use another attempt ID.
    with partial.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
            for i, generation in enumerate(metadata):
                record = dict(generation)
                prompt_ids, response_ids = record['prompt_token_ids'], record['response_token_ids']
                count = len(response_ids)
                prompt_width = batch.batch['prompts'].shape[1]
                attention = batch.batch['attention_mask'][i].detach().cpu()
                actual_prompt = batch.batch['prompts'][i].detach().cpu()[attention[:prompt_width].bool()].tolist()
                actual_response = batch.batch['responses'][i, :count].detach().cpu().tolist()
                mask = attention[prompt_width:].tolist()
                if actual_prompt != prompt_ids or actual_response != response_ids:
                    raise ValueError('Archive token IDs differ from the actual training batch')
                if mask != [1] * count + [0] * (len(mask) - count):
                    raise ValueError('Training mask differs from actual generation length')
                text = tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                has_think = generated_think_tags(response_ids) or '<think>' in text or '</think>' in text
                thinking += int(has_think)
                stopped += int(record['finish_reason'] == 'length')
                record.update({
                    'run_id': run_id, 'attempt_id': attempt_id, 'step': step, 'sample_index': i,
                    'source_commit': source_commit, 'protocol': protocol, 'enable_thinking': False,
                    'uid': str(batch.non_tensor_batch['uid'][i]),
                    'traj_uid': str(batch.non_tensor_batch['traj_uid'][i]),
                    'source_extra_info': batch.non_tensor_batch['source_extra_info'][i],
                    'prompt_text': tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    'response_text': text, 'response_length': count, 'response_mask': mask[:count],
                    'response_tensor_width': len(mask), 'padding_length': len(mask) - count,
                    'length_at_cap': count == record['sampling']['max_tokens'],
                    'generated_think_tags': bool(has_think),
                    'rollout_log_probs': batch.batch['rollout_log_probs'][i, :count].detach().cpu().tolist(),
                })
                compressed.write((json.dumps(record, ensure_ascii=False, allow_nan=False,
                                             default=_json_default) + '\n').encode('utf-8'))
        raw.flush()
        os.fsync(raw.fileno())
    # The exclusive step directory reserves this destination, including after a crash.
    # This filesystem supports rename but not hard links.
    partial.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.with_suffix(path.suffix + '.sha256').open('x') as handle:
        handle.write(f'{digest}  {path.name}\n')
    n = len(metadata)
    return {'rollout_archive/count': n, 'rollout_archive/length_stop_rate': stopped / max(n, 1),
            'rollout_archive/generated_think_tag_rate': thinking / max(n, 1)}

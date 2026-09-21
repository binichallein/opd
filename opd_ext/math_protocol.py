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
LLAMA_HISTORICAL_PROTOCOL = 'llama32_historical17_v1'
QWEN_HISTORICAL_PROTOCOL = 'qwen3_historical17_v1'
QWEN_COMPLETION_PROTOCOL = 'qwen3_completion_boxed_v1'
QWEN_INSTRUCT_PROTOCOL = 'qwen3_native_chat_no_thinking_boxed_v1'
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
    if expected_protocol == LLAMA_HISTORICAL_PROTOCOL:
        return actual == LLAMA_PROTOCOL and text.endswith(LLAMA_SUFFIX)
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


def completion_math_prompt(question):
    """Shared Base train/eval content; deliberately bypasses all chat templates."""
    question = question.strip()
    if not question or any(t in question for t in ('<|im_start|>', '<|im_end|>', '<think>', '</think>')):
        raise ValueError('Empty question or chat/thinking controls in completion input')
    return (question + '\n\nPlease solve the problem step by step and put '
            'the final answer in \\boxed{}.\n\nSolution:\n')


def completion_input_ids(tokenizer, question, *, max_prompt_length=2048):
    if tokenizer.bos_token_id is not None or tokenizer.eos_token_id != 151643:
        raise ValueError('Completion protocol requires the pinned Qwen Base tokenizer')
    if max_prompt_length < 2:
        raise ValueError('Invalid maximum prompt length')
    ids = tokenizer.encode(completion_math_prompt(question), add_special_tokens=False)
    if len(ids) > max_prompt_length:
        half = max_prompt_length // 2
        ids = ids[:half] + ids[-(max_prompt_length - half):]
    return ids


def qwen_instruct_math_prompt(question):
    """Literal user content from the accepted Qwen Instruct qualification."""
    return (question.strip() + '\n\nPlease solve the problem step by step and put '
            'the final answer in \\boxed{}.')


qwen_instruct_user_content = qwen_instruct_math_prompt


def qwen_instruct_render(tokenizer, question):
    if tokenizer.eos_token_id != 151645:
        raise ValueError('Qwen Instruct protocol requires native EOS151645')
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': qwen_instruct_math_prompt(question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not text.endswith('<think>\n\n</think>\n\n'):
        raise ValueError('Native non-thinking prompt must prefill the empty closed think block')
    return text


def qwen_instruct_input_ids(tokenizer, question, max_prompt_length=2048):
    """Native template IDs; only over-budget prompts undergo middle truncation."""
    if not 2 <= max_prompt_length <= 2048:
        raise ValueError('Qwen Instruct maximum prompt length must be between 2 and 2048')
    text = qwen_instruct_render(tokenizer, question)
    ids = list(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': qwen_instruct_math_prompt(question)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    ))
    if not ids or tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        raise ValueError('Native chat text/token IDs disagree')
    if len(ids) > max_prompt_length:
        half = max_prompt_length // 2
        ids = ids[:half] + ids[-(max_prompt_length - half):]
    return ids


def render_nonthinking(tokenizer, content):
    kwargs = {'date_string': LLAMA_DATE} if tokenizer_protocol(tokenizer) == LLAMA_PROTOCOL else {}
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False, **kwargs,
    )


def historical_math_prompt(question):
    """The historical 1.7B training instruction, without normalizing the question."""
    return (f'Math problem: {question}\n\n'
            'Please carefully reason through the math problem step by step and derive the correct answer. '
            'You must conduct reasoning inside <think> and </think> and give the final answer within \\boxed{}.\n')


def render_qwen_historical(tokenizer, content):
    if tokenizer_protocol(tokenizer) != PROTOCOL:
        raise ValueError('Historical Qwen training requires a Qwen tokenizer')
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)


def validate_qwen_historical_prompt(tokenizer, question, prompt_ids, *, max_prompt_length=None):
    text = render_qwen_historical(tokenizer, historical_math_prompt(question))
    expected = tokenizer.encode(text, add_special_tokens=False)
    if max_prompt_length is not None and len(expected) > max_prompt_length:
        if max_prompt_length < 2:
            raise ValueError('Invalid maximum prompt length')
        half = max_prompt_length // 2
        expected = expected[:half] + expected[-(max_prompt_length - half):]
    if list(prompt_ids) != expected:
        raise ValueError('Actual historical Qwen prompt IDs differ from reference')
    return {'protocol': QWEN_HISTORICAL_PROTOCOL, 'enable_thinking': None,
            'prompt_text': text, 'prompt_token_ids': expected, 'stop_token_ids': []}


def render_historical_training(tokenizer, content):
    """Render already-formatted historical user content; thinking remains unspecified."""
    if tokenizer_protocol(tokenizer) != LLAMA_PROTOCOL:
        raise ValueError('Historical Llama training requires a native Llama tokenizer')
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': content}], tokenize=False,
        add_generation_prompt=True, date_string=LLAMA_DATE,
    )


def validate_historical_training_prompt(tokenizer, question, prompt_ids, *, max_prompt_length=None,
                                        truncation='error'):
    """Check actual (unpadded) training IDs against the legacy instruction/native template."""
    text = render_historical_training(tokenizer, historical_math_prompt(question))
    if (not valid_control_prefix(text, tokenizer, LLAMA_HISTORICAL_PROTOCOL)
            or '<|im_start|>' in text or '<|im_end|>' in text):
        raise ValueError('Historical training prompt lacks native Llama control tokens')
    expected = tokenizer.encode(text, add_special_tokens=False)
    if max_prompt_length is not None:
        if max_prompt_length < 2:
            raise ValueError('Invalid maximum prompt length')
        if len(expected) > max_prompt_length:
            if truncation == 'middle':
                half = max_prompt_length // 2
                expected = expected[:half] + expected[-(max_prompt_length - half):]
            elif truncation == 'left':
                expected = expected[-max_prompt_length:]
            elif truncation == 'right':
                expected = expected[:max_prompt_length]
            else:
                raise ValueError('Historical training prompt exceeds maximum length')
    actual = list(prompt_ids)
    if actual != expected:
        raise ValueError('Actual historical training prompt IDs differ from the reference')
    if not actual or actual[0] != 128000 or actual.count(128000) != 1 or 128009 not in actual:
        raise ValueError('Historical training prompt requires one native BOS and native EOT')
    return {'protocol': LLAMA_HISTORICAL_PROTOCOL, 'enable_thinking': None,
            'prompt_text': text, 'prompt_token_ids': actual,
            'date_string': LLAMA_DATE, 'stop_token_ids': list(LLAMA_STOP_IDS)}


def generated_think_tags(ids, *, tokenizer=None, text=None):
    is_qwen = tokenizer is None or tokenizer_protocol(tokenizer) == PROTOCOL
    if text is None and tokenizer is not None:
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return ((is_qwen and any(token in (151667, 151668) for token in ids))
            or '<think>' in (text or '') or '</think>' in (text or ''))


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


def select_response_mask(response, lengths, legacy_mask, *, preserve_legacy=False):
    # Archiving the historical Qwen run must not silently change its EOS-mask semantics.
    if preserve_legacy:
        return legacy_mask
    return length_mask(response, lengths).to(legacy_mask.dtype)


def _json_default(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Unsupported archive value: {type(value)}')


def save_rollouts(batch, tokenizer, directory, *, step, run_id, attempt_id, source_commit, protocol=None):
    native_protocol = tokenizer_protocol(tokenizer)
    protocol = native_protocol if protocol is None else protocol
    expected_native = {LLAMA_HISTORICAL_PROTOCOL: LLAMA_PROTOCOL,
                       QWEN_HISTORICAL_PROTOCOL: PROTOCOL,
                       QWEN_INSTRUCT_PROTOCOL: PROTOCOL,
                       QWEN_COMPLETION_PROTOCOL: PROTOCOL}.get(protocol, protocol)
    if native_protocol != expected_native:
        raise ValueError('Archive protocol does not match the tokenizer')
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
                historical_qwen = protocol in (QWEN_HISTORICAL_PROTOCOL, QWEN_COMPLETION_PROTOCOL)
                if not historical_qwen and mask != [1] * count + [0] * (len(mask) - count):
                    raise ValueError('Training mask differs from actual generation length')
                text = tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                has_think = generated_think_tags(response_ids, tokenizer=tokenizer, text=text)
                thinking += int(has_think)
                stopped += int(record['finish_reason'] == 'length')
                record.update({
                    'run_id': run_id, 'attempt_id': attempt_id, 'step': step, 'sample_index': i,
                    'source_commit': source_commit, 'protocol': protocol,
                    'enable_thinking': None if protocol in (LLAMA_HISTORICAL_PROTOCOL, QWEN_HISTORICAL_PROTOCOL) else False,
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
                if historical_qwen:
                    record.update(mask_policy='historical_eos_mask', training_response_mask=mask,
                                  training_response_token_ids=batch.batch['responses'][i].detach().cpu().tolist(),
                                  training_rollout_log_probs=batch.batch['rollout_log_probs'][i].detach().cpu().tolist())
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

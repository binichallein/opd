from types import SimpleNamespace

import pytest

from opd_ext import math_protocol as protocol


def llama_tokenizer():
    return SimpleNamespace(get_vocab=lambda: {'<|begin_of_text|>': 128000,
        '<|end_of_text|>': 128001, '<|eom_id|>': 128008, '<|eot_id|>': 128009})


def test_llama_protocol_and_stops_are_native():
    tokenizer = llama_tokenizer()
    assert protocol.tokenizer_protocol(tokenizer) == 'llama32_nonthinking_v1'
    assert protocol.evaluation_stop_ids(tokenizer) == [128001, 128008, 128009]
    assert protocol.valid_control_prefix('question' + protocol.LLAMA_SUFFIX, tokenizer,
                                         protocol.LLAMA_PROTOCOL)
    assert not protocol.valid_control_prefix('question' + protocol.DISABLED_SUFFIX, tokenizer,
                                             protocol.LLAMA_PROTOCOL)
    assert not protocol.valid_control_prefix('question' + protocol.LLAMA_SUFFIX, tokenizer,
                                             protocol.PROTOCOL)


def test_qwen_stops_and_default_protocol_unchanged():
    tokenizer = SimpleNamespace(encode=lambda text, **kw: [151645] if text == '<|im_end|>' else [151643])
    assert protocol.tokenizer_protocol(tokenizer) == protocol.PROTOCOL
    assert protocol.evaluation_stop_ids(tokenizer) == [151645, 151643]


def test_partial_llama_mapping_rejected():
    tokenizer = SimpleNamespace(get_vocab=lambda: {'<|begin_of_text|>': 128000})
    with pytest.raises(ValueError, match='Llama'):
        protocol.evaluation_stop_ids(tokenizer)


def test_llama_eval_uses_explicit_ids_to_avoid_a_second_bos():
    tokenizer = llama_tokenizer()
    def encode(text, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        return [128000, 42]
    tokenizer.encode = encode
    assert protocol.evaluation_inputs(tokenizer, ['rendered']) == [{'prompt_token_ids': [128000, 42]}]
    assert protocol.evaluation_inputs(SimpleNamespace(), ['rendered']) == ['rendered']


def test_llama_template_date_is_fixed_across_days():
    tokenizer = llama_tokenizer()
    def render(messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['date_string'] == '18 Sep 2026'
        return 'rendered'
    tokenizer.apply_chat_template = render
    assert protocol.render_nonthinking(tokenizer, 'question') == 'rendered'


def test_eval_metadata_checks_actual_prompt_and_keeps_native_stop():
    output = SimpleNamespace(prompt_token_ids=[128000, 42], outputs=[SimpleNamespace(
        token_ids=[7, 128009], finish_reason='stop', stop_reason=128009)])
    item = protocol.native_eval_record(output, [128000, 42])
    assert item['num_generated_tokens'] == 2
    assert item['response_token_ids'] == [7, 128009]
    assert item['finish_reason'] == 'stop'
    with pytest.raises(ValueError, match='prompt'):
        protocol.native_eval_record(output, [128000, 128000, 42])

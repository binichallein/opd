import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def module(monkeypatch):
    path = ROOT / 'scripts/diagnose_llama_prompt.py'
    assert path.exists(), 'Missing native Llama prompt diagnostic'
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('llama_probe', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def test_historical_replay_checks_exact_input(monkeypatch):
    probe = module(monkeypatch)
    row = {'prompt_text': 'abc', 'prompt_token_ids': [97, 98, 99]}
    assert probe.prompt_variant(row, Tokenizer(), 'historical') == ('abc', [97, 98, 99])
    with pytest.raises(ValueError, match='archived'):
        probe.prompt_variant({**row, 'prompt_token_ids': [1]}, Tokenizer(), 'historical')


def test_remove_think_preserves_native_wrapper(monkeypatch):
    probe = module(monkeypatch)
    text = '<|begin_of_text|>native You must conduct reasoning inside <think> and </think> and give the final answer within \\boxed{}.<|eot_id|>assistant'
    result, ids = probe.prompt_variant({'prompt_text': text}, Tokenizer(), 'no_think')
    assert result == text.replace('conduct reasoning inside <think> and </think> and ', '')
    assert ids == [ord(c) for c in result]


def test_completion_preserves_user_content_and_one_bos(monkeypatch):
    probe = module(monkeypatch)
    text = probe.completion_text('two plus two')
    assert text.count('<|begin_of_text|>') == 1
    assert 'Math problem: two plus two' in text
    assert '\\boxed{}' in text
    assert text.endswith('\n\nSolution:\n')
    assert not any(t in text for t in ('<think>', '<|start_header_id|>', '<|im_start|>'))


def test_native_stops_do_not_report_qwen_boundaries(monkeypatch):
    probe = module(monkeypatch)
    stats = probe.response_stats([42, 128009], 'answer<|eot_id|>', 16384)
    assert stats['first_native_stop_index'] == 1
    assert stats['tokens_after_first_native_stop'] == 0
    assert 'first_im_end_index' not in stats
    assert stats['contains_boxed_marker'] is False


def test_pinned_assets_are_checked_not_just_rehashed(monkeypatch, tmp_path):
    probe = module(monkeypatch)
    f = tmp_path / 'config.json'
    f.write_text('original')
    expected = {str(f): probe.sha256(f)}
    assert probe.verify_files(expected) == expected
    f.write_text('modified')
    with pytest.raises(ValueError, match='asset'):
        probe.verify_files(expected)

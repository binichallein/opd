import importlib.util
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_token_truncation.py"
    assert path.is_file(), "The raw-trajectory diagnostic implementation is missing"
    spec = importlib.util.spec_from_file_location("truncation_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_touching_cap_is_not_an_engine_finish_reason():
    mod = load_module()
    info = mod.analyze_tokens([7, 8, 151643], max_tokens=3)
    assert info["length_at_cap"] is True
    assert info["first_eos_index"] == 2
    assert "finish_reason" not in info


def test_message_end_before_cap_is_retained():
    mod = load_module()
    info = mod.analyze_tokens([7, 151645, 9, 9, 9], max_tokens=5)
    assert info["first_im_end_index"] == 1
    assert info["tokens_after_first_im_end"] == 3
    assert info["first_eos_index"] is None


def test_periodic_tail_is_detected_without_confusing_short_answers():
    mod = load_module()
    info = mod.analyze_tokens([4, 5, 6] * 1000, max_tokens=3000)
    assert info["tail_period"] == 3
    assert info["tail_period_match"] == 1
    assert mod.analyze_tokens([4, 5, 6], max_tokens=100)["tail_period"] is None


def test_nonrepeating_tail_is_not_labelled_a_loop():
    mod = load_module()
    info = mod.analyze_tokens(list(range(3000)), max_tokens=3000)
    assert info["tail_period"] is None
    assert info["tail_repeated_4gram_fraction"] == 0


def test_raw_writer_refuses_overwrite_and_keeps_unicode(tmp_path):
    mod = load_module()
    path = tmp_path / "raw.jsonl"
    mod.write_records(path, [{"token_ids": [151645], "text": "<|im_end|>"}])
    with pytest.raises(FileExistsError):
        mod.write_records(path, [])
    assert "151645" in path.read_text()


def test_duplicate_metadata_keys_do_not_replace_engine_reason():
    mod = load_module()
    record = mod.build_record(
        {"index": 12, "prompt_token_ids": [5], "prompt": "question"},
        [7, 151645], "answer<|im_end|>", "stop", 151645, -1.0,
        label="test", seed=21, max_tokens=2,
    )
    assert record["finish_reason"] == "stop"
    assert record["length_at_cap"] is True
    assert record["response_token_ids"] == [7, 151645]
    assert record["response_raw"].endswith("<|im_end|>")


def test_no_think_condition_changes_only_the_instruction():
    mod = load_module()
    text = ('Math problem: Find x.\n\nPlease carefully reason through the math problem '
            'step by step and derive the correct answer. You must conduct reasoning '
            'inside <think> and </think> and give the final answer within \\boxed{}.\n')
    changed = mod.without_think_requirement(text)
    assert changed == text.replace('conduct reasoning inside <think> and </think> and ', '')
    assert 'Math problem: Find x.' in changed
    assert 'You must give the final answer within \\boxed{}.' in changed
    with pytest.raises(ValueError):
        mod.without_think_requirement('unrecognized template')

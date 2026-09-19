import importlib.util
import math
from pathlib import Path

import pytest


def module(monkeypatch):
    root = Path(__file__).parents[1]
    path = root / 'scripts/audit_llama_prompt_probe.py'
    assert path.exists(), 'Missing prompt probe audit'
    monkeypatch.syspath_prepend(str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('audit_probe', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def record():
    return {'request_id': 'a', 'index': 4, 'seed': 21, 'prompt_token_ids': [128000, 8],
            'response_token_ids': [10, 128009], 'response_length': 2,
            'sampled_logprobs': [-1., -.1], 'finish_reason': 'stop'}


def test_audit_accepts_exact_complete_records(monkeypatch):
    probe = module(monkeypatch)
    assert probe.validate_records([record()], [record()], 16384) is None


@pytest.mark.parametrize('change', [{'seed': 22}, {'prompt_token_ids': [128000, 9]},
                                   {'sampled_logprobs': [math.nan, 0.]},
                                   {'response_token_ids': [128009, 10]},
                                   {'finish_reason': 'length'}])
def test_audit_rejects_corrupt_or_unmatched_records(monkeypatch, change):
    probe = module(monkeypatch)
    with pytest.raises(ValueError):
        probe.validate_records([{**record(), **change}], [record()], 16384)


def test_audit_rejects_missing_and_duplicate_requests(monkeypatch):
    probe = module(monkeypatch)
    with pytest.raises(ValueError):
        probe.validate_records([], [record()], 16384)
    with pytest.raises(ValueError):
        probe.validate_records([record(), record()], [record()], 16384)

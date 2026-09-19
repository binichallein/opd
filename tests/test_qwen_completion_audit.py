import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def test_qwen_audit_catches_missing_inputs_and_ignored_eos():
    assert importlib.util.find_spec('audit_qwen_completion') is not None
    from audit_qwen_completion import validate_records
    req = {'request_id': 'a', 'index': 1, 'seed': 21, 'prompt_token_ids': [9]}
    row = {**req, 'response_token_ids': [7, 151643], 'sampled_logprobs': [-1., -.2],
           'response_length': 2, 'finish_reason': 'stop'}
    validate_records([row], [req], 16384)
    with pytest.raises(ValueError):
        validate_records([], [req], 16384)
    row['response_token_ids'] = [151643, 7]
    with pytest.raises(ValueError):
        validate_records([row], [req], 16384)

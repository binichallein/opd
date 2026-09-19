import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def gate():
    assert importlib.util.find_spec('run_qwen_completion_gate') is not None
    return importlib.import_module('run_qwen_completion_gate')


def test_gate_is_bounded_paired_and_starts_from_original(tmp_path):
    m = gate()
    for variant in ('block3_mean', 'token_opd'):
        first = m.training_env(tmp_path/'runtime', 'abc', tmp_path/'run', variant, 1)
        second = m.training_env(tmp_path/'runtime', 'abc', tmp_path/'run', variant, 2)
        assert first['STUDENT_MODEL'] == second['STUDENT_MODEL'] == str(m.assets.STUDENT)
        assert first['TOTAL_TRAINING_STEPS'] == second['TOTAL_TRAINING_STEPS'] == '2'
        assert first['ENV_SEED'] == second['ENV_SEED'] == '21'
        assert first['OPD_REQUEST_SEED_RULE'] == second['OPD_REQUEST_SEED_RULE'] == m.SEED_RULE
        assert first['OPD_PROMPT_PROTOCOL'] == second['OPD_PROMPT_PROTOCOL'] == m.PROTOCOL
        assert first['RESUME_MODE'] == 'disable' and first['STOP_AFTER_STEP'] == '1'
        assert second['RESUME_MODE'] == 'resume_path'
        assert second['RESUME_FROM_PATH'] == str(tmp_path/'run'/variant/'checkpoints/global_step_1')
        assert first['OPD_DIAG_INTERVAL'] == second['OPD_DIAG_INTERVAL'] == '1'
        assert first['MAX_RESPONSE_LENGTH'] == '16384'
    with pytest.raises(ValueError):
        m.training_env(tmp_path, 'abc', tmp_path, 'token_opd', None)


def test_gate_row_validation_detects_fixed_seed_and_prompt_drift():
    m = gate()
    from opd_ext.request_seeds import request_identities, request_seed
    identity = request_identities([{'index': 1, 'question': 'Q'}], group_size=1, global_seed=21, step=2)[0]
    row = {'protocol': m.PROTOCOL, 'enable_thinking': False, 'step': 2, 'mask_policy': 'historical_eos_mask',
           'eos_token_id': 151643, 'request_identity': identity, 'request_turn': 0,
           'sampling': {'seed': request_seed(identity, turn=0), 'temperature': 1., 'top_p': .9,
                        'top_k': -1, 'max_tokens': 16384, 'n': 1, 'ignore_eos': False, 'stop_token_ids': []}}
    m.validate_seed_settings(row, identity, 2)
    row['sampling']['seed'] = 21
    with pytest.raises(ValueError):
        m.validate_seed_settings(row, identity, 2)

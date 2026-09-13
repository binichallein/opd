import importlib.util
from pathlib import Path


def load():
    path=Path(__file__).resolve().parents[1]/'scripts/run_qwen06_nonthinking.py'
    assert path.exists(), 'The new standalone Token launcher must exist'
    spec=importlib.util.spec_from_file_location('nonthinking_launcher',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_new_run_is_token_only_and_preserves_protocol_controls(tmp_path):
    mod=load()
    env=mod.training_env(tmp_path/'runtime','abc',tmp_path/'run')
    assert env['VARIANT']=='token_opd'
    assert env['TOTAL_TRAINING_STEPS']=='200'
    assert env['RESUME_MODE']=='disable'
    assert env['OPD_PROMPT_PROTOCOL']=='math_eval_nonthinking_v1'
    assert env['ROLLOUT_ATTEMPT_ID']=='formal'
    assert env['LOSSLESS_ROLLOUT_DIR']==str(tmp_path/'run/token_opd/rollouts')
    assert env['DIAGNOSTIC_SAVE_STEPS']=='50,100,200'
    assert env['OPD_DIAG_INTERVAL']=='5'
    assert 'block3' not in str(mod.RUN_ROOT)


def test_probe_resume_uses_a_distinct_attempt_without_changing_input_seed(tmp_path):
    mod=load()
    first=mod.training_env(tmp_path/'runtime','abc',tmp_path/'probe',1)
    second=mod.training_env(tmp_path/'runtime','abc',tmp_path/'probe',2)
    assert first['ROLLOUT_ATTEMPT_ID']=='probe1'
    assert second['ROLLOUT_ATTEMPT_ID']=='probe2'
    assert second['RESUME_MODE']=='resume_path'
    assert first['ENV_SEED']==second['ENV_SEED']=='21'

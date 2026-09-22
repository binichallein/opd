import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def subject():
    assert (SCRIPTS / 'recover_qwen17_instruct_token.py').is_file(), 'Token-only recovery controller missing'
    return importlib.import_module('recover_qwen17_instruct_token')


def test_only_train_token_then_four_evaluations():
    s = subject()
    reused = {n: {'passed': True} for n in s.REUSED_NAMES}
    events = []
    def train():
        events.append('train_token')
        return {'passed': True}
    def evaluate(name):
        events.append(name)
        return {'passed': True}
    results = s.execute_remaining(reused, train, evaluate)
    assert events == ['train_token'] + [f'token_opd_step{i}' for i in (200,150,100,50)]
    assert len(results) == 9
    assert reused == {n: {'passed': True} for n in s.REUSED_NAMES}


@pytest.mark.parametrize('failure', ['reused', 'training', 'token_opd_step150'])
def test_failure_stops_without_retries(failure):
    s = subject()
    events = []
    reused = {n: {'passed': failure != 'reused'} for n in s.REUSED_NAMES}
    def train():
        events.append('training')
        return {'passed': failure != 'training'}
    def evaluate(name):
        events.append(name)
        return {'passed': failure != name}
    with pytest.raises(ValueError):
        s.execute_remaining(reused, train, evaluate)
    assert events == {'reused': [], 'training': ['training'], 'token_opd_step150':
                      ['training','token_opd_step200','token_opd_step150']}[failure]


def test_recovery_card_allows_only_output_relocation():
    s = subject()
    old = s.q.expected_card(s.SOURCE, 'token_opd', s.TRAIN_COMMIT)
    new = s.q.expected_card(s.RUN_ROOT, 'token_opd', s.TRAIN_COMMIT)
    s.validate_relocated_card(old, new)
    for key, val in [('learning_rate', 6e-6), ('seed',7), ('resume_mode','auto'),
                     ('total_training_steps',100), ('diagnostic_save_steps','50,100,200')]:
        with pytest.raises(ValueError):
            s.validate_relocated_card(old, {**new,key:val})


def test_formal_env_is_original_initialization_and_four_milestones():
    s = subject()
    env = s.formal_env()
    old = s.q.training_env(s.TRAIN_RUNTIME,s.TRAIN_COMMIT,s.RUN_ROOT,'token_opd')
    assert {k for k in env if env[k] != old[k]} == {'LOCAL_CACHE_ROOT'}
    assert env['TOTAL_TRAINING_STEPS'] == '200'
    assert env['DIAGNOSTIC_SAVE_STEPS'] == '50,100,150,200'
    assert env['RESUME_MODE'] == 'disable' and env['RESUME_FROM_PATH'] == ''


def test_failed_attempt_must_have_zero_training_artifacts(tmp_path):
    s = subject()
    j = s.jobs
    run = tmp_path/'token_opd'
    (run/'logs').mkdir(parents=True)
    j.write_json(tmp_path/'queue_manifest.json', {'source_commit':s.TRAIN_COMMIT})
    j.write_json(tmp_path/'queue_state.json', {'status':'failed','job':str(run)})
    (run/'exit_code.txt').write_text('1')
    (run/'logs/nohup.log').write_text('The current node timed out during startup.')
    s.validate_failure(tmp_path)
    (run/'checkpoints/global_step_1').mkdir(parents=True)
    with pytest.raises(ValueError,match='training artifacts'):
        s.validate_failure(tmp_path)


def test_source_and_runtime_are_separate_and_pinned():
    s = subject()
    assert s.RUN_ROOT != s.SOURCE
    assert s.TRAIN_RUNTIME.name == 'be736b5fac4f26ff59f4e2c21abafc456c2503f6'
    assert str(s.CACHE).startswith('/dev/shm/')
    assert s.CACHE != s.q.CACHE

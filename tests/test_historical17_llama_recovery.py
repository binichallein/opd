import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/recover_historical17_llama.py'
    spec = importlib.util.spec_from_file_location('h17_recovery', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_short_cache_covers_actual_training_tmp_and_maximum_pid():
    m = module()
    env = m.llama.training_env(Path('/runtime'), 'a' * 40, m.RUN_ROOT, 'token_opd',
                               training_protocol=m.historical.LEGACY_PROTOCOL, cache=m.CACHE)
    tmp = Path(env['LOCAL_CACHE_ROOT']) / 'tmp'
    assert tmp == m.training_tmp()
    assert m.check_socket_budget(tmp) <= 107
    with pytest.raises(ValueError, match='107'):
        m.check_socket_budget(Path('/limx_embap/tos/h17/0918v2/llama/train/tmp'))
    assert env['RESUME_MODE'] == 'disable' and env['RESUME_FROM_PATH'] == ''


def failed_attempt(tmp_path, m):
    f = tmp_path / 'llama32/queue_jobs/token_opd_probe1'
    (f / 'logs').mkdir(parents=True)
    (f / 'exit_code.txt').write_text('1\n')
    (f / 'logs/job.log').write_text('OSError: AF_UNIX path length cannot exceed 107 bytes\n')
    (tmp_path / 'queue_state.json').write_text(json.dumps({'status': 'failed', 'job': str(f)}))
    (tmp_path / 'queue_manifest.json').write_text(json.dumps({'source_commit': m.OLD_COMMIT}))


def test_recovery_requires_reviewed_failure_before_any_training(tmp_path):
    m = module()
    failed_attempt(tmp_path, m)
    m.validate_failure(tmp_path)
    p = tmp_path / 'llama32/probes/token_opd/checkpoints/global_step_1'
    p.mkdir(parents=True)
    with pytest.raises(ValueError, match='training artifacts'):
        m.validate_failure(tmp_path)


def test_recovery_rejects_unknown_failures(tmp_path):
    m = module()
    failed_attempt(tmp_path, m)
    (tmp_path / 'llama32/queue_jobs/token_opd_probe1/logs/job.log').write_text('CUDA out of memory')
    with pytest.raises(ValueError, match='socket'):
        m.validate_failure(tmp_path)


def test_reuses_accepted_initial_eval_without_generating_it_again():
    m = module()
    calls = []
    initial = {'passed': True, 'source': 'old_initial_eval'}
    def evaluate(name):
        calls.append(name)
        return {'passed': True}
    result = m.execute_remaining(initial, evaluate, lambda name: calls.append('train_' + name))
    assert result['student_base'] is initial
    assert calls == ['train_token_opd', 'token_opd_step200', 'token_opd_step100', 'token_opd_step50',
                     'train_block3_mean', 'block3_mean_step200', 'block3_mean_step100', 'block3_mean_step50']
    calls.clear()
    with pytest.raises(ValueError):
        m.execute_remaining(initial, lambda name: {'passed': False}, lambda name: calls.append(name))
    assert calls == ['token_opd']


def test_recovery_retains_original_eval_and_training_runtime_identity():
    m = module()
    assert m.TRAIN_RUNTIME == m.llama.ROOT / 'deployments' / m.OLD_COMMIT
    assert m.RUN_ROOT != m.SOURCE
    assert m.SOURCE == m.historical.RUN_ROOT

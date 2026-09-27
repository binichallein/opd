import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / 'scripts/recover_acp_base17_evaluation.py'
    spec = importlib.util.spec_from_file_location('recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_missing_tasks_change_in_command():
    recovery = load()
    original = ['python', 'eval.py', '--tasks', 'math500', 'aime24', 'aime25', 'amc23',
                '--n', '8', '--eval-seed', '21', '--retain-rollouts']
    assert recovery.missing_command(original) == [
        'python', 'eval.py', '--tasks', 'aime25', 'amc23', '--n', '8',
        '--eval-seed', '21', '--retain-rollouts']
    assert original[3] == 'math500'
    with pytest.raises(ValueError):
        recovery.missing_command(['python', '--tasks', 'math500', '--n', '8'])


def test_seven_accepted_results_required():
    recovery = load()
    models = {key: {'passed': True} for key in recovery.ACCEPTED}
    report = {'passed': True, 'complete': False, 'models': models}
    assert recovery.accepted_results(report) == models
    models['token_opd_step25'] = {'passed': True}
    with pytest.raises(ValueError):
        recovery.accepted_results(report)
    del models['token_opd_step25']
    models['token_opd_step100']['passed'] = False
    with pytest.raises(ValueError):
        recovery.accepted_results(report)


def test_config_allows_only_task_and_archive_location_changes():
    recovery = load()
    old = dict(tasks=list(recovery.TASKS), rollout_archive_dir='/old',
               model_path='/same', prompt_protocol='same', n=8, eval_seed=21)
    new = dict(old, tasks=list(recovery.MISSING), rollout_archive_dir='/new')
    recovery.validate_partial_config(old, new)
    for key, value in [('eval_seed', 22), ('prompt_protocol', 'other'), ('model_path', '/other')]:
        with pytest.raises(ValueError):
            recovery.validate_partial_config(old, dict(new, **{key: value}))


def test_copy_is_exclusive_and_hash_checked(tmp_path):
    recovery = load()
    source, dest = tmp_path / 'old', tmp_path / 'new'
    source.write_bytes(b'original output\n')
    digest = recovery.sha256(source)
    recovery.copy_verified(source, dest, digest)
    assert dest.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        recovery.copy_verified(source, dest, digest)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):
        recovery.copy_verified(source, tmp_path / 'another', digest)

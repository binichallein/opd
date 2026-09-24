import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    path = ROOT/'scripts/run_historical_pair_n1.py'
    assert path.exists(), 'The 100-step block-first queue is not implemented'
    spec = importlib.util.spec_from_file_location('historical_n1_queue_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_exact_order(module):
    calls = []
    module.execute_ordered(lambda v: calls.append(('train', v)) or {'passed': True},
                           lambda v: calls.append(('eval', v)) or {'passed': True})
    assert calls == [('train', 'block3_mean'), ('eval', 'block3_mean_step100'),
        ('eval', 'block3_mean_step75'), ('eval', 'block3_mean_step50'),
        ('eval', 'block3_mean_step25'), ('train', 'token_opd'),
        ('eval', 'token_opd_step100'), ('eval', 'token_opd_step75'),
        ('eval', 'token_opd_step50'), ('eval', 'token_opd_step25')]


def test_checkpoint_schedule_is_shared_with_preparer(module):
    assert module.preparation.SAVE_STEPS == (25, 50, 75, 100)
    assert module.STEPS == tuple(reversed(module.preparation.SAVE_STEPS))


@pytest.mark.parametrize('variant', ['block3_mean', 'token_opd'])
def test_nested_ray_gate_socket_stays_within_unix_limit(module, variant):
    from recover_historical17_llama import check_socket_budget
    check_socket_budget(module.ray_gate_cache(variant)/'gate/tmp')
    check_socket_budget(module.CACHE/variant/'train/tmp')


def test_eval_failure_prevents_token_training(module):
    trained = []
    with pytest.raises(ValueError):
        module.execute_ordered(lambda v: trained.append(v) or {'passed': True},
                               lambda v: {'passed': False})
    assert trained == ['block3_mean']


@pytest.mark.parametrize('variant', ['block3_mean', 'token_opd'])
@pytest.mark.parametrize('step', [1, 2])
def test_probe_keeps_100_step_horizon_and_new_sampling(module, variant, step):
    import run_historical_component_ablation as historical
    run = historical.configured_runner()
    env = module.probe_env(run, Path('/runtime'), 'abc', Path('/root'), variant, step)
    assert env['TOTAL_TRAINING_STEPS'] == '100'
    assert env['TRAIN_BATCH_SIZE'] == '32' and env['ROLLOUT_GROUP_SIZE'] == '1'
    assert env['STOP_AFTER_STEP'] == str(step)
    assert env['RESUME_FROM_PATH'] == (f'/root/probes/{variant}/checkpoints/global_step_1' if step == 2 else '')
    assert env['OPD_PROMPT_PROTOCOL'] == 'qwen3_historical17_v1'
    assert env['OPD_REQUEST_SEED_RULE'] == 'legacy'


def batch():
    sources = [{'index': i, 'question': f'q{i}'} for i in range(32)]
    rows = [dict(source_extra_info=s, sample_index=i, traj_uid=f'id{i}', step=1,
                 prompt_token_ids=[i], sampling={'n': 1, 'seed': 21}) for i, s in enumerate(sources)]
    return sources, rows


def test_32x1_batch_rejects_4x8_or_wrong_order(module):
    sources, rows = batch()
    assert module.batch_fingerprint(rows, sources, 1)
    with pytest.raises(ValueError):
        module.batch_fingerprint([rows[i//8] for i in range(32)], sources, 1)
    with pytest.raises(ValueError):
        module.batch_fingerprint(list(reversed(rows)), sources, 1)


def test_paired_audit_requires_all_100_steps(module):
    steps = [dict(step=s, count=32, paired_input_sha256=f'h{s}') for s in range(1, 101)]
    left = dict(passed=True, total_rollouts=3200, steps=steps)
    assert module.validate_paired_rollouts(left, left)['passed']
    with pytest.raises(ValueError):
        module.validate_paired_rollouts(left, dict(left, steps=steps[:-1]))
    right = dict(left, steps=[dict(s) for s in steps])
    right['steps'][10]['paired_input_sha256'] = 'wrong'
    with pytest.raises(ValueError):
        module.validate_paired_rollouts(left, right)

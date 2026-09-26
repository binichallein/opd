import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    path = ROOT / 'scripts/run_acp_component_ablation.py'
    assert path.exists(), 'ACP component queue is missing'
    spec = importlib.util.spec_from_file_location('acp_component_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('variant,mode', [('adv3', 'adv_only'), ('scale3', 'token_scale')])
@pytest.mark.parametrize('probe', [None, 1, 2])
def test_component_contract_and_paths(module, variant, mode, probe):
    run = module.configured_runner()
    root = module.RUN_ROOT
    env = run.training_env(Path('/runtime'), 'a' * 40, root, variant, probe)
    card = run.expected_card(Path('/runtime'), 'a' * 40, root, variant, probe)
    assert env['VARIANT'] == variant and card['variant'] == variant
    assert card['opd_block_ablation'] == mode
    assert card['opd_block_size'] == 3 and card['opd_block_advantage_mode'] == 'mean'
    assert env['TOTAL_TRAINING_STEPS'] == '100'
    assert env['TRAIN_BATCH_SIZE'] == '32' and env['ROLLOUT_GROUP_SIZE'] == '1'
    assert env['PPO_MINI_BATCH_SIZE'] == '32' and env['OPD_REQUEST_SEED_RULE'] == 'legacy'
    assert env['OPD_PROMPT_PROTOCOL'] == run.PROTOCOL
    assert env['LEARNING_RATE'] == '2e-6'
    assert env['RESUME_FROM_PATH'] == (str(root / variant / 'checkpoints/global_step_1') if probe == 2 else '')
    assert env['OPD_DIAG_OUTPUT_DIR'] == str(root / variant / 'diagnostics')
    assert env['LOSSLESS_ROLLOUT_DIR'] == str(root / variant / 'rollouts')
    assert card['total_training_steps'] == 100
    assert card['diagnostic_save_steps'] == {None:'25,50,75,100', 1:'1', 2:'1,2'}[probe]
    assert str(module.BASELINE_ROOT) not in env['RUN_ROOT']


def test_order_and_eval_failure_stop_queue(module):
    calls = []
    module.execute_ordered(lambda v: calls.append(('train', v)) or {'passed': True},
                           lambda v: calls.append(('eval', v)) or {'passed': True})
    assert calls == [(kind, name) for v in ('adv3', 'scale3') for kind, name in
                     [('train', v), *[('eval', f'{v}_step{s}') for s in (100, 75, 50, 25)]]]
    trained = []
    with pytest.raises(ValueError):
        module.execute_ordered(lambda v: trained.append(v) or {'passed': True}, lambda _: {'passed': False})
    assert trained == ['adv3']


def test_diagnostic_granularity_matches_actual_objective():
    from opd_ext.diagnostics import component_diagnostic_sizes
    assert component_diagnostic_sizes(3, 'legacy') == (3, 3)
    assert component_diagnostic_sizes(3, 'adv_only') == (3, 1)
    assert component_diagnostic_sizes(3, 'joint_tokenmean') == (3, 3)
    assert component_diagnostic_sizes(3, 'token_scale') == (1, 1)
    with pytest.raises(ValueError):
        component_diagnostic_sizes(3, 'wrong')


def test_baseline_alignment_rejects_recipe_drift(module):
    source = dict(seed=21, total_training_steps=100, learning_rate=2e-6,
                  request_seed_rule='legacy', source_commit='a', variant='adv3')
    module.validate_baseline_card(source, dict(source, variant='token_opd', source_commit='b'))
    for key in ('seed', 'total_training_steps', 'learning_rate', 'request_seed_rule'):
        with pytest.raises(ValueError):
            module.validate_baseline_card(source, dict(source, **{key:'changed'}))


def test_component_evaluation_paths_and_protocol(module):
    run = module.configured_runner()
    run.configure()
    for variant in module.MODES:
        for step in (25, 50, 75, 100):
            checkpoint, model = run.q.model_paths(module.RUN_ROOT, f'{variant}_step{step}')
            assert checkpoint == module.RUN_ROOT / variant / f'checkpoints/global_step_{step}/actor'
            command = run.q.evaluation_command(Path('/runtime'), model, Path('/out'))
            assert command[command.index('--prompt-protocol')+1] == run.PROTOCOL
            assert '--retain-rollouts' in command

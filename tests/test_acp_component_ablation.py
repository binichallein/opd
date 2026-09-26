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


@pytest.mark.parametrize('mode', ['adv_only', 'token_scale', 'legacy', 'joint_tokenmean'])
@pytest.mark.parametrize('length', [5, 6, 16384])
def test_actual_trainer_ratio_snapshot_expansion(mode, length):
    import ast
    from types import SimpleNamespace
    import torch
    from opd_ext.diagnostics import (component_diagnostic_sizes,
        compute_block_ratio_diagnostics, expand_block_values_to_tokens)

    path = ROOT / 'external/revisiting_opd/verl/trainer/ppo/ray_trainer_multitask.py'
    tree = ast.parse(path.read_text())
    segment = None
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(body, list):
            continue
        for index, item in enumerate(body[:-1]):
            following = body[index+1]
            if (isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'block_size' for t in item.targets)
                    and isinstance(following, ast.If) and isinstance(following.test, ast.Name)
                    and following.test.id == 'window_kwargs'
                    and 'expand_block_values_to_tokens' in ast.unparse(following)):
                segment = ast.Module(body=[item, following], type_ignores=[])
    assert segment is not None, 'Must exercise the real trainer snapshot expansion'
    mask = torch.ones(2, length)
    mask[1, -2:] = 0
    old = torch.zeros_like(mask)
    current = torch.ones_like(mask) * .1
    _, ratio_size = component_diagnostic_sizes(3, mode)
    ratio = compute_block_ratio_diagnostics(old, current, mask, ratio_size, .2, .2)
    env = dict(torch=torch, int=int, ratio_size=ratio_size, window_kwargs={}, block_ratio=ratio,
        batch=SimpleNamespace(batch={'response_mask':mask}),
        self=SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={'opd_block_size':3}))),
        expand_block_values_to_tokens=expand_block_values_to_tokens)
    exec(compile(ast.fix_missing_locations(segment), str(path), 'exec'), env)
    expected, valid = expand_block_values_to_tokens(ratio['block_log_ratio'].abs(), mask, ratio_size)
    torch.testing.assert_close(env['expanded_log_ratio'], expected)
    assert env['expanded_finite'].shape == mask.shape
    assert env['expanded_outside_clip'].shape == mask.shape
    assert torch.equal(env['response_valid'], valid)

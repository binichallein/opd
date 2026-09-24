import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    path = ROOT / 'scripts/prepare_historical_pair_n1.py'
    assert path.exists(), 'Token/legacy Block3 preparation is not implemented'
    spec = importlib.util.spec_from_file_location('historical_pair_n1_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('variant', ['token_opd', 'block3_mean'])
def test_real_recipe_uses_requested_pair_and_budget(module, variant):
    import run_historical_component_ablation as historical
    run = historical.configured_runner()
    env = module.training_env(run, Path('/runtime'), 'abc', Path('/new'), variant)
    assert env['VARIANT'] == variant
    assert env['TRAIN_BATCH_SIZE'] == '32'
    assert env['ROLLOUT_GROUP_SIZE'] == '1'
    assert env['PPO_MINI_BATCH_SIZE'] == '32'
    assert env['TOTAL_TRAINING_STEPS'] == '100'
    assert env['DIAGNOSTIC_SAVE_STEPS'] == '50,100'
    assert env['STUDENT_MODEL'].endswith('/Qwen3-1.7B-Base')
    assert env['MATH_TEACHER'].endswith('/Qwen3-4B-Base-GRPO')
    assert env['OPD_PROMPT_PROTOCOL'] == 'qwen3_historical17_v1'
    assert env['OPD_REQUEST_SEED_RULE'] == 'legacy'
    assert env['ENV_SEED'] == '21'
    assert env['RESUME_MODE'] == 'disable' and env['RESUME_FROM_PATH'] == ''
    assert env['PREPARE_ONLY'] == 'true'
    assert env['LOSSLESS_ROLLOUT_DIR'] == f'/new/{variant}/rollouts'
    assert env['OPD_DIAG_INTERVAL'] == '5'


def test_ablation_variants_not_allowed(module):
    for variant in ('adv3', 'joint3'):
        with pytest.raises(ValueError):
            module.training_env(None, Path('/runtime'), 'abc', Path('/new'), variant)


@pytest.mark.parametrize('variant,size,mode', [('token_opd', 1, 'sum'), ('block3_mean', 3, 'mean')])
def test_full_legacy_loss_and_strict_pair_validation(module, variant, size, mode):
    import run_historical_component_ablation as historical
    run = historical.configured_runner()
    old = run.expected_card(Path('/old'), 'adv3', 'abc')
    # Actual source cards carry additional fields; they must remain unchanged too.
    old['extra_configuration'] = 'retained'
    card = module.expected_card(old, Path('/new'), variant)
    assert card['opd_block_ablation'] == 'legacy'
    assert card['opd_block_size'] == size and card['opd_block_advantage_mode'] == mode
    assert card['total_training_steps'] == 100
    assert card['diagnostic_save_steps'] == '50,100'
    module.validate_card(old, card, Path('/new'), variant)
    for changed in ({'opd_block_ablation': 'adv_only'}, {'seed': 42},
                    {'opd_prompt_protocol': 'qwen3_completion_boxed_v1'},
                    {'total_training_steps': 200}, {'extra_configuration': 'changed'}):
        with pytest.raises(ValueError):
            module.validate_card(old, dict(card, **changed), Path('/new'), variant)

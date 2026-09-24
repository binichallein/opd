import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module():
    path = ROOT / 'scripts/prepare_historical_single_rollout.py'
    spec = importlib.util.spec_from_file_location('single_rollout_prepare_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('variant', ['adv3', 'joint3'])
def test_only_sampling_changes(module, variant):
    old = {'TRAIN_BATCH_SIZE': '4', 'ROLLOUT_GROUP_SIZE': '8',
           'TOTAL_TRAINING_STEPS': '200', 'PPO_MINI_BATCH_SIZE': '32',
           'OPD_REQUEST_SEED_RULE': 'legacy', 'RESUME_MODE': 'disable',
           'VARIANT': variant, 'PREPARE_ONLY': 'true'}
    new = module.single_rollout_env(old)
    assert old['TRAIN_BATCH_SIZE'] == '4'
    assert {k: v for k, v in new.items() if old.get(k) != v} == {
        'TRAIN_BATCH_SIZE': '32', 'ROLLOUT_GROUP_SIZE': '1'}


def test_preparation_rejects_unexpected_old_recipe(module):
    with pytest.raises(ValueError):
        module.single_rollout_env({'TRAIN_BATCH_SIZE': '32'})


def test_card_diff_rejects_unrequested_changes(module):
    old = dict(train_batch_size=4, rollout_group_size=8, seed=21,
               diagnostic_output_dir='/old/diagnostics', lossless_rollout_dir='/old/rollouts')
    new = dict(old, train_batch_size=32, rollout_group_size=1,
               diagnostic_output_dir='/new/diagnostics', lossless_rollout_dir='/new/rollouts')
    diff = module.validate_card_diff(old, new)
    assert set(diff) == {'train_batch_size', 'rollout_group_size',
                         'diagnostic_output_dir', 'lossless_rollout_dir'}
    with pytest.raises(ValueError):
        module.validate_card_diff(old, dict(new, seed=42))
    with pytest.raises(ValueError):
        module.validate_card_diff(old, dict(new, rollout_group_size=8))


def test_existing_attempt_rejected(module, tmp_path):
    with pytest.raises(FileExistsError):
        module.reserve_root(tmp_path)

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/run_qwen8_teacher_pair.py'
    assert path.is_file(), 'Qwen8 fail-closed controller has not been implemented'
    spec = importlib.util.spec_from_file_location('qwen8_pair', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_declined_capability_cannot_train():
    m = module()
    for value in ({}, {'passed': False}, {'passed': 1}):
        with pytest.raises(ValueError):
            m.require_capability(value)
    m.require_capability({'passed': True})


def test_formal_does_not_warm_start_and_retains_four_milestones():
    m = module()
    changes = m.card_changes(Path('/new'), 'block3_mean')
    assert changes['total_training_steps'] == 200
    assert changes['diagnostic_save_steps'] == '50,100,150,200'
    assert changes['resume_mode'] == 'disable'
    assert changes['resume_from_path'] == ''
    assert changes['rollout_attempt_id'] == 'formal'
    assert changes['teacher_model'].endswith('/models/Qwen3-8B-Base')


def test_probe_resumes_only_its_own_initial_state():
    m = module()
    changes = m.card_changes(Path('/new/probes'), 'token_opd', 2)
    assert changes['total_training_steps'] == 2
    assert changes['resume_mode'] == 'resume_path'
    assert changes['resume_from_path'] == '/new/probes/token_opd/checkpoints/global_step_1'
    assert changes['diagnostic_save_steps'] == '1,2'


def test_only_approved_settings_change():
    m = module()
    original = {'learning_rate': 2e-6, 'seed': 21, 'max_response_length': 16384}
    root = Path('/new')
    card = {**original, **m.card_changes(root, 'token_opd')}
    m.validate_card(card, original, root, 'token_opd')
    with pytest.raises(ValueError, match='learning_rate'):
        m.validate_card({**card, 'learning_rate': 1e-5}, original, root, 'token_opd')


def test_pair_uses_same_teacher_and_only_method_differences():
    m = module()
    common = {'teacher_model': '8b', 'student_model': '4b', 'seed': 21}
    cards = {v: {**common, 'variant': v} for v in m.VARIANTS}
    m.validate_pair(cards)
    cards['block3_mean']['teacher_model'] = 'old-grpo'
    with pytest.raises(ValueError, match='teacher_model'):
        m.validate_pair(cards)


def test_both_real_ray_cache_paths_fit_unix_socket_budget():
    m = module()
    m.helpers()
    from recover_historical17_llama import check_socket_budget
    for variant in m.VARIANTS:
        check_socket_budget(m.CACHE / variant / 'train/tmp')

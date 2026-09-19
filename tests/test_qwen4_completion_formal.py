import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def controller():
    assert importlib.util.find_spec('run_qwen4_completion_block3') is not None
    return importlib.import_module('run_qwen4_completion_block3')


def test_formal_keeps_accepted_training_settings_and_four_checkpoints():
    m = controller()
    env = m.training_env()
    assert env['STUDENT_MODEL'] == str(m.gate.assets.STUDENT)
    assert env['SOURCE_COMMIT'] == m.TRAIN_COMMIT
    assert env['TOTAL_TRAINING_STEPS'] == '200'
    assert env['DIAGNOSTIC_SAVE_STEPS'] == '50,100,150,200'
    assert env['SAVE_FREQ'] == env['TEST_FREQ'] == '-1'
    assert env['STOP_AFTER_STEP'] == '-1'
    assert env['RESUME_MODE'] == 'disable' and env['RESUME_FROM_PATH'] == ''
    assert env['VARIANT'] == 'block3_mean'
    assert env['OPD_PROMPT_PROTOCOL'] == m.gate.PROTOCOL
    assert env['OPD_REQUEST_SEED_RULE'] == m.gate.SEED_RULE
    assert env['OPD_DIAG_INTERVAL'] == '1'
    assert env['ROLLOUT_ATTEMPT_ID'] == 'formal'
    assert env['REMOTE'] == 'ml2' and env['PREPARE_ONLY'] == 'true'
    probe = m.gate.training_env(m.RUNTIME, m.TRAIN_COMMIT, m.RUN_ROOT, 'block3_mean', 1)
    for key in ('ENV_SEED', 'TRAIN_BATCH_SIZE', 'ROLLOUT_GROUP_SIZE', 'MAX_RESPONSE_LENGTH',
                'LEARNING_RATE', 'ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU',
                'ROLLOUT_GPU_MEMORY_UTILIZATION', 'MATH_TEACHER'):
        assert env[key] == probe[key]


def test_card_validation_rejects_probe_resume_loss_changes_and_missing_milestone():
    m = controller()
    reference = {'learning_rate': 2e-6, 'opd_block_size': 3, 'seed': 21,
                 'total_training_steps': 2, 'diagnostic_save_steps': '1',
                 'stop_after_step': 1, 'resume_mode': 'disable', 'resume_from_path': ''}
    card = {**reference, **m.formal_card_overrides()}
    m.validate_card(card, reference)
    for key, wrong in [('learning_rate', 1e-6), ('opd_block_size', 1), ('seed', 7),
                       ('diagnostic_save_steps', '50,100,200'), ('resume_mode', 'resume_path'),
                       ('resume_from_path', '/probe/global_step_2')]:
        with pytest.raises(ValueError):
            m.validate_card({**card, key: wrong}, reference)


def test_post_training_audit_requires_all_four_weights_and_all_diagnostics():
    m = controller()
    command = list(map(str, m.checkpoint_audit_command()))
    assert command[command.index('--checkpoint-steps') + 1] == '50,100,150,200'
    assert command[command.index('--expected-diag-interval') + 1] == '1'
    assert command[command.index('--expected-diagnostic-steps') + 1] == ','.join(map(str, range(1, 201)))
    assert '--skip-eval' in command


def test_gate_precondition_requires_both_successful_resume_arms():
    m = controller()
    acceptance = {'passed': True, 'arms': {'block3_mean': [], 'token_opd': []}}
    m.validate_gate({'status': 'complete'}, acceptance, m.TRAIN_COMMIT)
    for state, report, commit in [({'status': 'running'}, acceptance, m.TRAIN_COMMIT),
                                  ({'status': 'complete'}, {'passed': False}, m.TRAIN_COMMIT),
                                  ({'status': 'complete'}, acceptance, 'other')]:
        with pytest.raises(ValueError):
            m.validate_gate(state, report, commit)


def test_retry_uses_executable_local_tmpfs_not_nfs():
    m = controller()
    assert m.CACHE == Path('/dev/shm/opd-q4-c3')
    assert '20260920v3_' in str(m.RUN_ROOT)
    assert m.training_env()['LOCAL_CACHE_ROOT'] == str(m.CACHE / 'train')
    m.validate_cache_mount('tmpfs', 'rw,relatime', 100_000_000_000)
    for fstype, options, free in [('nfs4', 'rw', 10**12),
                                  ('tmpfs', 'rw,noexec', 10**12), ('tmpfs', 'rw', 100)]:
        with pytest.raises(ValueError):
            m.validate_cache_mount(fstype, options, free)


def test_failure_cleanup_targets_only_recorded_training_group(monkeypatch):
    m = controller()
    calls = []
    monkeypatch.setattr(m.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: None)
    m.cleanup_failed_group(12345)
    assert calls == [(12345, m.signal.SIGTERM), (12345, m.signal.SIGKILL)]

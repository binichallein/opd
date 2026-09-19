import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'scripts/run_qwen4_pair.py'
    assert path.is_file(), 'Qwen4 block-first controller is missing'
    spec = importlib.util.spec_from_file_location('qwen4_queue_tests', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_user_order_includes_all_four_checkpoints_and_late_initial_eval():
    m = module()
    events = []
    def evaluate(name):
        events.append(('eval', name))
        return {'passed': True}
    results = m.execute_ordered(evaluate, lambda name: events.append(('train', name)))
    assert events == [('train', 'block3_mean')] + [
        ('eval', f'block3_mean_step{s}') for s in (200, 150, 100, 50)] + [
        ('eval', 'student_base'), ('train', 'token_opd')] + [
        ('eval', f'token_opd_step{s}') for s in (200, 150, 100, 50)]
    assert len(results) == 9


def test_failure_does_not_launch_initial_or_token():
    m = module()
    events = []
    with pytest.raises(ValueError, match='Incomplete'):
        m.execute_ordered(lambda name: {'passed': False}, lambda name: events.append(name))
    assert events == ['block3_mean']


def test_two_arms_start_from_identical_original_weights_and_save150():
    m = module()
    envs = [m.training_env(ROOT, 'sha', Path('/new'), name) for name in m.VARIANTS]
    allowed = {'VARIANT', 'EXP_NAME', 'OPD_DIAG_OUTPUT_DIR', 'LOSSLESS_ROLLOUT_DIR'}
    assert {k: v for k, v in envs[0].items() if k not in allowed} == {
        k: v for k, v in envs[1].items() if k not in allowed}
    for env in envs:
        assert env['STUDENT_MODEL'] == str(m.assets.STUDENT)
        assert env['RESUME_MODE'] == 'disable' and env['RESUME_FROM_PATH'] == ''
        assert env['DIAGNOSTIC_SAVE_STEPS'] == '50,100,150,200'
        assert env['TOTAL_TRAINING_STEPS'] == '200'
        assert env['OPD_PROMPT_PROTOCOL'] == 'qwen3_historical17_v1'
        assert env['TEACHER_MODEL_REVISION'] == ''


def test_probe_and_formal_audit_steps_are_separate():
    m = module()
    for variant in m.VARIANTS:
        formal = m.audit_command(ROOT, Path('/new') / variant, 'sha')
        assert formal[formal.index('--checkpoint-steps') + 1] == '50,100,150,200'
        for step in (1, 2):
            probe = m.audit_command(ROOT, Path('/probe') / variant, 'sha', step)
            assert probe[probe.index('--checkpoint-steps') + 1] == ('1' if step == 1 else '1,2')


def cards(m):
    common = {**m.base.paired.EXPECTED_TRAINING_VALUES, 'source_commit': 'sha',
              'diagnostic_save_steps': '50,100,150,200', 'student_model': str(m.assets.STUDENT),
              'student_model_revision': m.assets.REVISION, 'teacher_model': str(m.assets.TEACHER),
              'teacher_model_revision': '', 'opd_prompt_protocol': m.PROTOCOL,
              'opd_window_mode': 'fixed', 'ppo_epochs': 1, 'resume_mode': 'disable',
              'resume_from_path': '', 'rollout_attempt_id': 'formal',
              'expected_train_sha256': m.jobs.TRAIN_SHA}
    return {name: {**common, 'variant': name, 'opd_block_size': size,
                   'opd_block_advantage_mode': mode} for name, size, mode in
            [('token_opd', 1, 'sum'), ('block3_mean', 3, 'mean')]}


def test_pair_validation_rejects_silent_source_or_hyperparameter_changes():
    m = module()
    original = cards(m)
    m.validate_pair(original, 'sha')
    for key, value in [('diagnostic_save_steps', '50,100,200'), ('learning_rate', 1e-6),
                       ('student_model', '/block3/trained'), ('opd_block_size', 10)]:
        altered = copy.deepcopy(original)
        altered['token_opd'][key] = value
        with pytest.raises(ValueError):
            m.validate_pair(altered, 'sha')


def test_initial_model_resolution_cannot_return_trained_or_probe_model():
    m = module()
    assert m.model_paths(Path('/new'), 'student_base') == (None, m.assets.STUDENT)
    actor, model = m.model_paths(Path('/new'), 'block3_mean_step150')
    assert actor == Path('/new/block3_mean/checkpoints/global_step_150/actor')
    assert model == Path('/new/merged/block3_mean_step150')
    for invalid in ('token_opd_step125', 'sliding3_step150', 'student_base_step0'):
        with pytest.raises(ValueError):
            m.model_paths(Path('/new'), invalid)


def test_report_keeps_step150_and_each_benchmark_separate(tmp_path):
    m = module()
    results = {}
    names = ['student_base'] + [f'{v}_step{s}' for v in m.VARIANTS for s in m.STEPS]
    for name in names:
        results[name] = {'per_task': {task: {'avg_at_8': .1, 'pass_at_8': .2,
            'format_error_rate': .3, 'engine_truncation_ratio': .4} for task in m.shared.TASK_COUNTS}}
    m.write_comparison(tmp_path, results)
    result = json.loads((tmp_path / 'paired_comparison.json').read_text())
    assert set(result['per_benchmark']) == set(m.shared.TASK_COUNTS)
    for values in result['per_benchmark'].values():
        assert 'delta_step150_pp' in values
    assert 'macro' not in result

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location('llama_queue', Path(__file__).parents[1] / 'scripts/run_llama32_pair.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_waits_for_all_predecessor_steps_and_rejects_failure(tmp_path):
    m = module()
    state = tmp_path / 'queue_state.json'
    state.write_text(json.dumps({'status': 'running'}))
    assert not m.predecessor_ready(tmp_path)
    state.write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(ValueError):
        m.predecessor_ready(tmp_path)
    state.write_text(json.dumps({'status': 'complete', 'block3_eval_steps': [50, 100, 200],
                                 'protected_inputs_verified': True}))
    with pytest.raises((ValueError, FileNotFoundError)):
        m.predecessor_ready(tmp_path)


def test_matched_training_and_native_protocol(tmp_path):
    m = module()
    left = m.training_env(tmp_path, 'a' * 40, tmp_path / 'run', 'token_opd')
    right = m.training_env(tmp_path, 'a' * 40, tmp_path / 'run', 'block3_mean')
    allowed = {'VARIANT', 'EXP_NAME', 'OPD_DIAG_OUTPUT_DIR', 'LOSSLESS_ROLLOUT_DIR'}
    assert {k for k in left if left[k] != right[k]} == allowed
    assert left['STUDENT_MODEL'].endswith('Llama-3.2-1B-Instruct')
    assert left['MATH_TEACHER'].endswith('Llama-3.2-3B-Instruct')
    assert left['OPD_PROMPT_PROTOCOL'] == 'llama32_nonthinking_v1'
    assert left['RESUME_MODE'] == right['RESUME_MODE'] == 'disable'
    assert left['TOTAL_TRAINING_STEPS'] == '200'
    assert left['DIAGNOSTIC_SAVE_STEPS'] == '50,100,200'
    assert left['ENV_SEED'] == '21'
    second = m.training_env(tmp_path, 'a' * 40, tmp_path / 'probes', 'block3_mean', 2)
    assert second['RESUME_MODE'] == 'resume_path'
    assert '/probes/' in second['RESUME_FROM_PATH']


def test_execution_order_and_failed_eval_blocks_next_training():
    m = module()
    calls = []
    def evaluate(name):
        calls.append(name)
        return {'passed': True}
    m.execute_ordered(evaluate, lambda v: calls.append('train_' + v))
    assert calls == ['student_base', 'train_token_opd', 'token_opd_step200', 'token_opd_step100',
                     'token_opd_step50', 'train_block3_mean', 'block3_mean_step200',
                     'block3_mean_step100', 'block3_mean_step50']
    calls.clear()
    with pytest.raises(ValueError):
        m.execute_ordered(lambda _: {'passed': False}, lambda v: calls.append(v))
    assert not calls


def test_audit_targets_both_llama_revisions(tmp_path):
    m = module()
    cmd = list(map(str, m.audit_command(tmp_path, tmp_path / 'token_opd', 'a' * 40)))
    assert cmd[cmd.index('--expected-student-model-suffix') + 1] == m.assets.STUDENT.name
    assert cmd[cmd.index('--expected-teacher-model-revision') + 1] == m.assets.SPECS['teacher']['revision']


def test_pair_blocks_warm_start_and_changed_hyperparameters():
    m = module()
    left = {**m.base.paired.EXPECTED_TRAINING_VALUES,
        'source_commit': 'abc', 'student_model': str(m.assets.STUDENT), 'teacher_model': str(m.assets.TEACHER),
        'student_model_revision': m.assets.SPECS['student']['revision'],
        'teacher_model_revision': m.assets.SPECS['teacher']['revision'], 'opd_prompt_protocol': m.LLAMA_PROTOCOL,
        'opd_window_mode': 'fixed', 'ppo_epochs': 1, 'resume_mode': 'disable', 'resume_from_path': '',
        'rollout_attempt_id': 'formal', 'expected_train_sha256': m.jobs.TRAIN_SHA,
        'variant': 'token_opd', 'opd_block_size': 1, 'opd_block_advantage_mode': 'sum'}
    right = {**left, 'variant': 'block3_mean', 'opd_block_size': 3, 'opd_block_advantage_mode': 'mean'}
    m.validate_pair({'token_opd': left, 'block3_mean': right}, 'abc')
    for key, value in [('resume_from_path', '/token/checkpoints'), ('learning_rate', 1e-6),
                       ('seed', 22), ('opd_prompt_protocol', 'legacy')]:
        with pytest.raises(ValueError):
            m.validate_pair({'token_opd': left, 'block3_mean': {**right, key: value}}, 'abc')


def test_predecessor_requires_each_full_accepted_evaluation(tmp_path):
    m = module()
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    put(tmp_path / 'queue_state.json', {'status': 'complete', 'block3_eval_steps': [50,100,200],
                                       'protected_inputs_verified': True})
    result = {'passed': True, 'grader_sha256': m.grading.HISTORICAL_GRADER_SHA256,
              'per_task': {task: {'num_examples': count, 'num_rollouts': count*8, 'avg_at_8': 0., 'pass_at_8': 0.}
                           for task, count in m.shared.TASK_COUNTS.items()}}
    put(tmp_path / 'block3_eval_acceptance.json', {'passed': True, 'models': {str(s): result for s in m.STEPS}})
    for step in m.STEPS:
        folder = tmp_path / 'evaluations' / f'block3_step{step}'
        put(folder / 'acceptance.json', result)
        (folder / 'exit_code.txt').write_text('0\n')
    assert m.predecessor_ready(tmp_path)
    (tmp_path / 'evaluations/block3_step50/exit_code.txt').write_text('1\n')
    with pytest.raises(ValueError):
        m.predecessor_ready(tmp_path)

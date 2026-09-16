import importlib.util
import json
from pathlib import Path

import pytest


def load():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_nonthinking_eval_block3.py'
    assert path.exists(), 'The ordered evaluation/Block3 controller is missing'
    spec = importlib.util.spec_from_file_location('eval_block3', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_block_uses_original_runtime_and_matched_nonthinking_controls(tmp_path):
    m = load()
    env = m.block_env(tmp_path)
    assert env['SOURCE_COMMIT'] == m.TRAIN_COMMIT
    assert env['REMOTE_ROOT'] == str(m.TRAIN_RUNTIME)
    assert env['VARIANT'] == 'block3_mean'
    assert env['STUDENT_MODEL'] == str(m.base.assets.STUDENT)
    assert env['RESUME_MODE'] == 'disable'
    assert env['TOTAL_TRAINING_STEPS'] == '200'
    assert env['ENV_SEED'] == '21'
    assert env['OPD_PROMPT_PROTOCOL'] == 'math_eval_nonthinking_v1'
    assert env['ROLLOUT_ATTEMPT_ID'] == 'formal'
    assert env['LOSSLESS_ROLLOUT_DIR'] == str(tmp_path / 'block3_mean/rollouts')
    assert env['OPD_DIAG_INTERVAL'] == '5'
    assert env['DIAGNOSTIC_SAVE_STEPS'] == '50,100,200'
    assert env['ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU'] == '1'
    assert env['ROLLOUT_GPU_MEMORY_UTILIZATION'] == '0.6'
    probe = m.block_env(tmp_path / 'probes', 2)
    assert probe['RESUME_MODE'] == 'resume_path'
    assert probe['ROLLOUT_ATTEMPT_ID'] == 'probe2'
    assert probe['RESUME_FROM_PATH'].endswith('block3_mean/checkpoints/global_step_1')


def test_pair_accepts_only_method_and_output_identity_changes():
    m = load()
    left = {**m.base.paired.EXPECTED_TRAINING_VALUES,
            'student_model': str(m.base.assets.STUDENT),
            'student_model_revision': m.base.assets.REVISION,
            'teacher_model': str(m.base.assets.TEACHER), 'source_commit': m.TRAIN_COMMIT,
            'variant': 'token_opd', 'opd_block_size': 1, 'opd_block_advantage_mode': 'sum',
            'opd_window_mode': 'fixed', 'ppo_epochs': 1,
            'opd_prompt_protocol': m.PROTOCOL, 'lossless_rollout_dir': '/token/rollouts'}
    right = {**left, 'variant': 'block3_mean', 'opd_block_size': 3,
             'opd_block_advantage_mode': 'mean', 'lossless_rollout_dir': '/block/rollouts'}
    m.validate_pair(left, right)
    for key, value in [('seed', 22), ('source_commit', 'a' * 40), ('opd_window_mode', 'sliding'),
                       ('learning_rate', 1e-6), ('opd_prompt_protocol', 'legacy'),
                       ('unlisted_key', 'unexpected')]:
        with pytest.raises(ValueError):
            m.validate_pair(left, {**right, key: value})


def fixture(m, root, monkeypatch):
    counts = {'math500': 2, 'aime24': 1, 'aime25': 1, 'amc23': 1}
    monkeypatch.setattr(m, 'TASK_COUNTS', counts)
    out, data, model = root / 'outputs', root / 'data', root / 'model'
    out.mkdir(); data.mkdir()
    config = {**m.EVAL_VALUES, 'model_path': str(model), 'eval_jsonl_dir': str(data),
              'tasks': list(counts), 'gpus': ['0', '1', '2', '3']}
    (out / 'eval_config.json').write_text(json.dumps(config))
    summary = {'tasks': {}}
    for task, count in counts.items():
        problems = [{'id': str(i), 'problem': f'Question {i}', 'prompt': f'Question {i}',
                     'answer': '1', 'source': task} for i in range(count)]
        (data / f'{task}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in problems))
        rows = [{'task': task, 'example_id': x['id'], 'source': x['source'],
                 'problem': x['problem'], 'prompt': x['prompt'], 'answer': x['answer'],
                 'rollout_id': j, 'seed': 21+j, 'response': 'one\u2028two'}
                for x in problems for j in range(8)]
        (out / f'{task}_t1.0_p0.9_n8-MNT16384.jsonl').write_text(
            ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows))
        (out / f'{task}_graded.jsonl').write_text(
            ''.join(json.dumps({**x, 'correct': x['rollout_id'] == 0}, ensure_ascii=False)+'\n' for x in rows))
        summary['tasks'][task] = {'num_examples': count, 'total_rollouts': count*8,
                                 'avg_at_n': .125, 'pass_at_n': 1.0}
    (out / 'summary.json').write_text(json.dumps(summary))
    return out, data, model


def test_full_eval_is_audited_and_reported_per_benchmark(tmp_path, monkeypatch):
    m = load()
    out, data, model = fixture(m, tmp_path, monkeypatch)
    result = m.audit_evaluation(out, data, model)
    assert result['passed'] is True
    assert result['per_task']['math500']['avg_at_8'] == .125
    assert result['per_task']['math500']['pass_at_8'] == 1.0
    assert 'macro' not in str(result)


@pytest.mark.parametrize('fault', ['missing', 'wrong_problem', 'wrong_score', 'thinking', 'grader'])
def test_eval_gate_rejects_incomplete_or_mismatched_evidence(tmp_path, monkeypatch, fault):
    m = load()
    out, data, model = fixture(m, tmp_path, monkeypatch)
    if fault == 'missing':
        p = out / 'math500_graded.jsonl'
        p.write_text(''.join(p.open().readlines()[:-1]))
    elif fault == 'wrong_problem':
        p = data / 'math500.jsonl'
        rows = [json.loads(x) for x in p.open()]
        rows[0]['problem'] = 'different question'
        p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    elif fault == 'wrong_score':
        p = out / 'summary.json'
        v = json.loads(p.read_text()); v['tasks']['math500']['avg_at_n'] = .9
        p.write_text(json.dumps(v))
    else:
        p = out / 'eval_config.json'
        v = json.loads(p.read_text()); v['enable_thinking' if fault == 'thinking' else 'grader'] = True if fault == 'thinking' else 'verl'
        p.write_text(json.dumps(v))
    with pytest.raises(ValueError):
        m.audit_evaluation(out, data, model)


def test_block_cannot_run_before_all_evals_pass(tmp_path):
    m = load()
    calls = []
    def evaluate(name):
        calls.append(name)
        if name == 'token_step50':
            raise ValueError('incomplete eval')
        return {'passed': True}
    with pytest.raises(ValueError):
        m.execute_ordered(tmp_path, evaluate, lambda: calls.append('train'))
    assert calls == ['token_step200', 'token_step100', 'token_step50']
    assert not (tmp_path / 'token_effect.json').exists()


def test_complete_eval_produces_separate_scores_before_training(tmp_path):
    m = load()
    calls = []
    def evaluate(name):
        calls.append(name)
        return {'passed': True, 'per_task': {t: {'num_examples': n, 'avg_at_8': .2, 'pass_at_8': .4}
                                           for t, n in m.TASK_COUNTS.items()}}
    def train():
        assert (tmp_path / 'token_effect.json').exists()
        calls.append('train')
    m.execute_ordered(tmp_path, evaluate, train)
    assert calls == list(m.EVAL_ORDER) + ['train']
    report = json.loads((tmp_path / 'token_effect.json').read_text())
    assert set(report['per_benchmark']) == set(m.TASK_COUNTS)
    assert 'macro' not in str(report)
    assert report['per_benchmark']['math500']['token_step200']['delta_base_avg_pp'] == 0

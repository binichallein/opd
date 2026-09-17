import importlib.util
import json
from pathlib import Path

import pytest


def load(name='run_nonthinking_block3_eval'):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    assert path.exists(), 'Missing Block3 evaluation-only controller'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checkpoint_evaluation_uses_explicit_block_source(tmp_path):
    m = load('run_nonthinking_eval_block3')
    source = tmp_path / 'source_block'
    calls = []
    def runner(command, job, **kwargs):
        calls.append(command)
        raise RuntimeError('stop before model execution')
    with pytest.raises(RuntimeError, match='stop before model execution'):
        m.evaluate_model('block3_step200', tmp_path, runner,
                         checkpoint_run=source, checkpoint_prefix='block3_step')
    command = calls[0]
    assert command[command.index('--local_dir') + 1] == source / 'checkpoints/global_step_200/actor'
    assert command[command.index('--target_dir') + 1] == tmp_path / 'merged/block3_step200'
    assert command[1] == m.TRAIN_RUNTIME / 'external/revisiting_opd/scripts/model_merger.py'


def result(m, avg=.2):
    return {'passed': True, 'grader_sha256': m.shared.grading.HISTORICAL_GRADER_SHA256,
            'per_task': {t: {'num_examples': n, 'num_rollouts': n * 8,
                             'avg_at_8': avg, 'pass_at_8': .5}
                         for t, n in m.shared.TASK_COUNTS.items()}}


def test_eval_only_order_and_separate_matched_scores(tmp_path):
    m = load()
    calls = []
    token = {step: result(m) for step in m.STEPS}
    def evaluate(step):
        calls.append(step)
        return result(m, .3)
    m.execute_ordered(tmp_path, token, evaluate)
    assert calls == [200, 100, 50]
    report = json.loads((tmp_path / 'block3_comparison.json').read_text())
    assert set(report['per_benchmark']) == set(m.shared.TASK_COUNTS)
    assert 'macro' not in str(report)
    assert report['per_benchmark']['math500']['200']['delta_avg_pp'] == pytest.approx(10)
    assert (tmp_path / 'block3_eval_acceptance.json').exists()


@pytest.mark.parametrize('fault', ['failed', 'missing_task', 'wrong_count', 'wrong_grader'])
def test_bad_eval_stops_before_next_checkpoint_or_report(tmp_path, fault):
    m = load()
    calls = []
    def evaluate(step):
        calls.append(step)
        r = result(m)
        if fault == 'failed': r['passed'] = False
        if fault == 'missing_task': del r['per_task']['math500']
        if fault == 'wrong_count': r['per_task']['math500']['num_rollouts'] = 8
        if fault == 'wrong_grader': r['grader_sha256'] = 'incorrect'
        return r
    with pytest.raises(ValueError):
        m.execute_ordered(tmp_path, {s: result(m) for s in m.STEPS}, evaluate)
    assert calls == [200]
    assert not (tmp_path / 'block3_eval_acceptance.json').exists()
    assert not (tmp_path / 'block3_comparison.json').exists()


def test_unaccepted_token_reference_prevents_any_eval(tmp_path):
    m = load()
    token = {s: result(m) for s in m.STEPS}
    token[50]['passed'] = False
    with pytest.raises(ValueError):
        m.execute_ordered(tmp_path, token, lambda s: pytest.fail('must not launch'))

import importlib.util
from pathlib import Path

import pytest


def test_probe_counts_keep_missing_format_unobserved():
    rows = module().probe_rows([{'id': 'qwen06_qwen17_truncation_192', 'data': {
        'primary32': {'q06_base': {'n': 32, 'length_stop_count': 12,
                                  'periodic_tail_count': 10, 'correct_count': 0}},
        'prompt_ablation_first16': {}}}])
    assert rows[0]['n'] == 32 and rows[0]['length_stops'] == 12
    assert rows[0]['missing_box'] is None and rows[0]['correct'] == 0


def test_probe_rejects_impossible_count():
    with pytest.raises(ValueError):
        module().probe_rows([{'id': 'llama_prompt_16', 'models': {'student': {'cells': {
            'historical': {'responses': 32, 'length_stops': 33}}}}}])


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/build_history_paper_tables.py'
    spec = importlib.util.spec_from_file_location('history_tables', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def entry(task='math500', value=.2, view='historical_external'):
    return {'id': 'E1', 'series': 'qwen17_ml2', 'view': view, 'arm': 'token_opd',
            'checkpoint_step': 200, 'training_seed': 21, 'n': 8, 'source': 'private',
            'model_path': '/home/private/model', 'status': 'completed',
            'per_task': {task: {'metrics': {'avg_at_8': value, 'pass_at_8': .4}}}}


def test_task_columns_are_not_averaged():
    m = module()
    rows = m.group_evaluations([entry(), entry('aime24', .1)])
    assert len(rows) == 1
    assert rows[0]['per_task']['math500']['avg_at_8'] == .2
    assert rows[0]['per_task']['aime24']['avg_at_8'] == .1
    assert 'model_path' not in rows[0] and 'source' not in rows[0]


def test_grader_views_are_separate_and_duplicates_rejected():
    m = module()
    assert len(m.group_evaluations([entry(), entry(view='builtin_verl')])) == 2
    with pytest.raises(ValueError):
        m.group_evaluations([entry(), entry()])


def test_missing_is_not_zero_and_metrics_not_renamed():
    m = module()
    assert m.metric_cell({'avg_at_8': None, 'pass_at_8': None}) == '--'
    assert m.metric_cell({'avg_at_8': 0., 'pass_at_8': 0.}) == '0.00 / 0.00'
    assert m.metric_cell({'accuracy': .1234}) == '12.34'
    assert m.metric_cell({'mean_score': .12, 'pass_at_k': .2}) == '12.00 / 20.00'

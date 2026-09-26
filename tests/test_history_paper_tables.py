import importlib.util
import json
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


PAPER = Path(__file__).resolve().parents[1] / 'paper/iclr2027'


def rows():
    return json.loads((PAPER / 'data/history_public.json').read_text())['rows']


def test_complete_endpoint_table_keeps_negative_results_unchanged():
    text = module().endpoint_table(rows())
    assert text == (PAPER / 'generated/history_endpoints.tex').read_text()
    assert 'Qwen 0.6B & Block3 & 0.57' in text
    assert 'Llama 1B & Block3 & 6.30' in text


def test_explicit_highlights_are_a_subset_not_a_replacement():
    text = module().endpoint_table(rows(), series_ids=('qwen17_ml2', 'deepseek_justrl'))
    assert 'Qwen 1.7B & Block3 & 54.75' in text
    assert 'DeepSeek 1.5B & Block3 & 84.75' in text
    assert 'Qwen 0.6B' not in text and 'Llama 1B' not in text
    full = module().endpoint_table(rows())
    for line in text.splitlines():
        if ' & ' in line:
            assert line in full


def test_endpoint_selection_rejects_unknown_series():
    with pytest.raises(ValueError, match='Unknown endpoint series'):
        module().endpoint_table(rows(), series_ids=('typo',))


def test_manuscript_keeps_complete_table_and_labels_highlights():
    for language in ('en', 'zh'):
        results = (PAPER / f'results_{language}.tex').read_text()
        appendix = (PAPER / f'appendix_{language}.tex').read_text()
        protocol = (PAPER / f'protocol_{language}.tex').read_text()
        assert r'\input{generated/history_highlights}' in results
        assert r'\ref{tab:historyendpointsfull}' in results
        assert r'\input{generated/history_endpoints}' in appendix
        assert r'\label{tab:historyendpointsfull}' in appendix
        assert 'Rethinking' in protocol and r'\citep{rethinking}' in protocol

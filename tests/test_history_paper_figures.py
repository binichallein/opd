import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/build_history_paper_figures.py'
    spec = importlib.util.spec_from_file_location('history_figures', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def row(step, score, view='historical_external_audited'):
    return {'series': 'qwen17_ml2', 'view': view, 'arm': 'token_opd',
            'checkpoint_step': step, 'training_seed': 21, 'n': 8,
            'per_task': {'math500': {'avg_at_8': score, 'pass_at_8': .6}},
            'evidence_ids': [f'E{step}']}


def test_selects_exact_view_without_mixing_old_grader():
    m = module()
    points = m.select_points([row(200, .5), row(50, .4), row(100, .9, 'builtin_verl')],
                             'qwen17_ml2', 'historical_external_audited',
                             'token_opd', 'math500', 'avg_at_8')
    assert [(p['step'], p['value']) for p in points] == [(50, .4), (200, .5)]


def test_missing_point_is_not_zero():
    m = module()
    assert m.select_points([row(50, None)], 'qwen17_ml2', 'historical_external_audited',
                           'token_opd', 'math500', 'avg_at_8') == []


def test_rejects_duplicate_step_or_invalid_metric():
    m = module()
    args = ('qwen17_ml2', 'historical_external_audited', 'token_opd', 'math500', 'avg_at_8')
    with pytest.raises(ValueError):
        m.select_points([row(50, .5), row(50, .4)], *args)
    with pytest.raises(ValueError):
        m.select_points([row(50, 1.5)], *args)
    with pytest.raises(ValueError):
        m.select_points([row(50, float('nan'))], *args)

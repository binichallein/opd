import importlib.util
import copy
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    'paper_check', Path(__file__).resolve().parents[1] / 'scripts/check_iclr2027_paper.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def fixture_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK.subprocess, 'check_output', lambda *a, **kw: 'Author:          \nPages: 20\n')
    for lang in ('en', 'zh'):
        path = tmp_path / 'build' / lang
        path.mkdir(parents=True)
        (path / f'main_{lang}.pdf').write_bytes(b'%PDF-1.5')
        (path / f'main_{lang}.log').write_text('')
        (path / f'main_{lang}.aux').write_text(r'\newlabel{mainend}{{7}{9}{Conclusion}{section.7}{}}')
        (path / f'main_{lang}.txt').write_text('AI Use Statement' if lang == 'en' else 'AI 使用声明')
    return tmp_path


def test_valid_paper_with_chinese_extracted_spacing(tmp_path, monkeypatch):
    result = CHECK.inspect_paper(fixture_paper(tmp_path, monkeypatch), require_evaluation=False)
    assert result['passed'] and result['papers']['en']['main_text_pages'] == 9


@pytest.mark.parametrize('name,value', [
    ('aux', r'\newlabel{mainend}{{7}{10}{Conclusion}{section.7}{}}'),
    ('log', "LaTeX Warning: Citation `missing' on page 3 undefined"),
    ('txt', 'AI Use Statement /home/tyf/paper'),
    ('txt', 'AI Use Statement TBD'),
])
def test_unready_paper_rejected(tmp_path, monkeypatch, name, value):
    paper = fixture_paper(tmp_path, monkeypatch)
    (paper / 'build/en' / f'main_en.{name}').write_text(value)
    with pytest.raises(ValueError):
        CHECK.inspect_paper(paper, require_evaluation=False)


def complete_report():
    return {'complete': True, 'missing_checkpoints': [], 'rows': [
        {'variant': arm, 'step': step, 'task': task, 'status': 'accepted',
         'avg_at_8': .5, 'pass_at_8': .7, 'format_error_rate': .1, 'engine_truncation_ratio': .1}
        for arm in ('token_opd', 'block3_mean') for step in (50, 100, 150, 200)
        for task in ('math500', 'aime24', 'aime25', 'amc23')],
        'bootstrap': [{'step': 200, 'task': task} for task in ('math500', 'aime24', 'aime25', 'amc23')]}


def test_finalization_requires_all_checkpoints():
    report = complete_report()
    CHECK.validate_evaluation(report)
    report['complete'] = False
    with pytest.raises(ValueError):
        CHECK.validate_evaluation(report)


def test_finalization_rejects_stale_partial_figure_label(tmp_path, monkeypatch):
    paper = fixture_paper(tmp_path, monkeypatch)
    directory = paper / 'generated/final'
    directory.mkdir(parents=True)
    (directory / 'qwen4_summary.json').write_text(json.dumps(complete_report()))
    (paper / 'asset_paths.tex').write_text(r'\newcommand{\qwenresults}{generated/final}')
    (paper / 'build/en/main_en.txt').write_text('AI Use Statement Qwen4 Avg@8 (PARTIAL)')
    CHECK.inspect_paper(paper, require_evaluation=False)
    with pytest.raises(ValueError, match='partial'):
        CHECK.inspect_paper(paper, require_evaluation=True)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'nonfinite', 'no_ci'])
def test_incomplete_evidence_cannot_be_finalized(change):
    report = copy.deepcopy(complete_report())
    if change == 'missing':
        report['rows'].pop()
    elif change == 'duplicate':
        report['rows'][-1] = report['rows'][0]
    elif change == 'nonfinite':
        report['rows'][0]['avg_at_8'] = float('nan')
    else:
        report['bootstrap'] = []
    with pytest.raises(ValueError):
        CHECK.validate_evaluation(report)

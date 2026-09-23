import copy
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def module():
    path = SCRIPTS / 'build_qwen17_paper_assets.py'
    assert path.exists(), 'Missing offline Qwen17 manuscript asset builder'
    spec = importlib.util.spec_from_file_location('qwen17_paper', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def report():
    root = SCRIPTS.parent / 'results/qwen17_instruct_20260922/final_20260922_2316'
    import json
    return json.loads((root / 'results.json').read_text())


def test_complete_nine_view_instruct_report_is_required():
    m = module()
    m.validate_report(report())


@pytest.mark.parametrize('change', ['partial', 'missing_view', 'base_student', 'wrong_teacher',
                                   'thinking', 'wrong_seed', 'missing_task', 'nonfinite'])
def test_reject_wrong_pair_or_incomplete_evidence(change):
    m = module()
    j = copy.deepcopy(report())
    if change == 'partial':
        j['complete'] = False
    elif change == 'missing_view':
        del j['per_model']['student_base']
    elif change == 'base_student':
        j['training']['block3_mean']['run_card']['student_model'] += '-Base'
    elif change == 'wrong_teacher':
        j['training']['token_opd']['run_card']['teacher_model'] = 'Qwen3-4B'
    elif change == 'thinking':
        j['training']['token_opd']['run_card']['opd_prompt_protocol'] = 'chat_thinking'
    elif change == 'wrong_seed':
        j['training']['token_opd']['run_card']['seed'] = 7
    elif change == 'missing_task':
        del j['per_model']['block3_mean_step200']['amc23']
    else:
        j['per_model']['block3_mean_step200']['math500']['avg_at_8'] = float('nan')
    with pytest.raises(ValueError):
        m.validate_report(j)


def test_question_aggregation_rejects_summary_disagreement():
    m = module()
    rows = [{'id': str(i), 'correct_count': 4, 'missing_box_count': 0,
             'length_stop_count': 0} for i in range(30)]
    expected = {'avg_at_8': .5, 'pass_at_8': 1., 'format_error_rate': 0.,
                'engine_truncation_ratio': 0.}
    m.verify_question_scores(rows, expected, 'aime24')
    expected['avg_at_8'] = .6
    with pytest.raises(ValueError):
        m.verify_question_scores(rows, expected, 'aime24')


def test_generated_tables_preserve_original_student_and_all_checkpoints():
    m = module()
    text = m.all_checkpoint_table(report(), 'en', 'avg_at_8')
    assert 'Original student' in text and '71.45' in text
    for step in (50, 100, 150, 200):
        assert f'Token {step}' in text and f'Block3 {step}' in text
    assert 'macro' not in text.lower()

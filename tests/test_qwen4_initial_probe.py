import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / 'scripts/diagnose_qwen4_initial_rollouts.py'


def module():
    assert PATH.exists(), 'Missing inference-only diagnostic'
    spec = importlib.util.spec_from_file_location('initial_probe', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_fixed_and_independent_seeds_do_not_change_prompt_ids():
    probe = module()
    rows = [{'prompt_token_ids': [3, 4], 'index': 10}]
    fixed = probe.requests(rows, seed_mode='fixed', n=8)
    independent = probe.requests(rows, seed_mode='independent', n=8)
    assert [r['seed'] for r in fixed] == [21] * 8
    assert [r['seed'] for r in independent] == list(range(21, 29))
    assert [r['prompt_token_ids'] for r in fixed] == [[3, 4]] * 8
    assert [r['prompt_token_ids'] for r in independent] == [[3, 4]] * 8
    assert len({r['request_id'] for r in independent}) == 8
    assert 'seed' not in rows[0]


def test_select_sources_retains_actual_original_inputs():
    probe = module()
    original = {'source_extra_info': {'index': 5, 'question': 'q', 'answer': 'a'},
                'prompt_token_ids': [1, 2], 'prompt_text': 'p',
                'response_token_ids': [3, 4]}
    rows = probe.select_sources([original, original])
    assert len(rows) == 1
    assert rows[0]['prompt_token_ids'] == [1, 2]
    assert rows[0]['archived_response_token_ids'] == [3, 4]


def test_summary_separates_duplicates_loops_and_cap():
    probe = module()
    rows = [{'index': 1, 'response_token_ids': [1, 2], 'response_length': 2,
             'finish_reason': 'stop', 'tail_period': None}] * 8
    rows += [{'index': 2, 'response_token_ids': [3] * 300, 'response_length': 300,
              'finish_reason': 'length', 'tail_period': 1}] * 8
    stats = probe.summarize(rows)
    assert stats['responses'] == 16
    assert stats['unique_prompt_response_pairs'] == 2
    assert stats['length_stops'] == 8
    assert stats['periodic_tails'] == 8


def test_delete_chat_markers_preserves_all_other_token_ids():
    probe = module()
    assert hasattr(probe, 'delete_chat_markers'), 'Missing isolated marker intervention'
    ids = [151644, 872, 198, 91, 151667, 92, 151668, 151645, 198, 151644, 77091, 198]
    assert probe.delete_chat_markers(ids) == [872, 198, 91, 151667, 92, 151668, 198, 77091, 198]
    assert ids[0] == 151644

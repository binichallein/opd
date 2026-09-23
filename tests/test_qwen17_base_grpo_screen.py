import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def subject():
    return importlib.import_module('qualify_qwen17_base_grpo').configured_qualifier()


def test_only_the_requested_pair_without_mutating_historical_screen():
    historical = importlib.import_module('qualify_qwen06_teachers')
    before = historical.main_views()
    s = subject()
    assert s.PAIRS == {'g4_b17': ('g4', 'b17')}
    assert s.main_views() == {'b17_b17': {'model': 'b17', 'input': 'b17'},
                             'g4_b17': {'model': 'g4', 'input': 'b17'}}
    assert s.model_label('b17') == 'student'
    assert s.model_label('g4') == 'teacher'
    assert historical.main_views() == before and len(before) == 10


def test_readonly_exact_model_identities():
    s = subject()
    specs = s.assets.specifications()
    assert set(specs) == {'b17', 'g4'}
    assert specs['b17']['repo'] == 'Qwen/Qwen3-1.7B-Base'
    assert specs['b17']['revision'] == 'b0786a09cd6ee101cd8c90e30a5727beb8230544'
    assert specs['g4']['repo'] == 'lllyx/Qwen3-4B-Base-GRPO'
    assert all(spec['download_allowed'] is False for spec in specs.values())


def test_base_requests_match_training_and_preserve_causal_prefix():
    from test_qwen17_pairs import Tokenizer, selection, record
    s = subject()
    tok = Tokenizer()
    tok.eos_token_id = 151643
    chosen = selection(s)
    for row in chosen:
        row['answer'] = 'SECRET_GOLD_ANSWER_NOT_FOR_INPUT'
    direct = s.build_requests(chosen, 'b17', 'direct', tok)
    assert len(direct) == 128
    assert all(r['completion_prompt_text'].endswith('\n\nSolution:\n') for r in direct)
    assert all('SECRET_GOLD' not in r['prompt_text'] and '<think>' not in r['prompt_text'] for r in direct)
    assert all(r['sampling']['stop_token_ids'] == [] for r in direct)
    raw = [record(r, 'student', 'abcdefghij' if r['sample_index'] == 0 else 'OTHER') for r in direct]
    continuation = s.build_requests(chosen, 'b17', 'continuation', tok, raw)
    assert len(continuation) == 64
    transferred = s.for_target(continuation, tok, 200000)
    for row, teacher in zip(continuation, transferred):
        assert row['prefix_text'] == 'abcde'
        assert row['sampling']['max_tokens'] == 16379
        assert row['prompt_token_ids'] == row['base_prompt_token_ids'] + row['prefix_token_ids']
        assert teacher['prompt_token_ids'] == row['prompt_token_ids']
        assert teacher['sampling'] == row['sampling']
        assert row['seed'] == s.old.request_seed(row['question_id'], row['sample_index'], 'continuation')
    raw[0]['input_source'] = 'wrong_student'
    with pytest.raises(ValueError):
        s.build_requests(chosen, 'b17', 'continuation', tok, raw)


def test_native_teacher_control_keeps_its_own_stops_not_base_main_stops():
    from test_qwen17_pairs import Tokenizer, selection
    s = subject()
    tok = Tokenizer()
    tok.eos_token_id = 151643
    native = s.for_target(s.build_requests(selection(s), 'g4', 'direct', tok), tok, 200000)
    assert len(native) == 128
    assert all(r['sampling']['stop_token_ids'] == [151645, 151643] for r in native)
    assert all(r['completion_prompt_text'].endswith('<think>\n\n</think>\n\n') for r in native)
    assert all(r['enable_thinking'] is False for r in native)


def test_protocol_and_invocation_explicitly_exclude_training():
    s = subject()
    p = s.protocol()
    assert p['training_authorized'] is False and p['full_benchmark_eval'] is False
    assert p['stop_token_ids'] == []
    assert p['native_control_stop_token_ids'] == [151645, 151643]
    assert p['main_prompt_protocol'] == 'qwen3_completion_boxed_v1'
    assert p['response_budget'] == 16384
    assert p['bootstrap']['unit'] == 'paired_question'
    hashes = s.invocation()['runtime_hashes']
    assert any(p.endswith('/qualify_qwen17_base_grpo.py') for p in hashes)
    assert any(p.endswith('/run_qwen17_base_grpo_screen.py') for p in hashes)


def test_queue_runs_only_this_qualification_and_no_downloads():
    queue = importlib.import_module('run_qwen17_base_grpo_screen').configured_queue()
    assert queue.RUN.name == '20260923v3_qwen17_base_grpo_teacher_screen_ml2'
    assert str(queue.CACHE) == '/dev/shm/q17gs0923'
    assert queue.QUALIFIER_SCRIPT == 'qualify_qwen17_base_grpo.py'
    assert queue.ASSET_DOWNLOADS == ()
    commands = queue.preparation_commands(Path('/runtime'))
    assert len(commands) == 1
    assert commands[0][0] == 'prepare'
    assert str(commands[0][1][1]) == '/runtime/scripts/qualify_qwen17_base_grpo.py'
    assert queue.q.PAIRS == {'g4_b17': ('g4', 'b17')}


def test_existing_queue_defaults_unchanged():
    q = importlib.import_module('run_qwen06_teacher_screen')
    assert q.QUALIFIER_SCRIPT == 'qualify_qwen06_teachers.py'
    assert q.ASSET_DOWNLOADS == ('i06', 'i4')
    assert [name for name, _ in q.preparation_commands(Path('/runtime'))] == ['asset_i06', 'asset_i4', 'prepare']


def test_preexisting_scientific_thresholds_stay_in_force():
    from test_qwen17_pairs import cells
    s = subject()
    historical = importlib.import_module('qualify_qwen06_teachers')
    data = cells()
    assert s.evaluate_pair(data) == historical.evaluate_pair(data)
    assert s.evaluate_pair(data)['training_authorized'] is False

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def subject():
    return importlib.import_module('qualify_qwen06_teachers')


def test_exact_pairs_and_shared_views():
    s = subject()
    assert s.PAIRS == {'b4_b06': ('b4', 'b06'), 'i4_i06': ('i4', 'i06'),
                       'g4_b06': ('g4', 'b06'), 'g4_i06': ('g4', 'i06'),
                       'i8_i06': ('i8', 'i06'), 'b8_b06': ('b8', 'b06'),
                       'i8_b06': ('i8', 'b06'), 'b8_i06': ('b8', 'i06')}
    views = s.main_views()
    assert len(views) == 10
    assert views['b06_b06'] == {'model': 'b06', 'input': 'b06'}
    assert views['i8_b06'] == {'model': 'i8', 'input': 'b06'}


class Tok:
    def __init__(self, extra=False, eos=151643):
        self.eos_token_id = eos
        self.vocab = {'a': 1, 'b': 2, '<think>': 9} if extra else {'a': 1, 'b': 2}

    def get_vocab(self):
        return self.vocab


def test_cross_tokenizer_audit_not_blanket_vocab_equality():
    s = subject()
    assert s.input_audit(Tok(True), Tok(), [1, 9], 16)['teacher_unmapped_input_ids'] == [9]
    with pytest.raises(ValueError, match='embedding'):
        s.input_audit(Tok(True), Tok(), [16], 16)
    conflict = Tok()
    conflict.vocab['a'] = 3
    with pytest.raises(ValueError, match='mapping'):
        s.input_audit(Tok(), conflict, [1], 16)
    conflict.vocab = {'other': 1, 'b': 2}
    with pytest.raises(ValueError, match='mapping'):
        s.input_audit(Tok(), conflict, [1], 16)


def test_transfer_preserves_student_input_and_native_target_eos():
    s = subject()
    row = dict(prompt_token_ids=[1, 9], prefix_token_ids=[9], prompt_text='a<think>',
               prefix_text='<think>', seed=77, sampling={'seed': 77, 'max_tokens': 16383},
               eos_token_id=151645)
    result = s.for_target([row], Tok(), 16)[0]
    assert result['prompt_token_ids'] == row['prompt_token_ids']
    assert result['prefix_token_ids'] == row['prefix_token_ids']
    assert result['eos_token_id'] == 151643
    assert result['sampling']['stop_token_ids'] == [151643, 151645]
    assert row['sampling'] == {'seed': 77, 'max_tokens': 16383}


def test_protocol_explicitly_diagnostic_only():
    s = subject()
    p = s.protocol()
    assert p['training_authorized'] is False
    assert p['stop_token_ids'] == [151643, 151645]
    assert p['bootstrap']['unit'] == 'paired_question'
    assert p['engine']['tensor_parallel_size'] == 1
    assert p['response_budget'] == 16384


def test_scheduler_dependencies_and_failure_isolation():
    s = subject()
    cells = {'student': {'depends': []}, 'teacher': {'depends': []},
             'suffix': {'depends': ['student']}, 'other': {'depends': ['teacher']}}
    assert s.ready_cells(cells, {}, set()) == ['student', 'teacher']
    assert s.ready_cells(cells, {'student': 'failed', 'teacher': 'complete'}, set()) == ['other']
    assert s.blocked_cells(cells, {'student': 'failed'}) == ['suffix']
    assert s.ready_cells(cells, {}, {'student', 'teacher'}) == []


def test_health_failures_do_not_become_positive_gates():
    s = subject()
    checks = {'direct_gain': True, 'direct_ci': True, 'continuation_gain': True,
              'direct_length_stop': False}
    assert s.gate_status(checks) == 'rejected'
    assert s.gate_status(dict(checks, direct_length_stop=True, direct_ci=False)) == 'inconclusive'
    assert s.gate_status(dict(checks, direct_length_stop=True)) == 'passed'


@pytest.mark.parametrize('key', ['b06', 'i06'])
def test_real_request_path_shared_seeds_prefix_budget_and_no_answer_in_prompt(key):
    from test_qwen17_pairs import Tokenizer, selection, record
    s = subject()
    tok = Tokenizer()
    tok.eos_token_id = 151643 if key == 'b06' else 151645
    chosen = selection(s)
    for row in chosen:
        row['answer'] = 'SECRET_GOLD_ANSWER_NOT_FOR_INPUT'
    direct = s.build_requests(chosen, key, 'direct', tok)
    assert len(direct) == 128
    assert all('SECRET_GOLD' not in r['prompt_text'] for r in direct)
    raw = [record(r, 'student', 'abcdefghij' if r['sample_index'] == 0 else 'OTHER') for r in direct]
    cont = s.build_requests(chosen, key, 'continuation', tok, raw)
    assert len(cont) == 64
    for row in cont:
        assert row['prefix_text'] == 'abcde'
        assert row['sampling']['max_tokens'] == 16379
        assert row['prompt_token_ids'] == row['base_prompt_token_ids'] + row['prefix_token_ids']
        assert row['seed'] == s.old.request_seed(row['question_id'], row['sample_index'], 'continuation')
        assert row['sampling']['stop_token_ids'] == [151643, 151645]
    raw[0]['input_source'] = 'wrong_student'
    with pytest.raises(ValueError):
        s.build_requests(chosen, key, 'continuation', tok, raw)


def test_native_control_alias_requires_actual_execution_equality():
    s = subject()
    import copy
    r = dict(request_id='direct:q:0', prompt_token_ids=[1], sampling={'seed': 1}, eos_token_id=151643)
    other = dict(r, input_source='teacher_native')
    assert s.execution_signature([r]) == s.execution_signature([other])
    other = copy.deepcopy(other)
    other['prompt_token_ids'] = [2]
    assert s.execution_signature([r]) != s.execution_signature([other])


def test_gate_rejects_mismatched_continuation_questions():
    from test_qwen17_pairs import cells
    s = subject()
    data = cells()
    for label in ('student', 'teacher'):
        for r in data['continuation'][label]:
            r['question_id'] += '_wrong'
    with pytest.raises(ValueError, match='subset'):
        s.evaluate_pair(data)

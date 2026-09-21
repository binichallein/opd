import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import types

import pytest

PATH = Path(__file__).parents[1] / 'scripts/qualify_qwen17_pairs.py'


def module():
    spec = importlib.util.spec_from_file_location('qwen17_pairs', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Tokenizer:
    bos_token_id = None
    eos_token_id = 151643

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        special = {151643: b'<|endoftext|>', 151645: b'<|im_end|>'}
        return b''.join(bytes([i]) if i < 256 else special[i] for i in ids).decode(errors='replace')

    def get_vocab(self):
        return {**{str(i): i for i in range(256)}, '<|endoftext|>': 151643, '<|im_end|>': 151645}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt is True and enable_thinking is False
        text = '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return self.encode(text) if tokenize else text


def selection(m):
    return m.old.select_questions([
        {'extra_info': {'question': f'Question {i}', 'answer': '2'},
         'reward_model': {'ground_truth': '2'}} for i in range(70)], [], [])


def record(request, label, text='abcdefghij', model=None):
    ids = Tokenizer().encode(text)
    return {**copy.deepcopy(request), 'label': label, 'model': model or f'/models/{label}',
            'response_token_ids': ids, 'response_text': text,
            'combined_token_ids': request['prefix_token_ids'] + ids,
            'combined_text': request['prefix_text'] + text, 'finish_reason': 'stop',
            'stop_reason': request['eos_token_id'], 'sampled_logprobs': [-.5] * len(ids),
            'cumulative_logprob': -.5 * len(ids)}


def test_import_cpu_only_and_help():
    code = "import runpy,sys; runpy.run_path(sys.argv[1],run_name='cpu'); assert not ({'torch','vllm','transformers'} & sys.modules.keys())"
    subprocess.run([sys.executable, '-c', code, str(PATH)], check=True)
    p = subprocess.run([sys.executable, str(PATH), '--help'], capture_output=True, text=True)
    assert p.returncode == 0 and 'smoke' in p.stdout


@pytest.mark.parametrize('pair', ['base', 'instruct'])
def test_prompt_and_sampling_frozen(pair):
    m = module()
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = 151643 if pair == 'base' else 151645
    req = m.build_requests(selection(m), pair, 'direct', tokenizer)
    assert len(req) == 128 and len({r['seed'] for r in req}) == 128
    for row in req:
        assert row['enable_thinking'] is False
        assert row['sampling']['max_tokens'] == 16384
        assert row['sampling']['temperature'] == 1 and row['sampling']['top_p'] == .9
        assert row['sampling']['top_k'] == -1
        assert row['eos_token_id'] == tokenizer.eos_token_id
        assert row['seed'] == m.old.request_seed(row['question_id'], row['sample_index'], 'direct')
        if pair == 'base':
            assert row['prompt_token_ids'] == m.old.completion_input_ids(tokenizer, row['question'])
            assert '<think>' not in row['prompt_text']
        else:
            assert row['prompt_text'].endswith('<think>\n\n</think>\n\n')
            assert row['sampling']['stop_token_ids'] == [151645, 151643]


@pytest.mark.parametrize('pair', ['base', 'instruct'])
def test_continuation_uses_this_pair_student_and_exact_budget(pair):
    m = module()
    tok = Tokenizer()
    tok.eos_token_id = 151643 if pair == 'base' else 151645
    chosen = selection(m)
    req = m.build_requests(chosen, pair, 'direct', tok)
    raw = [record(r, 'student', 'abcdefghij' if r['sample_index'] == 0 else 'OTHER') for r in req]
    continuation = m.build_requests(chosen, pair, 'continuation', tok, student_rows=raw)
    assert len(continuation) == 64
    assert all(r['prefix_text'] == 'abcde' and r['sampling']['max_tokens'] == 16379 for r in continuation)
    raw[0]['pair'] = 'instruct' if pair == 'base' else 'base'
    with pytest.raises(ValueError):
        m.build_requests(chosen, pair, 'continuation', tok, student_rows=raw)


def test_smoke_checks_generated_text_not_template_think_block():
    m = module()
    tok = Tokenizer()
    tok.eos_token_id = 151645
    req = m.build_requests(selection(m), 'instruct', 'smoke', tok)
    assert len(req) == 4 and all(r['sampling']['max_tokens'] == 256 for r in req)
    good = [record(r, 'student', 'A normal answer') for r in req]
    assert m.validate_non_thinking(good)['passed'] is True
    bad = copy.deepcopy(good)
    bad[0]['response_text'] = '<think>unexpected reasoning</think>'
    with pytest.raises(ValueError, match='thinking'):
        m.validate_non_thinking(bad)


def cells():
    return {phase: {label: [
        {'question_id': f'q{i}', 'sample_index': j, 'correct': i < limit,
         'valid_boxed': True, 'length_stop': False, 'periodic': False,
         'generated_think_tags': False, 'response_length': 50}
        for i in range(n) for j in range(2)]
        for label, limit in [('student', 5), ('teacher', 25)]}
        for phase, n in [('direct', 64), ('continuation', 32)]}


def test_pair_gate_and_no_fabricated_reference():
    m = module()
    result = m.evaluate_pair(cells(), 'base')
    assert result['passed'] is True and result['training_authorized'] is False
    assert set(result['direct']['models']) == {'student', 'teacher'}
    same = cells()
    same['direct']['teacher'] = copy.deepcopy(same['direct']['student'])
    result = m.evaluate_pair(same, 'base')
    assert result['status'] == 'inconclusive' and not result['passed']


@pytest.mark.parametrize('fault', ['coverage', 'nan', 'duplicate', 'negative_continuation', 'length', 'think'])
def test_pair_gate_rejects_bad_or_insufficient_evidence(fault):
    m = module()
    data = cells()
    rows = data['direct']['teacher']
    if fault == 'coverage':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == 'nan':
        rows[0]['correct'] = float('nan')
    elif fault == 'negative_continuation':
        for row in data['continuation']['teacher']:
            row['correct'] = False
    elif fault == 'length':
        for row in rows[:13]:
            row['length_stop'] = True
    else:
        rows[0]['generated_think_tags'] = True
    assert not m.evaluate_pair(data, 'instruct')['passed']


def test_wrong_eos_and_selection_are_rejected():
    m = module()
    with pytest.raises(ValueError, match='EOS'):
        m.build_requests(selection(m), 'instruct', 'direct', Tokenizer())
    with pytest.raises(ValueError):
        m.build_requests(selection(m)[:-1], 'base', 'direct', Tokenizer())


def fixture_pipeline(tmp_path, monkeypatch, pair):
    m = module()
    chosen = selection(m)
    monkeypatch.setattr(m, 'SOURCE_SELECTION_SHA', m.old.object_hash(chosen))
    root = tmp_path / 'qualification'
    source = tmp_path / 'source.json'
    source.write_text('{}')
    monkeypatch.setattr(m, 'load_source', lambda: (
        {'selection_sha256': m.SOURCE_SELECTION_SHA}, chosen, {str(source): m.old.sha256(source)}))
    grader = tmp_path / 'grader.py'
    grader.write_text('fixture')
    monkeypatch.setattr(m.old, 'GRADER', grader)
    monkeypatch.setattr(m.old.historical, 'HISTORICAL_GRADER_SHA256', m.old.sha256(grader))
    def extract(text):
        matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        return matches[-1] if matches else None
    monkeypatch.setattr(m.old, 'load_historical_grader', lambda p: (lambda text, answer: extract(text) == answer, extract))
    tok = Tokenizer()
    tok.eos_token_id = 151643 if pair == 'base' else 151645
    monkeypatch.setattr(m.old, 'load_tokenizer', lambda path: tok)
    models = {}
    for label in ('student', 'teacher'):
        p = tmp_path / f'{pair}_{label}'
        p.mkdir()
        (p / 'weights').write_bytes(label.encode())
        models[f'{pair}_{label}'] = dict(path=p, repo=f'Qwen/{pair}_{label}', revision='a' * 40)
    def ensure(key):
        path = models[key]['path'] / 'weights'
        return {str(path): m.old.sha256(path)}
    assets = types.SimpleNamespace(MODELS=models, ensure_asset=ensure, __file__=str(PATH))
    monkeypatch.setattr(m, 'asset_helpers', lambda: assets)
    monkeypatch.setattr(m.importlib.metadata, 'version', lambda name: m.EXPECTED_VERSIONS[name])
    class Engine:
        def __init__(self, model):
            self.requests = []
            self.model = model
        def add_request(self, rid, prompt, params):
            expected = m.old.read_json(root / pair / rid.split(':')[0] / self.model.name.split('_')[-1] / 'requests.json')
            request = next(row for row in expected if row['request_id'] == rid)
            assert prompt == {'prompt_token_ids': request['prompt_token_ids']}
            assert params == request['sampling']
            self.requests.append((rid, prompt, params))
        def has_unfinished_requests(self):
            return bool(self.requests)
        def step(self):
            return [self.one_output() for _ in range(min(2, len(self.requests)))]
        def one_output(self):
            rid, prompt, params = self.requests.pop()
            # The fake model genuinely depends on the model path so mixed identities cannot pass.
            answer = '2' if self.model.name.endswith('teacher') else '3'
            ids = tok.encode(r'One step. More work. Finally \boxed{' + answer + '}') + [tok.eos_token_id]
            assert len(ids) <= params['max_tokens']
            sample = types.SimpleNamespace(token_ids=ids, logprobs=[{i: types.SimpleNamespace(logprob=-.5)} for i in ids],
                                           cumulative_logprob=-.5 * len(ids), finish_reason='stop', stop_reason=None)
            return types.SimpleNamespace(request_id=rid, prompt_token_ids=prompt['prompt_token_ids'],
                                         outputs=[sample], finished=True)
    class LLM:
        def __init__(self, **kwargs):
            assert kwargs['model'] in {str(v['path']) for v in models.values()}
            assert kwargs['tokenizer'] == str(models[f'{pair}_student']['path'])
            for key, value in m.protocol(pair)['engine'].items():
                assert kwargs[key] == value
            self.llm_engine = Engine(Path(kwargs['model']))
        def get_tokenizer(self):
            return tok
    monkeypatch.setitem(sys.modules, 'vllm', types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setattr(m.os, 'environ', dict(m.os.environ))
    m.prepare(root, pair)
    return m, root, tok


@pytest.mark.parametrize('pair', ['base', 'instruct'])
def test_whole_pipeline_preserves_and_regrades_all_cells(tmp_path, monkeypatch, pair):
    m, root, tok = fixture_pipeline(tmp_path, monkeypatch, pair)
    if pair == 'instruct':
        for label in m.LABELS:
            m.generate(root, pair, 'smoke', label)
    for phase in ('direct', 'continuation'):
        for label in m.LABELS:
            result = m.generate(root, pair, phase, label)
            assert result['num_rollouts'] == (128 if phase == 'direct' else 64)
    result = m.summarize(root, pair)
    assert result['status'] == 'passed'
    assert result['direct']['models']['student']['correct_count'] == 0
    assert result['direct']['models']['teacher']['correct_count'] == 128
    assert result['training_authorized'] is False
    assert m.old.read_sealed(root / pair / 'gate_acceptance.json') == result
    with pytest.raises(FileExistsError):
        m.summarize(root, pair)
    with pytest.raises(FileExistsError):
        m.generate(root, pair, 'direct', 'student')
    path = root / pair / 'direct/student/raw.jsonl'
    with path.open('a') as stream:
        stream.write('{}\n')
    prep, selected = m.load_preparation(root, pair)
    requests = m.build_requests(selected, pair, 'direct', tok)
    with pytest.raises(ValueError, match='hash'):
        m.read_cell(root, pair, 'direct', 'student', requests, tok, prep)


def test_instruct_cannot_skip_gpu_smoke(tmp_path, monkeypatch):
    m, root, tok = fixture_pipeline(tmp_path, monkeypatch, 'instruct')
    with pytest.raises(FileNotFoundError):
        m.generate(root, 'instruct', 'direct', 'student')


def test_completion_cannot_omit_required_hashes(tmp_path, monkeypatch):
    m, root, tok = fixture_pipeline(tmp_path, monkeypatch, 'base')
    m.generate(root, 'base', 'direct', 'student')
    path = root / 'base/direct/student/completion.json'
    value = m.old.read_sealed(path)
    value['files'].pop('raw.jsonl')
    path.write_text(json.dumps(value))
    Path(str(path) + '.sha256.json').write_text(json.dumps({'sha256': m.old.sha256(path)}))
    prep, selected = m.load_preparation(root, 'base')
    requests = m.build_requests(selected, 'base', 'direct', tok)
    with pytest.raises(ValueError, match='file inventory'):
        m.read_cell(root, 'base', 'direct', 'student', requests, tok, prep)


def test_grader_exception_preserves_finished_raw_response(tmp_path, monkeypatch):
    m, root, tok = fixture_pipeline(tmp_path, monkeypatch, 'base')
    def fail(*args):
        raise RuntimeError('injected grader failure')
    monkeypatch.setattr(m.old, 'score_records', fail)
    with pytest.raises(RuntimeError, match='grader failure'):
        m.generate(root, 'base', 'direct', 'student')
    folder = root / 'base/direct/student'
    rows = list(m.old.read_jsonl(folder / 'raw.jsonl'))
    assert len(rows) == 2 and rows[0]['response_token_ids'][-1] == 151643
    events = [json.loads(line) for line in (folder / 'engine_events.jsonl').open()]
    assert len(events) == 1 and len(events[0]) == 2
    assert (folder / 'results.jsonl').read_bytes() == b''
    assert not (folder / 'completion.json').exists()


@pytest.mark.parametrize('reason,terminal,finish', [(999999, 151643, 'stop'), ({}, 151643, 'stop'),
                                                  ('eos', 151643, 'stop'), (None, 10, 'stop'),
                                                  (151643, 151643, 'length')])
def test_invalid_stop_metadata_is_rejected(reason, terminal, finish):
    m = module()
    row = {'stop_reason': reason, 'response_token_ids': [terminal], 'finish_reason': finish,
           'eos_token_id': 151643, 'sampling': {'stop_token_ids': [151643], 'max_tokens': 1}}
    with pytest.raises(ValueError):
        m.validate_stops([row])


def test_valid_native_and_secondary_eos_and_length_priority():
    m = module()
    for reason, terminal, finish in [(None, 151645, 'stop'), (151643, 151643, 'stop'), (None, 151645, 'length')]:
        m.validate_stops([{'stop_reason': reason, 'response_token_ids': [terminal], 'finish_reason': finish,
                          'eos_token_id': 151645, 'sampling': {'stop_token_ids': [151645, 151643], 'max_tokens': 1 if finish == 'length' else 2}}])


@pytest.mark.parametrize('reason,budget', [(151643, 2), (None, 1)])
def test_native_eos_reason_and_cap_priority(reason, budget):
    m = module()
    with pytest.raises(ValueError):
        m.validate_stops([{'stop_reason': reason, 'response_token_ids': [151643], 'finish_reason': 'stop',
                          'eos_token_id': 151643, 'sampling': {'stop_token_ids': [151643], 'max_tokens': budget}}])


def test_runtime_change_fails_before_gpu(tmp_path, monkeypatch):
    m, root, tok = fixture_pipeline(tmp_path, monkeypatch, 'base')
    monkeypatch.setattr(m.importlib.metadata, 'version', lambda key: 'changed')
    with pytest.raises(ValueError, match='Runtime'):
        m.generate(root, 'base', 'direct', 'student')

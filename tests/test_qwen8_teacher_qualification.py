import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PATH = Path(__file__).parents[1] / 'scripts/qualify_qwen8_teacher.py'


def module():
    assert PATH.exists(), 'Missing teacher qualification implementation'
    spec = importlib.util.spec_from_file_location('qwen8_qualification', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Tokenizer:
    bos_token_id = None
    eos_token_id = 151643

    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8', errors='replace')

    def get_vocab(self):
        return {str(i): i for i in range(256)}


def pool_row(question, answer='2'):
    return {'env_kwargs': {'question': question, 'ground_truth': answer,
                           'data_source': 'dapo'},
            'extra_info': {'question': question, 'answer': answer, 'index': 7},
            'reward_model': {'ground_truth': answer}, 'data_source': 'dapo',
            'prompt': [{'role': 'user', 'content': question}]}


def selected(m):
    return m.select_questions((pool_row(f'Question {i}') for i in range(80)), [], [])


def raw_record(m, request, label, text=r'Reasoning steps. Finally \boxed{2}'):
    ids = Tokenizer().encode(text)
    return {**copy.deepcopy(request), 'label': label, 'model': f'/models/{label}',
            'response_token_ids': ids, 'response_text': text,
            'combined_token_ids': request['prefix_token_ids'] + ids,
            'combined_text': request['prefix_text'] + text,
            'finish_reason': 'stop', 'stop_reason': 151643,
            'sampled_logprobs': [-0.5] * len(ids), 'cumulative_logprob': -0.5 * len(ids)}


def test_import_and_help_are_cpu_only():
    m = module()
    assert callable(m.prepare) and callable(m.generate) and callable(m.summarize)
    code = ("import runpy,sys; runpy.run_path(sys.argv[1], run_name='cpu_import'); "
            "assert not ({'torch','vllm','transformers','pyarrow'} & sys.modules.keys())")
    subprocess.run([sys.executable, '-c', code, str(PATH)], check=True)
    result = subprocess.run([sys.executable, str(PATH), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert all(name in result.stdout for name in ('prepare', 'generate', 'summarize'))


def test_selection_deduplicates_excludes_and_preserves_exact_source():
    m = module()
    original = pool_row('  Unique\n question  ')
    rows = [pool_row(f'Question {i}') for i in range(80)]
    rows += [original, pool_row('Unique question')]
    chosen = m.select_questions(iter(rows), ['Question\t0'], ['Question  1'])
    assert len(chosen) == 64
    assert len({r['question_id'] for r in chosen}) == 64
    assert not {'Question 0', 'Question 1'} & {m.normalize_question(r['question']) for r in chosen}
    assert chosen == m.select_questions(iter(rows), ['Question 0'], ['Question\n1'])
    one = m.select_questions([original, pool_row('Unique question')], [], [], count=1)[0]
    assert one['source_record'] == original
    assert one['question'] == original['env_kwargs']['question']
    assert one['source_row_number'] == 0 and one['duplicate_count'] == 2
    assert [r['selection_hash'] for r in chosen] == sorted(r['selection_hash'] for r in chosen)
    metadata = {}
    clean = m.select_questions([original, pool_row('Unique question', '3'), pool_row('clean')],
                               [], [], count=1, metadata=metadata)
    assert clean[0]['question'] == 'clean'
    assert metadata['ambiguous_groups'][0]['answers'] == ['2', '3']
    assert metadata['ambiguous_groups'][0]['source_row_numbers'] == [0, 1]
    with pytest.raises(ValueError, match='64'):
        m.select_questions(rows[:63], [], [])


def test_sha_request_seeds_and_frozen_sampling_are_model_independent():
    m = module()
    rows = selected(m)
    requests = m.build_requests(rows, 'direct', Tokenizer())
    assert len(requests) == 128
    assert len({r['seed'] for r in requests}) == 128
    assert requests == m.build_requests(rows, 'direct', Tokenizer())
    for r in requests:
        assert r['seed'] == m.request_seed(r['question_id'], r['sample_index'], 'direct')
        assert r['sampling']['n'] == 1 and r['sampling']['max_tokens'] == 16384
        assert r['sampling']['temperature'] == 1 and r['sampling']['top_p'] == .9
        assert r['sampling']['top_k'] == -1 and r['sampling']['ignore_eos'] is False
        assert r['enable_thinking'] is False
        assert '<think>' not in r['prompt_text'] and '<|im_start|>' not in r['prompt_text']
    assert m.request_seed(rows[0]['question_id'], 0, 'direct') != m.request_seed(rows[0]['question_id'], 0, 'continuation')


@pytest.mark.parametrize('text', [r'ab\boxed{2} after longer reasoning', r'\boxed{2}',
                                  'abcdefghi', r'abc\boxed{unterminated', '\u4e2d\u6587abc\\boxed{2}more'])
def test_prefix_is_causal_raw_token_slice_before_first_box(text):
    m = module()
    ids = Tokenizer().encode(text)
    prefix = m.causal_prefix(ids, Tokenizer())
    assert prefix['prefix_token_ids'] == ids[:len(prefix['prefix_token_ids'])]
    assert len(prefix['prefix_token_ids']) <= len(ids) // 2
    assert text.startswith(prefix['prefix_text'])
    assert '\\boxed' not in prefix['prefix_text']
    if '\\boxed' in text:
        assert len(prefix['prefix_text']) <= text.index('\\boxed')


def test_continuation_uses_only_student_sample_zero_and_total_budget():
    m = module()
    rows = selected(m)
    direct = m.build_requests(rows, 'direct', Tokenizer())
    outputs = [raw_record(m, r, 'student', 'abcdefghij' if r['sample_index'] == 0 else 'DIFFERENT')
               for r in direct]
    continuation = m.build_requests(rows, 'continuation', Tokenizer(), student_rows=outputs)
    assert len(continuation) == 64
    assert {r['question_id'] for r in continuation} == {r['question_id'] for r in rows[:32]}
    for r in continuation:
        assert r['prefix_text'] == 'abcde'
        assert r['sampling']['max_tokens'] == 16384 - 5
        assert r['prompt_token_ids'][-5:] == list(b'abcde')
        assert r['prefix_source_sample_index'] == 0 and r['prefix_source_label'] == 'student'
    outputs[0]['label'] = 'teacher'
    with pytest.raises(ValueError, match='student|label'):
        m.build_requests(rows, 'continuation', Tokenizer(), student_rows=outputs)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'extra', 'nan', 'inf', 'short_logprobs',
                                  'prompt', 'seed', 'sampling', 'finish', 'text', 'combined', 'label'])
def test_raw_validation_rejects_adversarial_records(fault):
    m = module()
    reqs = m.build_requests(selected(m), 'direct', Tokenizer())
    rows = [raw_record(m, r, 'student') for r in reqs]
    if fault == 'missing':
        rows.pop()
    elif fault == 'duplicate':
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == 'extra':
        rows.append(copy.deepcopy(rows[0]))
    elif fault in ('nan', 'inf'):
        rows[0]['sampled_logprobs'][0] = float(fault)
    elif fault == 'short_logprobs':
        rows[0]['sampled_logprobs'].pop()
    elif fault == 'prompt':
        rows[0]['prompt_token_ids'][0] += 1
    elif fault == 'seed':
        rows[0]['seed'] += 1
    elif fault == 'sampling':
        rows[0]['sampling']['max_tokens'] = 100
    elif fault == 'finish':
        rows[0]['finish_reason'] = 'abort'
    elif fault == 'text':
        rows[0]['response_text'] = r'\boxed{9}'
    elif fault == 'combined':
        rows[0]['combined_text'] = r'\boxed{9}'
    elif fault == 'label':
        rows[0]['label'] = 'teacher'
    with pytest.raises(ValueError):
        m.validate_records(rows, reqs, 'student', Tokenizer())


def scored_cells(m, student_correct=20, teacher_correct=40):
    cells = {}
    for phase, count in [('direct', 64), ('continuation', 32)]:
        cells[phase] = {}
        for label, correct in [('student', student_correct), ('teacher', teacher_correct), ('reference', 0)]:
            cells[phase][label] = [
                {'question_id': f'q{i}', 'sample_index': j, 'correct': i < correct,
                 'valid_boxed': True, 'length_stop': False, 'periodic': False,
                 'generated_think_tags': False, 'response_length': 50}
                for i in range(count) for j in range(2)]
    return cells


def test_bootstrap_pairs_questions_not_individual_samples():
    m = module()
    result = m.paired_question_bootstrap([0, 0, 1, 1], [1, 1, 0, 0])
    assert result['difference'] == 0 and result['ci95'][0] < 0 < result['ci95'][1]
    assert result == m.paired_question_bootstrap([0, 0, 1, 1], [1, 1, 0, 0])
    perfect = m.paired_question_bootstrap([0] * 64, [1] * 64)
    assert perfect['ci95'] == [1., 1.]
    for left, right in [([], []), ([0], [0, 1]), ([float('nan')], [1])]:
        with pytest.raises(ValueError):
            m.paired_question_bootstrap(left, right)


def test_gate_passes_only_all_checks_and_reports_separate_counts():
    m = module()
    gate = m.evaluate_gate(scored_cells(m))
    assert gate['passed'] is True
    assert gate['status'] == 'passed'
    assert gate['direct']['models']['student']['correct_count'] == 40
    assert gate['direct']['models']['teacher']['num_rollouts'] == 128
    assert gate['continuation']['models']['teacher']['num_rollouts'] == 64
    assert gate['direct']['teacher_minus_student']['num_questions'] == 64
    assert gate['direct']['models']['reference']['correct_count'] == 0


@pytest.mark.parametrize('fault', ['equal', 'small_gain', 'negative_continuation', 'length',
                                  'length_relative', 'periodic', 'box', 'box_relative', 'missing', 'nan'])
def test_gate_is_fail_closed_on_coverage_health_and_insufficient_evidence(fault):
    m = module()
    cells = scored_cells(m)
    teacher = cells['direct']['teacher']
    if fault in ('equal', 'small_gain'):
        cells = scored_cells(m, 20, 20 if fault == 'equal' else 23)
    elif fault == 'negative_continuation':
        for r in cells['continuation']['teacher']:
            r['correct'] = False
    elif fault in ('length', 'length_relative', 'periodic', 'box', 'box_relative'):
        field, count = {'length': ('length_stop', 13), 'length_relative': ('length_stop', 7),
                        'periodic': ('periodic', 7), 'box': ('valid_boxed', 33),
                        'box_relative': ('valid_boxed', 7)}[fault]
        for r in teacher[:count]:
            r[field] = field != 'valid_boxed'
    elif fault == 'missing':
        teacher.pop()
    elif fault == 'nan':
        teacher[0]['correct'] = float('nan')
    gate = m.evaluate_gate(cells)
    assert gate['passed'] is False
    assert gate['failures']
    if fault in ('equal', 'small_gain', 'negative_continuation'):
        assert gate['status'] == 'inconclusive'


def test_scoring_uses_combined_text_and_historical_extractor():
    m = module()
    seen = []

    def grade(text, answer):
        seen.append((text, answer))
        return text == r'prefix\boxed{2}'

    def extract(text):
        return '2' if text.endswith(r'\boxed{2}') else None

    rows = [{'question_id': 'q', 'sample_index': 0, 'answer': '2',
             'combined_text': r'prefix\boxed{2}', 'combined_token_ids': list(range(30)),
             'response_token_ids': [1, 2], 'finish_reason': 'stop'}]
    score = m.score_records(rows, grade, extract)[0]
    assert score['correct'] and score['valid_boxed'] and seen == [(r'prefix\boxed{2}', '2')]
    for malformed in [r'\boxed{}', r'\boxed{', r'\boxed nonsense']:
        rows[0]['combined_text'] = malformed
        assert not m.score_records(rows, lambda *_: False, extract)[0]['valid_boxed']
    with pytest.raises(ValueError, match='grader'):
        m.score_records(rows, lambda *_: float('nan'), extract)


def test_summarize_missing_artifacts_writes_rejection(tmp_path):
    m = module()
    result = m.summarize(tmp_path)
    assert result['passed'] is False and result['status'] == 'rejected'
    assert json.loads((tmp_path / 'gate_acceptance.json').read_text()) == result


def test_json_reader_rejects_nonfinite_duplicate_keys_and_partial_lines(tmp_path):
    m = module()
    for text in ['{"x":NaN}\n', '{"x":Infinity}\n', '{"x":1e999}\n',
                 '{"x":1,"x":2}\n', '{"x":1']:
        path = tmp_path / 'raw.jsonl'
        path.write_text(text)
        with pytest.raises(ValueError):
            list(m.read_jsonl(path))


def test_pool_streams_all_batches_and_uses_reward_ground_truth(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    m = module()
    rows = [pool_row(f'q{i}') for i in range(65)]
    rows += [pool_row('late unique')]
    rows[-1].pop('env_kwargs')
    path = tmp_path / 'train.parquet'
    # Arrow fills absent structs with null, so exercise the real source schema.
    for row in rows:
        row.pop('env_kwargs', None)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=7)
    streamed = list(m.iter_pool(path, expected_rows=66))
    assert len(streamed) == 66
    chosen = m.select_questions(iter(streamed), [], [])
    assert all(r['answer'] == '2' for r in chosen)
    assert any(r['source_row_number'] >= 7 for r in chosen)
    with pytest.raises(ValueError, match='pool.*rows'):
        list(m.iter_pool(path, expected_rows=1791700))


def test_historical_exclusions_require_actual_complete_steps(tmp_path):
    m = module()
    run = tmp_path / 'history'
    run.mkdir()
    m.write_json(run / 'run_card.json', {'seed': 21, 'total_training_steps': 200,
                                      'student_model': str(m.STUDENT)})
    rows = []
    for i in range(32):
        rows.append({'step': 1, 'sample_index': i,
                     'source_extra_info': {'question': f'Visited {i // 8}', 'answer': '2', 'index': i // 8}})
    folder = run / 'rollouts/formal/step_000001'
    folder.mkdir(parents=True)
    path = folder / 'raw.jsonl.gz'
    with gzip.open(path, 'wt') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    path.with_suffix(path.suffix + '.sha256').write_text(f'{m.sha256(path)}  {path.name}\n')
    result = m.collect_historical_exclusions([run], steps=1)
    assert {r['question'] for r in result['questions']} == {f'Visited {i}' for i in range(4)}
    assert result['files'][str(path)] == m.sha256(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.collect_historical_exclusions([run], steps=2)
    path.with_suffix(path.suffix + '.sha256').write_text('0' * 64 + '  raw.jsonl.gz\n')
    with pytest.raises(ValueError, match='hash'):
        m.collect_historical_exclusions([run], steps=1)


def test_model_label_identity_is_not_just_an_arbitrary_path(tmp_path):
    m = module()
    model = tmp_path / 'Qwen3-8B-Base'
    model.mkdir()
    config = {'architectures': ['Qwen3ForCausalLM'], 'hidden_size': 4096,
              'num_hidden_layers': 36, 'num_attention_heads': 32, 'num_key_value_heads': 8,
              'vocab_size': 151936, 'eos_token_id': 151643,
              'max_position_embeddings': 32768}
    m.write_json(model / 'config.json', config)
    m.write_json(model / 'asset_manifest.json', {'repo': 'Qwen/Qwen3-8B-Base',
                                              'provider': 'modelscope', 'revision': 'a' * 40})
    (model / 'SOURCE_REVISION').write_text('a' * 40 + '\n')
    assert m.validate_model_identity(model, 'teacher')['repo'] == 'Qwen/Qwen3-8B-Base'
    with pytest.raises(ValueError, match='identity|label'):
        m.validate_model_identity(model, 'student')
    with pytest.raises(ValueError, match='identity|label'):
        m.validate_model_identity(model, 'reference')


def test_reference_decode_uses_own_tokenizer_without_losing_added_tokens():
    m = module()

    class ReferenceTokenizer(Tokenizer):
        def get_vocab(self):
            return {**super().get_vocab(), '<think>': 151667}

        def decode(self, ids, **kwargs):
            return ''.join('<think>' if i == 151667 else chr(i) for i in ids)

    reqs = m.build_requests(selected(m), 'direct', Tokenizer())
    rows = [raw_record(m, r, 'reference') for r in reqs]
    row = rows[0]
    row.update(response_token_ids=[151667, 65], response_text='<think>A',
               combined_token_ids=[151667, 65], combined_text='<think>A',
               sampled_logprobs=[-.5, -.5], cumulative_logprob=-1.)
    assert m.validate_records(rows, reqs, 'reference', Tokenizer(),
                              response_tokenizer=ReferenceTokenizer())
    with pytest.raises(ValueError, match='Unknown'):
        m.decode_ids(Tokenizer(), [151667])


def test_summarize_cli_rejected_gate_exits_zero_but_errors_nonzero(monkeypatch, tmp_path):
    m = module()
    monkeypatch.setattr(m, 'summarize', lambda root: {'passed': False, 'status': 'inconclusive',
                                                  'errors': []})
    assert m.main(['summarize', '--root', str(tmp_path)]) == 0
    monkeypatch.setattr(m, 'summarize', lambda root: {'passed': False, 'status': 'rejected',
                                                  'errors': ['missing artifact']})
    assert m.main(['summarize', '--root', str(tmp_path)]) != 0


def test_engine_output_must_match_prompt_and_include_every_sampled_logprob():
    from types import SimpleNamespace as NS

    m = module()
    req = m.build_requests(selected(m), 'direct', Tokenizer())[0]
    sample = NS(token_ids=[65, 66], logprobs=[{65: NS(logprob=-.5)}, {66: NS(logprob=-.5)}],
                cumulative_logprob=-1., finish_reason='stop', stop_reason=151643)
    output = NS(request_id=req['request_id'], prompt_token_ids=req['prompt_token_ids'],
                finished=True, outputs=[sample])
    record = m.engine_record(output, req, 'student', Path('/model'), Tokenizer())
    assert record['response_text'] == 'AB'
    sample.logprobs.pop()
    with pytest.raises(ValueError, match='logprob'):
        m.engine_record(output, req, 'student', Path('/model'), Tokenizer())
    output.prompt_token_ids = [9]
    with pytest.raises(ValueError, match='prompt'):
        m.engine_record(output, req, 'student', Path('/model'), Tokenizer())


@pytest.fixture
def cpu_attempt(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace as NS

    m = module()
    root = tmp_path / 'qualification'
    models = {label: tmp_path / label for label in m.LABELS}
    for model in models.values():
        model.mkdir()
        for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json'):
            m.write_json(model / name, {'fixture': True})
    pool = tmp_path / 'train.parquet'
    rows = [pool_row(f'Question {i}') for i in range(90)]
    rows += [pool_row(' ambiguous group ', '2'), pool_row('ambiguous\tgroup', '3')]
    pq.write_table(pa.Table.from_pylist(rows), pool, row_group_size=13)
    benchmarks = tmp_path / 'benchmarks'
    benchmarks.mkdir()
    for task, count in m.historical.TASK_COUNTS.items():
        m.write_jsonl(benchmarks / f'{task}.jsonl', ({'problem': f'{task} {i}'} for i in range(count)))
    grader_path = tmp_path / 'grader.py'
    grader_path.write_text('# CPU fixture only\n')
    monkeypatch.setattr(m, 'EXPECTED_POOL_ROWS', len(rows))
    monkeypatch.setattr(m, 'load_tokenizer', lambda path: Tokenizer())
    monkeypatch.setattr(m, 'load_historical_grader', lambda path: (
        lambda text, answer: text.endswith('\\boxed{' + answer + '}'),
        lambda text: text.rsplit('\\boxed{', 1)[1][:-1] if '\\boxed{' in text and text.endswith('}') else None))
    monkeypatch.setattr(m, 'collect_historical_exclusions', lambda runs: {
        'questions': [{'question': 'Question 0'}], 'files': {}, 'coverage': []})
    monkeypatch.setattr(m, 'validate_model_identity', lambda model, label, **kwargs: {'label': label})
    monkeypatch.setattr(m, 'validate_tokenizer_alignment', lambda student, model, label: {'match': True})
    monkeypatch.setattr(m, 'model_file_hashes', lambda model: {str(model / 'config.json'): m.sha256(model / 'config.json')})
    monkeypatch.setattr(m.importlib.metadata, 'version', lambda name: '0.11.0' if name == 'vllm' else 'fixture')

    class Engine:
        def __init__(self, model):
            self.model = model
            self.queue = []

        def add_request(self, request_id, prompt, params):
            assert params.n == 1 and params.detokenize is False
            assert len(prompt['prompt_token_ids']) + params.max_tokens <= 18432
            self.queue.append((request_id, prompt, params))

        def has_unfinished_requests(self):
            return bool(self.queue)

        def step(self):
            request_id, prompt, params = self.queue.pop()
            answer = '1' if self.model == str(models['student']) else '2'
            ids = list(('Reasoning carefully then \\boxed{' + answer + '}').encode())
            sample = NS(token_ids=ids, logprobs=[{i: NS(logprob=-.5)} for i in ids],
                        cumulative_logprob=-.5 * len(ids), finish_reason='stop', stop_reason=151643)
            return [NS(request_id=request_id, prompt_token_ids=prompt['prompt_token_ids'],
                       outputs=[sample], finished=True)]

    class LLM:
        def __init__(self, model, tokenizer, **kwargs):
            assert tokenizer == str(models['student'])
            assert kwargs == m.frozen_protocol()['engine']
            self.llm_engine = Engine(model)

        def get_tokenizer(self):
            return Tokenizer()

    monkeypatch.setitem(sys.modules, 'vllm', NS(LLM=LLM, SamplingParams=lambda **kwargs: NS(**kwargs)))
    m.prepare(root, train_pool=pool, benchmark_dir=benchmarks, historical_runs=[tmp_path / 'history'],
              student=models['student'], grader_path=grader_path)
    return m, root, models


def test_full_cpu_prepare_six_generation_cells_and_summary(cpu_attempt):
    m, root, models = cpu_attempt
    preparation = m.read_sealed(root / 'prepare_manifest.json')
    ambiguous = preparation['selection_metadata']['ambiguous_groups']
    assert len(ambiguous) == 1 and ambiguous[0]['answers'] == ['2', '3']
    assert m.read_json(root / 'excluded_question.json')['ambiguous_groups'] == ambiguous
    for phase in m.PHASES:
        for label in m.LABELS:
            completion = m.generate(root, models[label], label, phase)
            assert completion['num_rollouts'] == (128 if phase == 'direct' else 64)
            manifest = m.read_sealed(root / phase / label / 'manifest.json')
            assert manifest['invocation']['argv_sha256'] == m.object_hash(sys.argv)
            assert manifest['invocation']['script_sha256'] == m.sha256(PATH)
    gate = m.summarize(root)
    assert gate['passed'] is True and gate['errors'] == []
    assert len(gate['raw_integrity']) == 6
    assert gate['direct']['teacher_minus_student']['ci95'] == [1., 1.]
    assert not (root / 'benchmark').exists()
    with pytest.raises(FileExistsError):
        m.generate(root, models['student'], 'student', 'direct')


def test_raw_integrity_tamper_is_a_hard_error(cpu_attempt):
    m, root, models = cpu_attempt
    m.generate(root, models['student'], 'student', 'direct')
    raw = root / 'direct/student/raw.jsonl'
    raw.write_text(raw.read_text().replace('Reasoning', 'Different'))
    gate = m.summarize(root)
    assert gate['passed'] is False
    assert gate['errors'] and 'hash mismatch' in gate['errors'][0]


def test_continuation_rejects_wrong_prefix_even_with_correct_coverage():
    m = module()
    chosen = selected(m)
    direct = [raw_record(m, r, 'student') for r in m.build_requests(chosen, 'direct', Tokenizer())]
    requests = m.build_requests(chosen, 'continuation', Tokenizer(), student_rows=direct)
    rows = [raw_record(m, r, 'teacher') for r in requests]
    rows[0]['prefix_token_ids'] = list(b'wrong prefix')
    with pytest.raises(ValueError, match='prefix'):
        m.validate_records(rows, requests, 'teacher', Tokenizer())


def test_positive_direct_gain_is_inconclusive_when_paired_ci_crosses_zero():
    m = module()
    cells = scored_cells(m, 0, 0)
    for row in cells['direct']['student']:
        row['correct'] = int(row['question_id'][1:]) < 10
    for row in cells['direct']['teacher']:
        row['correct'] = 10 <= int(row['question_id'][1:]) < 25
    gate = m.evaluate_gate(cells)
    assert gate['direct']['teacher_minus_student']['difference'] > .05
    assert gate['direct']['teacher_minus_student']['ci95'][0] <= 0
    assert gate['passed'] is False and gate['status'] == 'inconclusive'


def test_historical_conflicting_answers_are_all_retained_and_excluded(tmp_path):
    m = module()
    run = tmp_path / 'history'
    run.mkdir()
    m.write_json(run / 'run_card.json', {'seed': 21, 'total_training_steps': 200,
                                      'student_model': str(m.STUDENT)})
    path = run / 'rollouts/formal/step_000001/raw.jsonl.gz'
    path.parent.mkdir(parents=True)
    with gzip.open(path, 'wt') as stream:
        for i in range(32):
            source = {'question': 'same question' if i < 16 else 'same\tquestion',
                      'answer': '2' if i < 16 else '3', 'index': i // 8}
            stream.write(json.dumps({'step': 1, 'sample_index': i, 'source_extra_info': source}) + '\n')
    path.with_suffix(path.suffix + '.sha256').write_text(f'{m.sha256(path)}  {path.name}\n')
    result = m.collect_historical_exclusions([run], steps=1)
    assert len(result['questions']) == 1
    assert result['questions'][0]['answers'] == ['2', '3']
    assert len(result['questions'][0]['visits']) == 4

#!/usr/bin/env python3
"""Frozen, task-level DAPO teacher diagnostic. Importing this module is CPU-only.

The controller runs prepare, all three direct cells, all three continuation cells,
then summarize. No downloads, training, benchmark evaluation, retries or scheduling
are performed here. Existing evidence is never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from opd_ext.math_protocol import completion_input_ids, completion_math_prompt
import regrade_opd_eval_external as historical

BASE = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
DATA = BASE / 'data/math_opd_dapo17k_hf_full_eval4'
STUDENT = BASE / 'models/Qwen3-4B-Base'
GRADER = BASE / 'runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/grading/historical_utils_sha04f7.py'
HISTORICAL_RUNS = (
    BASE / 'runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean',
    BASE / 'runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd',
)
LABELS = ('student', 'teacher', 'reference')
PHASES = ('direct', 'continuation')
PROTOCOL = 'qwen8_teacher_acceptance_v1'
SELECTION_SALT = 'qwen8_teacher_acceptance_20260921_selection_v1'
SEED_SALT = 'qwen8_teacher_acceptance_20260921_request_v1'
RESPONSE_BUDGET = 16384
EXPECTED_POOL_ROWS = 1791700
EXTRA_TOKENS = {151665: '<tool_response>', 151666: '</tool_response>',
                151667: '<think>', 151668: '</think>'}


def frozen_protocol():
    return {'version': PROTOCOL, 'prompt_protocol': 'qwen3_completion_boxed_v1',
            'enable_thinking': False, 'chat_template': False, 'eos_token_id': 151643,
            'direct_questions': 64, 'continuation_questions': 32, 'samples_per_question': 2,
            'response_budget': RESPONSE_BUDGET, 'max_prompt_length': 2048,
            'selection_salt': SELECTION_SALT, 'seed_salt': SEED_SALT,
            'normalization': 'collapse_unicode_whitespace_strip_v1',
            'ambiguous_answers': 'exclude_entire_normalized_question_group',
            'prefix_rule': 'student_sample0_token_midpoint_before_first_boxed_v1',
            'bootstrap': {'unit': 'paired_question', 'replicates': 10000, 'seed': 21,
                          'confidence': .95, 'quantile': 'linear'},
            'thresholds': {'direct_gain_min': .05, 'direct_ci_lower_strict_min': 0.,
                           'continuation_gain_min': 0., 'length_stop_max': .10,
                           'length_stop_excess_max': .05, 'periodic_max': .05,
                           'missing_boxed_max': .25, 'missing_boxed_excess_max': .05},
            'periodic_rule': 'last2048_min256_period1to128_min4cycles_match0.99',
            'engine': {'dtype': 'bfloat16', 'tensor_parallel_size': 1,
                       'enforce_eager': True, 'gpu_memory_utilization': .6,
                       'max_model_len': 18432, 'max_num_seqs': 32,
                       'max_num_batched_tokens': 18432, 'seed': 21,
                       'enable_prefix_caching': True, 'generation_config': 'vllm'}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON value: {value}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _loads(text):
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON float')
        return result

    return json.loads(text, parse_constant=_reject_constant, parse_float=finite_float,
                      object_pairs_hook=_unique_object)


def read_json(path):
    return _loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith('\n') or not line.strip():
                raise ValueError(f'Partial/blank JSONL line: {path}:{number}')
            row = _loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'Non-object JSONL line: {path}:{number}')
            yield row


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())


def normalize_question(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Missing/empty question')
    return ' '.join(question.split())


def select_questions(pool, benchmark_questions, historical_questions, *, count=64, metadata=None):
    """Stream and deduplicate the entire pool; preserve the first exact source row."""
    excluded = {normalize_question(q) for q in (*benchmark_questions, *historical_questions)}
    unique, groups = {}, {}
    for index, row in enumerate(pool):
        question, answer = source_question_answer(row)
        key = normalize_question(question)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError('Missing/string ground truth required')
        group = groups.setdefault(key, {'answers': set(), 'source_row_numbers': []})
        group['answers'].add(answer)
        group['source_row_numbers'].append(index)
        if key in unique:
            unique[key]['duplicate_count'] += 1
        else:
            unique[key] = {'question': question, 'answer': answer, 'source_record': row,
                           'source_row_number': index, 'duplicate_count': 1,
                           'question_id': hashlib.sha256(key.encode()).hexdigest(),
                           'selection_hash': hashlib.sha256((SELECTION_SALT + '\0' + key).encode()).hexdigest()}
    ambiguous = [{'normalized_question': q, 'question_id': unique[q]['question_id'],
                  'answers': sorted(g['answers']), 'source_row_numbers': g['source_row_numbers']}
                 for q, g in sorted(groups.items()) if len(g['answers']) > 1]
    excluded.update(g['normalized_question'] for g in ambiguous)
    if metadata is not None:
        metadata.update(pool_rows=sum(r['duplicate_count'] for r in unique.values()),
                        unique_questions=len(unique), ambiguous_groups=ambiguous)
    candidates = sorted((row for q, row in unique.items() if q not in excluded),
                        key=lambda row: (row['selection_hash'], row['question_id']))
    if len(candidates) < count:
        raise ValueError(f'Need {count} unique eligible questions, found {len(candidates)}')
    return candidates[:count]


def request_seed(question_id, sample_index, phase):
    if phase not in PHASES or type(sample_index) is not int or sample_index not in (0, 1):
        raise ValueError('Invalid request phase/sample')
    payload = f'{SEED_SALT}\0{phase}\0{question_id}\0{sample_index}'
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], 'big') & 0x7fffffff


def sampling_parameters(seed, max_tokens):
    return {'temperature': 1., 'top_p': .9, 'top_k': -1, 'max_tokens': max_tokens,
            'n': 1, 'seed': seed, 'ignore_eos': False, 'stop_token_ids': [151643],
            'logprobs': 0, 'detokenize': False, 'skip_special_tokens': False,
            'repetition_penalty': 1., 'presence_penalty': 0., 'frequency_penalty': 0.}


def decode_ids(tokenizer, ids):
    """Reject unknown IDs instead of silently dropping them during detokenization."""
    vocab_ids = set(tokenizer.get_vocab().values())
    if any(type(i) is not int or i not in vocab_ids for i in ids):
        raise ValueError('Unknown/noninteger generated token ID')
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def causal_prefix(response_token_ids, tokenizer):
    """Never re-encode a decoded prefix: tokenization at a cut need not round-trip."""
    ids = list(response_token_ids)
    raw = decode_ids(tokenizer, ids)
    marker = raw.find('\\boxed')
    boundary = len(raw) if marker < 0 else marker
    low, high = 0, len(ids) // 2
    while low < high:
        mid = (low + high + 1) // 2
        text = decode_ids(tokenizer, ids[:mid])
        if len(text) <= boundary:
            low = mid
        else:
            high = mid - 1
    text = decode_ids(tokenizer, ids[:low])
    # A byte-level token cut may leave a partial Unicode character.
    while low and not raw.startswith(text):
        low -= 1
        text = decode_ids(tokenizer, ids[:low])
    return {'prefix_token_ids': ids[:low], 'prefix_text': text}


def build_requests(selected, phase, tokenizer, *, student_rows=None):
    if phase not in PHASES or len(selected) != 64 or len({r['question_id'] for r in selected}) != 64:
        raise ValueError('Expected frozen 64-question selection and a valid phase')
    prefixes = {}
    if phase == 'continuation':
        direct = build_requests(selected, 'direct', tokenizer)
        validate_records(student_rows or [], direct, 'student', tokenizer)
        prefixes = {r['question_id']: r for r in student_rows if r['sample_index'] == 0}
    requests = []
    for row in selected[:64 if phase == 'direct' else 32]:
        base_ids = completion_input_ids(tokenizer, row['question'])
        prefix = {'prefix_token_ids': [], 'prefix_text': '', 'prefix_source_label': None,
                  'prefix_source_sample_index': None, 'prefix_source_record_sha256': None}
        if phase == 'continuation':
            source = prefixes[row['question_id']]
            prefix.update(causal_prefix(source['response_token_ids'], tokenizer),
                          prefix_source_label='student', prefix_source_sample_index=0,
                          prefix_source_record_sha256=object_hash(source))
        ids = base_ids + prefix['prefix_token_ids']
        for sample in range(2):
            seed = request_seed(row['question_id'], sample, phase)
            requests.append({'request_id': f'{phase}:{row["question_id"]}:{sample}',
                             'question_id': row['question_id'], 'question': row['question'],
                             'answer': row['answer'], 'phase': phase, 'sample_index': sample,
                             'seed': seed, 'protocol': PROTOCOL, 'enable_thinking': False,
                             'eos_token_id': 151643, 'base_prompt_token_ids': base_ids,
                             'completion_prompt_text': completion_math_prompt(row['question']),
                             'prompt_token_ids': ids, 'prompt_text': decode_ids(tokenizer, ids),
                             **prefix, 'sampling': sampling_parameters(seed, RESPONSE_BUDGET - len(prefix['prefix_token_ids']))})
    if len({r['seed'] for r in requests}) != len(requests):
        raise ValueError('SHA request seed collision')
    return requests


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_records(rows, requests, label, tokenizer, *, model=None, response_tokenizer=None):
    expected = {r['request_id']: r for r in requests}
    if label not in LABELS or len(expected) != len(requests) or len(rows) != len(requests):
        raise ValueError('Wrong label or incomplete/extra rollout coverage')
    seen, models = set(), set()
    for row in rows:
        request_id = row.get('request_id')
        if request_id not in expected or request_id in seen:
            raise ValueError('Duplicate/unexpected request coverage')
        seen.add(request_id)
        request = expected[request_id]
        if any(row.get(k) != v for k, v in request.items()) or row.get('label') != label:
            raise ValueError('Wrong request input/prefix/seed/sampling/label')
        if not isinstance(row.get('model'), str) or not row['model']:
            raise ValueError('Missing model identity')
        models.add(row['model'])
        if model is not None and row['model'] != str(model):
            raise ValueError('Wrong model identity')
        ids, lps = row.get('response_token_ids'), row.get('sampled_logprobs')
        if not isinstance(ids, list) or not ids or len(ids) > request['sampling']['max_tokens']:
            raise ValueError('Invalid response length/budget')
        if not isinstance(lps, list) or len(lps) != len(ids) or not all(_finite_number(x) for x in lps):
            raise ValueError('Missing/nonfinite sampled logprobs')
        cumulative = row.get('cumulative_logprob')
        if (not _finite_number(cumulative) or any(x > 1e-5 for x in lps)
                or not math.isclose(sum(lps), cumulative, rel_tol=1e-5, abs_tol=.01)):
            raise ValueError('Nonfinite/inconsistent cumulative logprob')
        if row.get('finish_reason') not in ('stop', 'length') or 'stop_reason' not in row:
            raise ValueError('Missing/invalid engine finish reason')
        if row['finish_reason'] == 'length' and len(ids) != request['sampling']['max_tokens']:
            raise ValueError('Length stop does not exhaust the response budget')
        raw = decode_ids(response_tokenizer or tokenizer, ids)
        if row.get('response_text') != raw:
            raise ValueError('Raw response text/IDs mismatch')
        if (row.get('combined_text') != request['prefix_text'] + raw
                or row.get('combined_token_ids') != request['prefix_token_ids'] + ids):
            raise ValueError('Combined prefix/suffix mismatch')
    if len(models) != 1:
        raise ValueError('Mixed model identities')
    return True


def paired_question_bootstrap(student, teacher, *, replicates=10000, seed=21):
    if (not student or len(student) != len(teacher) or replicates < 100
            or not all(_finite_number(x) and 0 <= x <= 1 for x in (*student, *teacher))):
        raise ValueError('Invalid paired question bootstrap inputs')
    differences = [t - s for s, t in zip(student, teacher)]
    rng = random.Random(seed)
    n = len(differences)
    draws = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(replicates))

    def quantile(q):
        index = (len(draws) - 1) * q
        lo = int(index)
        return draws[lo] + (draws[min(lo + 1, len(draws) - 1)] - draws[lo]) * (index - lo)

    return {'difference': sum(differences) / n, 'ci95': [quantile(.025), quantile(.975)],
            'num_questions': n, 'unit': 'paired_question', 'replicates': replicates, 'seed': seed}


def score_records(rows, grader, extractor):
    from diagnose_token_truncation import analyze_tokens

    result = []
    for row in rows:
        text = row['combined_text']
        score = grader(text, row['answer'])
        if type(score) not in (bool, int, float) or not math.isfinite(score) or score not in (0, 1):
            raise ValueError('Historical grader returned an invalid/nonfinite score')
        extracted = extractor(text)
        result.append({'question_id': row['question_id'], 'sample_index': row['sample_index'],
                       'correct': bool(score), 'valid_boxed': isinstance(extracted, str) and bool(extracted.strip()),
                       'extracted_answer': extracted, 'length_stop': row['finish_reason'] == 'length',
                       'periodic': analyze_tokens(row['combined_token_ids'], RESPONSE_BUDGET)['tail_period'] is not None,
                       'generated_think_tags': '<think>' in text or '</think>' in text,
                       'response_length': len(row['response_token_ids']),
                       'combined_length': len(row['combined_token_ids'])})
    return result


def _cell_statistics(rows, expected_questions):
    groups = {}
    if len(rows) != expected_questions * 2:
        raise ValueError('Incomplete scored coverage')
    for row in rows:
        if any(type(row.get(k)) is not bool for k in ('correct', 'valid_boxed', 'length_stop',
                                                     'periodic', 'generated_think_tags')):
            raise ValueError('Nonfinite/malformed scored metric')
        if type(row.get('response_length')) is not int or not 0 < row['response_length'] <= RESPONSE_BUDGET:
            raise ValueError('Invalid scored response length')
        key, sample = row['question_id'], row['sample_index']
        if not isinstance(key, str) or type(sample) is not int or sample not in (0, 1):
            raise ValueError('Invalid scored request identity')
        group = groups.setdefault(key, {})
        if sample in group:
            raise ValueError('Duplicate scored coverage')
        group[sample] = row['correct']
    if len(groups) != expected_questions or any(set(g) != {0, 1} for g in groups.values()):
        raise ValueError('Incomplete question/sample coverage')
    n = len(rows)
    result = {'num_questions': len(groups), 'num_rollouts': n,
              'per_question': {q: sum(g.values()) / 2 for q, g in groups.items()},
              'mean_response_length': sum(r['response_length'] for r in rows) / n}
    for name, key in [('correct', 'correct'), ('length_stop', 'length_stop'),
                      ('periodic', 'periodic'), ('think_tag', 'generated_think_tags')]:
        result[name + '_count'] = sum(r[key] for r in rows)
        result[name + '_rate'] = result[name + '_count'] / n
    result['missing_boxed_count'] = sum(not r['valid_boxed'] for r in rows)
    result['missing_boxed_rate'] = result['missing_boxed_count'] / n
    return result


def evaluate_gate(cells):
    result = {'passed': False, 'status': 'rejected', 'protocol': frozen_protocol(),
              'scope': 'DAPO task-level diagnostic; not benchmark evaluation or evidence of unseen pretraining data',
              'failures': []}
    try:
        if set(cells) != set(PHASES):
            raise ValueError('Missing/extra phases')
        for phase, n in [('direct', 64), ('continuation', 32)]:
            if set(cells[phase]) != set(LABELS):
                raise ValueError('Missing/extra model labels')
            models = {label: _cell_statistics(cells[phase][label], n) for label in LABELS}
            questions = sorted(models['student']['per_question'])
            if any(set(m['per_question']) != set(questions) for m in models.values()):
                raise ValueError('Mismatched paired question coverage')
            comparison = paired_question_bootstrap(
                [models['student']['per_question'][q] for q in questions],
                [models['teacher']['per_question'][q] for q in questions])
            reference = paired_question_bootstrap(
                [models['student']['per_question'][q] for q in questions],
                [models['reference']['per_question'][q] for q in questions])
            result[phase] = {'models': models, 'teacher_minus_student': comparison,
                             'reference_minus_student': reference}
        if not set(result['continuation']['models']['student']['per_question']).issubset(
                result['direct']['models']['student']['per_question']):
            raise ValueError('Continuation questions are not a subset of direct questions')
    except (KeyError, TypeError, ValueError) as exc:
        result['failures'].append(str(exc))
        return result
    checks = {}
    direct, continuation = [result[p]['teacher_minus_student'] for p in PHASES]
    checks['direct_gain'] = direct['difference'] >= .05
    checks['direct_ci'] = direct['ci95'][0] > 0
    checks['continuation_gain'] = continuation['difference'] >= 0
    for phase in PHASES:
        teacher, student = [result[phase]['models'][label] for label in ('teacher', 'student')]
        checks[phase + '_length_stop'] = teacher['length_stop_rate'] <= .10
        checks[phase + '_length_excess'] = teacher['length_stop_rate'] - student['length_stop_rate'] <= .05
        checks[phase + '_periodic'] = teacher['periodic_rate'] <= .05
        checks[phase + '_missing_boxed'] = teacher['missing_boxed_rate'] <= .25
        checks[phase + '_missing_boxed_excess'] = teacher['missing_boxed_rate'] - student['missing_boxed_rate'] <= .05
    result['checks'] = checks
    result['failures'] = [name for name, passed in checks.items() if not passed]
    result['passed'] = all(checks.values())
    result['status'] = ('passed' if result['passed'] else 'rejected' if any(
        not value for name, value in checks.items() if name not in ('direct_gain', 'direct_ci', 'continuation_gain'))
        else 'inconclusive')
    return result


def source_question_answer(row):
    extra, env = row.get('extra_info') or {}, row.get('env_kwargs') or {}
    reward = row.get('reward_model') or {}
    question = extra.get('question', env.get('question'))
    answer = reward.get('ground_truth', env.get('ground_truth', extra.get('answer')))
    normalize_question(question)
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('Missing string ground truth')
    if any(value != question for value in (extra.get('question', question), env.get('question', question))):
        raise ValueError('Source question fields disagree')
    if any(value != answer for value in (extra.get('answer', answer), env.get('ground_truth', answer))):
        raise ValueError('Source answer fields disagree')
    return question, answer


def iter_pool(path, *, expected_rows=EXPECTED_POOL_ROWS):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f'Unexpected training pool rows: {parquet.metadata.num_rows}')
    # Do not assume that the first 17,917 rows repeat; inspect every physical row.
    count = 0
    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            count += 1
            yield row
    if count != expected_rows:
        raise ValueError('Incomplete training pool rows')


def collect_historical_exclusions(runs, *, steps=200):
    if not runs:
        raise ValueError('Actual historical training archives are required')
    questions, files, coverage = {}, {}, []
    for run in map(Path, runs):
        card_path = run / 'run_card.json'
        card = read_json(card_path)
        if (card.get('seed') != 21 or card.get('total_training_steps') != 200
                or card.get('student_model') != str(STUDENT)):
            raise ValueError('Wrong historical seed21/200-step original 4B run identity')
        files[str(card_path)] = sha256(card_path)
        for step in range(1, steps + 1):
            path = run / f'rollouts/formal/step_{step:06d}/raw.jsonl.gz'
            digest = sha256(path)
            sidecar = path.with_suffix(path.suffix + '.sha256')
            if sidecar.read_text().split() != [digest, path.name]:
                raise ValueError(f'Historical rollout hash mismatch: {path}')
            files[str(path)] = digest
            files[str(sidecar)] = sha256(sidecar)
            rows = list(read_jsonl(path))
            if len(rows) != 32 or [r.get('sample_index') for r in rows] != list(range(32)):
                raise ValueError(f'Incomplete historical rollout sample coverage: {path}')
            sources = []
            for i, row in enumerate(rows):
                source = row['source_extra_info']
                if row.get('step') != step or any(k not in source for k in ('question', 'answer', 'index')):
                    raise ValueError('Wrong historical step/source metadata')
                if source != rows[i // 8 * 8]['source_extra_info']:
                    raise ValueError('Historical question group does not have eight samples')
                key = normalize_question(source['question'])
                entry = questions.setdefault(key, {'question': source['question'], 'answer': source['answer'],
                                                   'answers': [], 'normalized_question': key, 'visits': []})
                if source['answer'] not in entry['answers']:
                    entry['answers'] = sorted([*entry['answers'], source['answer']])
                if i % 8 == 0:
                    entry['visits'].append({'run': str(run), 'step': step, 'source_extra_info': source})
                    sources.append(source)
            coverage.append({'run': str(run), 'step': step, 'num_rollouts': 32, 'sources': sources})
    return {'questions': [questions[q] for q in sorted(questions)], 'files': files, 'coverage': coverage}


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


def load_historical_grader(path):
    historical.validate_grader_hash(Path(path), historical.HISTORICAL_GRADER_SHA256)
    grade = historical.load_grader(Path(path))
    extract = grade.__globals__.get('extract_answer')
    if (not callable(extract) or grade(r'\boxed{2}', '2') is not True
            or grade(r'\boxed{3}', '2') is not False or extract(r'\boxed{') is not None
            or extract(r'\boxed{2}') != '2'):
        raise ValueError('Historical grader/extractor self-test failed')
    return grade, extract


def code_hashes():
    paths = [Path(__file__), REPO / 'opd_ext/math_protocol.py',
             REPO / 'scripts/regrade_opd_eval_external.py',
             REPO / 'scripts/diagnose_token_truncation.py']
    return {str(path.resolve()): sha256(path) for path in paths}


def invocation_identity():
    argv = list(sys.argv)
    result = {'argv': argv, 'argv_sha256': object_hash(argv), 'script_sha256': sha256(__file__),
              'runtime_hashes': code_hashes(), 'python': sys.version,
              'executable': str(Path(sys.executable).resolve()),
              'executable_sha256': sha256(Path(sys.executable).resolve())}
    deployed = REPO / 'DEPLOYED_COMMIT'
    if deployed.exists():
        result['deployed_commit'] = deployed.read_text().strip()
        result['deployed_commit_sha256'] = sha256(deployed)
    result['runtime_sha256'] = object_hash(result['runtime_hashes'])
    return result


def verify_hashes(hashes):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Missing input hashes')
    for name, digest in hashes.items():
        if sha256(name) != digest:
            raise ValueError(f'Artifact hash mismatch: {name}')


def seal_json(path, value):
    write_json(path, value)
    write_json(Path(str(path) + '.sha256.json'), {'sha256': sha256(path)})


def read_sealed(path):
    if read_json(Path(str(path) + '.sha256.json')) != {'sha256': sha256(path)}:
        raise ValueError(f'Sealed artifact hash mismatch: {path}')
    return read_json(path)


def prepare(root, *, train_pool=None, benchmark_dir=None, historical_runs=None,
            student=None, grader_path=None):
    """Freeze data/exclusions and all direct requests before any model generation."""
    root = Path(root)
    train_pool, benchmark_dir = Path(train_pool or DATA / 'train.parquet'), Path(benchmark_dir or DATA / 'eval_jsonl')
    student, grader_path = Path(student or STUDENT), Path(grader_path or GRADER)
    historical_runs = HISTORICAL_RUNS if historical_runs is None else historical_runs
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ('prepare_manifest.json', 'selected.json', 'direct', 'continuation')):
        raise FileExistsError('Existing qualification attempt; no overwrite or automatic retry')
    load_historical_grader(grader_path)
    tokenizer = load_tokenizer(student)
    files = {str(train_pool): sha256(train_pool), str(grader_path): sha256(grader_path)}
    benchmarks = {}
    for task, count in historical.TASK_COUNTS.items():
        path = benchmark_dir / f'{task}.jsonl'
        rows = list(read_jsonl(path))
        if len(rows) != count or any(not isinstance(r.get('problem'), str) for r in rows):
            raise ValueError(f'Missing/malformed benchmark exclusion coverage: {task}')
        benchmarks[task] = rows
        files[str(path)] = sha256(path)
    history = collect_historical_exclusions(historical_runs)
    files.update(history['files'])
    benchmark_questions = [r['problem'] for rows in benchmarks.values() for r in rows]
    historical_questions = [r['question'] for r in history['questions']]
    selection_metadata = {}
    selected = select_questions(iter_pool(train_pool, expected_rows=EXPECTED_POOL_ROWS),
                                benchmark_questions, historical_questions, metadata=selection_metadata)
    # A changed source during the potentially long streaming pass invalidates preparation.
    verify_hashes(files)
    requests = build_requests(selected, 'direct', tokenizer)
    exclusions = {'benchmarks': benchmarks, 'history': history,
                  'ambiguous_groups': selection_metadata['ambiguous_groups'],
                  'normalized_questions': sorted({normalize_question(q) for q in benchmark_questions + historical_questions}
                      | {g['normalized_question'] for g in selection_metadata['ambiguous_groups']})}
    write_json(root / 'selected.json', selected)
    write_json(root / 'exclusions.json', exclusions)
    write_json(root / 'excluded_question.json', exclusions)
    write_json(root / 'direct_requests.json', requests)
    tokenizer_hashes = {str(student / name): sha256(student / name)
                        for name in ('tokenizer.json', 'tokenizer_config.json')}
    manifest = {'protocol': frozen_protocol(), 'student': str(student), 'grader_path': str(grader_path),
                'grader_sha256': historical.HISTORICAL_GRADER_SHA256,
                'train_pool': str(train_pool), 'train_pool_rows': EXPECTED_POOL_ROWS,
                'source_hashes': files, 'tokenizer_hashes': tokenizer_hashes,
                'artifacts': {name: sha256(root / name) for name in ('selected.json', 'exclusions.json', 'excluded_question.json', 'direct_requests.json')},
                'selection_metadata': selection_metadata,
                'selection_sha256': object_hash(selected), 'invocation': invocation_identity()}
    seal_json(root / 'prepare_manifest.json', manifest)
    return manifest


def verify_preparation(root, tokenizer=None):
    manifest = read_sealed(root / 'prepare_manifest.json')
    if manifest.get('protocol') != frozen_protocol() or manifest.get('grader_sha256') != historical.HISTORICAL_GRADER_SHA256:
        raise ValueError('Wrong frozen preparation protocol/grader')
    verify_hashes(manifest['invocation']['runtime_hashes'])
    verify_hashes(manifest['tokenizer_hashes'])
    verify_hashes({str(root / name): digest for name, digest in manifest['artifacts'].items()})
    selected, exclusions = read_json(root / 'selected.json'), read_json(root / 'exclusions.json')
    if object_hash(selected) != manifest['selection_sha256']:
        raise ValueError('Selection manifest hash mismatch')
    if len(selected) != 64 or [r['selection_hash'] for r in selected] != sorted(r['selection_hash'] for r in selected):
        raise ValueError('Wrong frozen selection count/order')
    for row in selected:
        q, a = source_question_answer(row['source_record'])
        normalized = normalize_question(q)
        if (q != row['question'] or a != row['answer'] or normalized in exclusions['normalized_questions']
                or row['question_id'] != hashlib.sha256(normalized.encode()).hexdigest()
                or row['selection_hash'] != hashlib.sha256((SELECTION_SALT + '\0' + normalized).encode()).hexdigest()):
            raise ValueError('Wrong selected source/question/exclusion identity')
    if tokenizer is not None and read_json(root / 'direct_requests.json') != build_requests(selected, 'direct', tokenizer):
        raise ValueError('Canonical direct prompt IDs differ from frozen preparation')
    return manifest, selected


def validate_model_identity(model, label, *, student=STUDENT):
    model = Path(model).resolve()
    if label not in LABELS:
        raise ValueError('Unknown model label')
    config = read_json(model / 'config.json')
    hidden, layers = (4096, 36) if label == 'teacher' else (2560, 36)
    if (config.get('architectures') != ['Qwen3ForCausalLM'] or config.get('hidden_size') != hidden
            or config.get('num_hidden_layers') != layers or config.get('vocab_size') != 151936
            or config.get('eos_token_id') != 151643 or config.get('max_position_embeddings', 0) < 18432):
        raise ValueError(f'Model architecture identity does not match label {label}')
    if label == 'reference':
        import prepare_qwen4_assets as assets

        if model != assets.TEACHER.resolve():
            raise ValueError('Reference label requires the frozen historical GRPO model identity')
        return {'repo': 'historical/Qwen3-4B-Base-GRPO', 'protected_hashes': assets.verify_teacher()}
    expected_repo = 'Qwen/Qwen3-8B-Base' if label == 'teacher' else 'Qwen/Qwen3-4B-Base'
    if label == 'student' and model != Path(student).resolve():
        raise ValueError('Student label must match the canonical original student identity')
    asset = read_json(model / 'asset_manifest.json')
    revision = (model / 'SOURCE_REVISION').read_text().strip()
    if (asset.get('repo') != expected_repo or asset.get('provider') != 'modelscope'
            or asset.get('revision') != revision or len(revision) != 40
            or any(c not in '0123456789abcdef' for c in revision)):
        raise ValueError(f'Official Base model identity does not match label {label}')
    return {'repo': expected_repo, 'revision': revision, 'asset_manifest_sha256': sha256(model / 'asset_manifest.json')}


def validate_tokenizer_alignment(student, model, label):
    from tokenizers import Tokenizer

    paths = [Path(p) / 'tokenizer.json' for p in (student, model)]
    values = [json.loads(Tokenizer.from_file(str(p)).to_str()) for p in paths]
    for component in ('model', 'normalizer', 'pre_tokenizer', 'decoder', 'post_processor'):
        if values[0].get(component) != values[1].get(component):
            raise ValueError(f'Tokenizer core component mismatch: {component}')
    additions = [{t['id']: t for t in read_json(p)['added_tokens']} for p in paths]
    if any(additions[1].get(i) != t for i, t in additions[0].items()):
        raise ValueError('Tokenizer shared added-token mapping mismatch')
    extras = {i: token for i, token in additions[1].items() if i not in additions[0]}
    expected = {i: {'id': i, 'content': text, 'single_word': False, 'lstrip': False,
                    'rstrip': False, 'normalized': False, 'special': False} for i, text in EXTRA_TOKENS.items()}
    if extras and (label != 'reference' or extras != expected):
        raise ValueError('Unexpected tokenizer extra-token mapping')
    return {'core_components_match': True, 'shared_mapping_matches': True,
            'reference_extra_tokens': extras, 'input_tokenizer': str(student),
            'output_tokenizer': str(model) if label == 'reference' else str(student)}


def model_file_hashes(model):
    model = Path(model)
    index = read_json(model / 'model.safetensors.index.json')
    weights = set(index.get('weight_map', {}).values())
    if not weights or any(not isinstance(w, str) or Path(w).name != w for w in weights):
        raise ValueError('Missing/invalid model shard index')
    if {p.name for p in model.glob('*.safetensors')} != weights:
        raise ValueError('Model shard coverage mismatch')
    names = weights | {'config.json', 'generation_config.json', 'tokenizer.json',
                       'tokenizer_config.json', 'model.safetensors.index.json'}
    names.update(name for name in ('asset_manifest.json', 'SOURCE_REVISION', 'source_metadata.json')
                 if (model / name).exists())
    return {str(model / name): sha256(model / name) for name in sorted(names)}


def engine_record(output, request, label, model, response_tokenizer):
    if (output.request_id != request['request_id'] or list(output.prompt_token_ids) != request['prompt_token_ids']
            or not output.finished or len(output.outputs) != 1):
        raise ValueError('Engine request/prompt IDs/finished sample mismatch')
    sample = output.outputs[0]
    ids = list(sample.token_ids)
    if sample.logprobs is None or len(sample.logprobs) != len(ids):
        raise ValueError('Missing sampled logprob coverage')
    logprobs = []
    for token, entry in zip(ids, sample.logprobs):
        if entry is None or token not in entry:
            raise ValueError('Missing sampled token logprob')
        logprobs.append(float(entry[token].logprob))
    raw = decode_ids(response_tokenizer, ids)
    row = {**request, 'label': label, 'model': str(model), 'response_token_ids': ids,
           'response_text': raw, 'combined_token_ids': request['prefix_token_ids'] + ids,
           'combined_text': request['prefix_text'] + raw, 'finish_reason': sample.finish_reason,
           'stop_reason': sample.stop_reason, 'sampled_logprobs': logprobs,
           'cumulative_logprob': sample.cumulative_logprob}
    validate_records([row], [request], label, response_tokenizer, model=model)
    return row


def generate(root, model, label, phase, gpu=0):
    """One standalone vLLM cell. The caller, not this worker, sequences cells."""
    import fcntl

    root, model = Path(root).resolve(), Path(model).resolve()
    if label not in LABELS or phase not in PHASES or type(gpu) is not int or gpu < 0:
        raise ValueError('Invalid generation label/phase/GPU')
    with (root / 'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _generate_cell(root, model, label, phase, gpu)


def _generate_cell(root, model, label, phase, gpu):
    folder = root / phase / label
    folder.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    preparation, selected = verify_preparation(root)
    tokenizer = load_tokenizer(preparation['student'])
    verify_preparation(root, tokenizer)
    identity = validate_model_identity(model, label, student=preparation['student'])
    alignment = validate_tokenizer_alignment(preparation['student'], model, label)
    response_tokenizer = load_tokenizer(model) if label == 'reference' else tokenizer
    grader, extractor = load_historical_grader(preparation['grader_path'])
    student_rows = None
    prefix_source_hash = None
    if phase == 'continuation':
        student_rows, _, _ = load_cell(root, 'direct', 'student', read_json(root / 'direct_requests.json'), tokenizer)
        prefix_source_hash = sha256(root / 'direct/student/raw.jsonl')
    requests = build_requests(selected, phase, tokenizer, student_rows=student_rows)
    frozen_requests = root / f'{phase}_requests.json'
    if frozen_requests.exists():
        if read_json(frozen_requests) != requests:
            raise ValueError('Frozen request IDs/prefixes differ between models')
    else:
        write_json(frozen_requests, requests)
    write_json(folder / 'requests.json', requests)
    model_hashes = model_file_hashes(model)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'vllm', 'tokenizers')}
    if not versions['vllm'].startswith('0.11.'):
        raise ValueError('The frozen diagnostic requires vLLM 0.11')
    manifest = {'protocol': frozen_protocol(), 'label': label, 'phase': phase, 'model': str(model),
                'model_identity': identity, 'model_hashes': model_hashes, 'tokenizer_alignment': alignment,
                'prepare_manifest_sha256': sha256(root / 'prepare_manifest.json'),
                'selection_sha256': preparation['selection_sha256'], 'requests_sha256': sha256(folder / 'requests.json'),
                'prefix_source_raw_sha256': prefix_source_hash,
                'grader_path': preparation['grader_path'], 'grader_sha256': preparation['grader_sha256'],
                'gpu': gpu, 'versions': versions, 'invocation': invocation_identity()}
    seal_json(folder / 'manifest.json', manifest)
    from vllm import LLM, SamplingParams

    llm = LLM(model=str(model), tokenizer=preparation['student'],
              **frozen_protocol()['engine'])
    # All raw prompt IDs are canonical Base IDs, including for the GRPO reference.
    if llm.get_tokenizer().get_vocab() != tokenizer.get_vocab():
        raise ValueError('Engine tokenizer is not the canonical student mapping')
    mapping = {r['request_id']: r for r in requests}
    for request in requests:
        llm.llm_engine.add_request(request['request_id'], {'prompt_token_ids': request['prompt_token_ids']},
                                  SamplingParams(**request['sampling']))
    seen = set()
    with (folder / 'raw.jsonl').open('x', encoding='utf-8') as raw_stream, \
            (folder / 'results.jsonl').open('x', encoding='utf-8') as score_stream:
        while llm.llm_engine.has_unfinished_requests():
            for output in llm.llm_engine.step():
                if not output.finished:
                    continue
                if output.request_id not in mapping or output.request_id in seen:
                    raise ValueError('Unexpected/duplicate engine result')
                request = mapping[output.request_id]
                record = engine_record(output, request, label, model, response_tokenizer)
                raw_stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                raw_stream.flush()
                os.fsync(raw_stream.fileno())
                score = score_records([record], grader, extractor)[0]
                score_stream.write(json.dumps(score, ensure_ascii=False, allow_nan=False) + '\n')
                score_stream.flush()
                os.fsync(score_stream.fileno())
                seen.add(output.request_id)
                print(json.dumps({'label': label, 'phase': phase, 'completed': len(seen),
                                  'total': len(requests), 'request_id': output.request_id,
                                  'response_length': len(record['response_token_ids']),
                                  'finish_reason': record['finish_reason']}), flush=True)
    if seen != set(mapping):
        raise ValueError('Incomplete engine output coverage')
    verify_hashes(model_hashes)
    verify_hashes(preparation['invocation']['runtime_hashes'])
    completion = {'complete': True, 'num_rollouts': len(seen), 'label': label, 'phase': phase,
                  'files': {name: sha256(folder / name) for name in ('manifest.json', 'requests.json', 'raw.jsonl', 'results.jsonl')}}
    seal_json(folder / 'completion.json', completion)
    return completion


def load_cell(root, phase, label, requests, tokenizer, *, verify_model_files=False):
    folder = root / phase / label
    completion = read_sealed(folder / 'completion.json')
    expected_files = {'manifest.json', 'requests.json', 'raw.jsonl', 'results.jsonl'}
    if (completion.get('complete') is not True or completion.get('label') != label
            or completion.get('phase') != phase or completion.get('num_rollouts') != len(requests)
            or set(completion.get('files', {})) != expected_files):
        raise ValueError(f'Incomplete cell evidence: {phase}/{label}')
    verify_hashes({str(folder / name): digest for name, digest in completion['files'].items()})
    manifest = read_sealed(folder / 'manifest.json')
    if (manifest.get('protocol') != frozen_protocol() or manifest.get('label') != label
            or manifest.get('phase') != phase
            or manifest.get('prepare_manifest_sha256') != sha256(root / 'prepare_manifest.json')
            or manifest.get('requests_sha256') != sha256(folder / 'requests.json')
            or manifest.get('grader_sha256') != historical.HISTORICAL_GRADER_SHA256
            or read_json(folder / 'requests.json') != requests):
        raise ValueError('Cell input/manifest/grader mismatch')
    invocation = manifest['invocation']
    if (invocation['argv_sha256'] != object_hash(invocation['argv'])
            or invocation['script_sha256'] != sha256(__file__)
            or invocation['runtime_sha256'] != object_hash(invocation['runtime_hashes'])):
        raise ValueError('Cell argv/script/runtime hash mismatch')
    verify_hashes(invocation['runtime_hashes'])
    if verify_model_files:
        verify_hashes(manifest['model_hashes'])
    expected_source = sha256(root / 'direct/student/raw.jsonl') if phase == 'continuation' else None
    if manifest.get('prefix_source_raw_sha256') != expected_source:
        raise ValueError('Continuation source is not the frozen student direct archive')
    response_tokenizer = load_tokenizer(manifest['model']) if label == 'reference' else tokenizer
    rows = list(read_jsonl(folder / 'raw.jsonl'))
    validate_records(rows, requests, label, tokenizer, model=manifest['model'], response_tokenizer=response_tokenizer)
    scores = list(read_jsonl(folder / 'results.jsonl'))
    return rows, scores, manifest


def summarize(root):
    """Revalidate all six cells and regrade; errors are distinct from a failed gate."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'gate_acceptance.json').exists():
        raise FileExistsError('Existing gate decision; no automatic overwrite/retry')
    result = {'passed': False, 'status': 'rejected', 'errors': [], 'failures': []}
    try:
        preparation, selected = verify_preparation(root)
        tokenizer = load_tokenizer(preparation['student'])
        verify_preparation(root, tokenizer)
        verify_hashes(preparation['source_hashes'])
        grader, extractor = load_historical_grader(preparation['grader_path'])
        cells, manifests, archive_hashes, student_rows = {}, {}, {}, None
        for phase in PHASES:
            requests = build_requests(selected, phase, tokenizer, student_rows=student_rows)
            if read_json(root / f'{phase}_requests.json') != requests:
                raise ValueError('Wrong frozen direct/continuation requests')
            cells[phase], manifests[phase] = {}, {}
            for label in LABELS:
                rows, saved_scores, manifest = load_cell(root, phase, label, requests, tokenizer,
                                                        verify_model_files=phase == 'direct')
                if phase == 'direct':
                    validate_model_identity(manifest['model'], label, student=preparation['student'])
                    validate_tokenizer_alignment(preparation['student'], manifest['model'], label)
                else:
                    first = manifests['direct'][label]
                    if any(manifest[k] != first[k] for k in ('model', 'model_hashes', 'model_identity', 'versions')):
                        raise ValueError('Model/runtime identity changed between phases')
                scores = score_records(rows, grader, extractor)
                if scores != saved_scores:
                    raise ValueError('Historical grader results do not match raw archive regrading')
                if label == 'student' and phase == 'direct':
                    student_rows = rows
                cells[phase][label], manifests[phase][label] = scores, manifest
                folder = root / phase / label
                archive_hashes[f'{phase}/{label}'] = {name: sha256(folder / name)
                    for name in ('manifest.json', 'completion.json', 'raw.jsonl', 'results.jsonl')}
        if len({manifests['direct'][label]['model'] for label in LABELS}) != 3:
            raise ValueError('Student/teacher/reference must have distinct model identities')
        result = evaluate_gate(cells)
        result.update(errors=[], grader_sha256=historical.HISTORICAL_GRADER_SHA256,
                      prepare_manifest_sha256=sha256(root / 'prepare_manifest.json'),
                      raw_integrity=archive_hashes, invocation=invocation_identity())
        if result['status'] == 'rejected' and 'checks' not in result:
            result['errors'] = list(result['failures'])
    except Exception as exc:
        result.update(passed=False, status='rejected', errors=[f'{type(exc).__name__}: {exc}'],
                      failures=[f'{type(exc).__name__}: {exc}'])
    seal_json(root / 'gate_acceptance.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'generate', 'summarize'):
        sub = commands.add_parser(command)
        sub.add_argument('--root', type=Path, required=True)
        if command == 'generate':
            sub.add_argument('--model', type=Path, required=True)
            sub.add_argument('--label', choices=LABELS, required=True)
            sub.add_argument('--phase', choices=PHASES, required=True)
            sub.add_argument('--gpu', type=int, default=0)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    result = globals()[command](**args)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 1 if command == 'summarize' and result.get('errors') else 0


if __name__ == '__main__':
    raise SystemExit(main())

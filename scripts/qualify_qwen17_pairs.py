#!/usr/bin/env python3
"""Read-only model capability diagnostics; never authorizes or launches training."""

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_qwen8_teacher as old

PAIRS = ('base', 'instruct')
LABELS = ('student', 'teacher')
SOURCE = old.BASE / 'runs/20260921v1_qwen4_from8b_completion_seed21_ml2/teacher_qualification'
SOURCE_GATE_SHA = 'ef60b0889ff0041432ddea91c59bf9bf5641dde716f3df62081bd8bd9bc396f8'
SOURCE_SELECTION_SHA = '8094e5d8c97c77a654f695b13943a227bb4a1e3fe891fdb3fbce63c19de51f3b'
VERSION = 'qwen8_to17_diagnostic_v1'
EXPECTED_VERSIONS = {'torch': '2.8.0', 'transformers': '4.57.6', 'vllm': '0.11.0', 'tokenizers': '0.22.2'}


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise ValueError(f'Runtime versions differ from frozen ml2 environment: {versions}')
    return versions


def validate_stops(rows):
    # Pinned vLLM0.11 checks length first, then native EOS (None), then explicit stop IDs.
    for row in rows:
        reason, terminal = row['stop_reason'], row['response_token_ids'][-1]
        if row['finish_reason'] == 'length':
            if reason is not None or len(row['response_token_ids']) != row['sampling']['max_tokens']:
                raise ValueError('Invalid length stop metadata')
        elif row['finish_reason'] == 'stop':
            if len(row['response_token_ids']) >= row['sampling']['max_tokens']:
                raise ValueError('Token cap has priority over EOS stop in pinned vLLM')
            if reason is None:
                valid = terminal == row['eos_token_id']
            else:
                valid = (type(reason) is int and reason != row['eos_token_id']
                         and reason in row['sampling']['stop_token_ids'] and terminal == reason)
            if not valid:
                raise ValueError('Stop metadata does not match native EOS/configured token stops')
        else:
            raise ValueError('Invalid finish reason')


def protocol(pair):
    if pair not in PAIRS:
        raise ValueError('Unknown pair')
    result = old.frozen_protocol()
    result.update(version=VERSION, pair=pair, chat_template=pair == 'instruct',
                  prompt_protocol='qwen3_completion_boxed_v1' if pair == 'base' else 'qwen3_native_chat_no_thinking_boxed_v1',
                  eos_token_id=151643 if pair == 'base' else 151645,
                  stop_token_ids=[151643] if pair == 'base' else [151645, 151643],
                  source_selection_sha256=SOURCE_SELECTION_SHA,
                  source_scope='Same diagnostic questions selected before old 4B/8B outputs; not newly unseen data',
                  smoke_questions=4, smoke_max_tokens=256, training_authorized=False)
    return result


def prompt_input(tokenizer, question, pair):
    p = protocol(pair)
    if tokenizer.eos_token_id != p['eos_token_id']:
        raise ValueError('Native EOS disagrees with frozen pair protocol')
    messages = None
    if pair == 'base':
        ids = old.completion_input_ids(tokenizer, question)
        text = old.completion_math_prompt(question)
    else:
        content = question.strip() + '\n\nPlease solve the problem step by step and put the final answer in \\boxed{}.'
        messages = [{'role': 'user', 'content': content}]
        kwargs = dict(add_generation_prompt=True, enable_thinking=False)
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        if not text.endswith('<think>\n\n</think>\n\n'):
            raise ValueError('Native non-thinking prompt must prefill the empty closed think block')
        if old.decode_ids(tokenizer, ids) != text:
            raise ValueError('Native chat text/token IDs disagree')
    if not ids or len(ids) > p['max_prompt_length']:
        raise ValueError('Prompt outside frozen length budget')
    return {'base_prompt_token_ids': ids, 'completion_prompt_text': text, 'messages': messages}


def build_requests(selected, pair, phase, tokenizer, *, student_rows=None):
    p = protocol(pair)
    if phase not in ('direct', 'continuation', 'smoke') or len(selected) != 64 or len({r['question_id'] for r in selected}) != 64:
        raise ValueError('Invalid phase or frozen selection')
    if phase == 'smoke' and pair != 'instruct':
        raise ValueError('Only the instruct pair has a thinking smoke test')
    prefixes = {}
    if phase == 'continuation':
        direct = build_requests(selected, pair, 'direct', tokenizer)
        old.validate_records(student_rows or [], direct, 'student', tokenizer)
        prefixes = {r['question_id']: r for r in student_rows if r['sample_index'] == 0}
    requests = []
    count = {'direct': 64, 'continuation': 32, 'smoke': 4}[phase]
    for row in selected[:count]:
        prompt = prompt_input(tokenizer, row['question'], pair)
        prefix = dict(prefix_token_ids=[], prefix_text='', prefix_source_label=None,
                      prefix_source_sample_index=None, prefix_source_record_sha256=None)
        if phase == 'continuation':
            source = prefixes[row['question_id']]
            prefix.update(old.causal_prefix(source['response_token_ids'], tokenizer),
                          prefix_source_label='student', prefix_source_sample_index=0,
                          prefix_source_record_sha256=old.object_hash(source))
        ids = prompt['base_prompt_token_ids'] + prefix['prefix_token_ids']
        for sample in range(1 if phase == 'smoke' else 2):
            seed = old.request_seed(row['question_id'], sample, 'direct' if phase == 'smoke' else phase)
            sampling = old.sampling_parameters(seed, (256 if phase == 'smoke' else 16384) - len(prefix['prefix_token_ids']))
            sampling['stop_token_ids'] = p['stop_token_ids']
            requests.append(dict(request_id=f'{phase}:{row["question_id"]}:{sample}',
                                 question_id=row['question_id'], question=row['question'], answer=row['answer'],
                                 phase=phase, pair=pair, sample_index=sample, seed=seed, protocol=VERSION,
                                 enable_thinking=False, eos_token_id=p['eos_token_id'],
                                 **prompt, prompt_token_ids=ids, prompt_text=old.decode_ids(tokenizer, ids),
                                 **prefix, sampling=sampling))
    if len({r['seed'] for r in requests}) != len(requests):
        raise ValueError('Request seed collision')
    return requests


def validate_non_thinking(rows):
    if not rows or any('<think>' in r['response_text'] or '</think>' in r['response_text'] for r in rows):
        raise ValueError('Generated thinking tags despite explicit non-thinking prompt; evidence retained')
    return {'passed': True, 'num_rollouts': len(rows), 'generated_think_tags': 0,
            'scope': 'Generated suffix only; input template empty think block is not generated reasoning'}


def archive_engine_batch(stream, outputs):
    """Keep every delivered completion before any individual validation/grading can fail.

    repr fields preserve diagnostic evidence even for a malformed or nonfinite engine
    record; canonical numeric logprobs remain in the separately validated raw archive.
    """
    records = [{'request_id': output.request_id, 'prompt_token_ids': list(output.prompt_token_ids),
                'finished': output.finished,
                'outputs': [{'token_ids': list(sample.token_ids), 'finish_reason': sample.finish_reason,
                             'stop_reason_repr': repr(sample.stop_reason),
                             'sampled_logprobs_repr': repr(sample.logprobs),
                             'cumulative_logprob_repr': repr(sample.cumulative_logprob)}
                            for sample in output.outputs]} for output in outputs]
    stream.write(json.dumps(records, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def evaluate_pair(cells, pair):
    result = {'pair': pair, 'protocol': protocol(pair), 'passed': False, 'status': 'rejected',
              'training_authorized': False, 'failures': [],
              'scope': 'Fixed DAPO diagnostic, not four-benchmark evaluation'}
    try:
        if set(cells) != {'direct', 'continuation'}:
            raise ValueError('Missing or extra phases')
        for phase, n in [('direct', 64), ('continuation', 32)]:
            if set(cells[phase]) != set(LABELS):
                raise ValueError('Wrong pair label coverage')
            models = {label: old._cell_statistics(cells[phase][label], n) for label in LABELS}
            questions = sorted(models['student']['per_question'])
            if set(models['teacher']['per_question']) != set(questions):
                raise ValueError('Mismatched paired questions')
            comparison = old.paired_question_bootstrap(
                [models['student']['per_question'][q] for q in questions],
                [models['teacher']['per_question'][q] for q in questions])
            result[phase] = {'models': models, 'teacher_minus_student': comparison}
        if not set(result['continuation']['models']['student']['per_question']).issubset(result['direct']['models']['student']['per_question']):
            raise ValueError('Wrong continuation subset')
    except (KeyError, TypeError, ValueError) as exc:
        result['failures'] = [str(exc)]
        return result
    direct, continuation = [result[p]['teacher_minus_student'] for p in ('direct', 'continuation')]
    checks = {'direct_gain': direct['difference'] >= .05, 'direct_ci': direct['ci95'][0] > 0,
              'continuation_gain': continuation['difference'] >= 0}
    for phase in ('direct', 'continuation'):
        teacher, student = [result[phase]['models'][label] for label in ('teacher', 'student')]
        checks.update({phase + '_length_stop': teacher['length_stop_rate'] <= .1,
                       phase + '_length_excess': teacher['length_stop_rate'] - student['length_stop_rate'] <= .05,
                       phase + '_periodic': teacher['periodic_rate'] <= .05,
                       phase + '_missing_boxed': teacher['missing_boxed_rate'] <= .25,
                       phase + '_missing_boxed_excess': teacher['missing_boxed_rate'] - student['missing_boxed_rate'] <= .05})
        if pair == 'instruct':
            checks[phase + '_no_generated_thinking'] = teacher['think_tag_count'] == student['think_tag_count'] == 0
    result['checks'] = checks
    result['failures'] = [name for name, passed in checks.items() if not passed]
    result['passed'] = all(checks.values())
    health_failures = set(result['failures']) - {'direct_gain', 'direct_ci', 'continuation_gain'}
    result['status'] = 'passed' if result['passed'] else 'rejected' if health_failures else 'inconclusive'
    return result


def asset_helpers():
    import prepare_qwen17_pair_assets as assets
    return assets


def invocation():
    result = old.invocation_identity()
    assets = asset_helpers()
    result['runtime_hashes'].update({str(Path(__file__).resolve()): old.sha256(__file__),
                                     str(Path(assets.__file__).resolve()): old.sha256(assets.__file__)})
    result['runtime_sha256'] = old.object_hash(result['runtime_hashes'])
    result['script_sha256'] = old.sha256(__file__)
    return result


def load_source():
    if old.sha256(SOURCE / 'gate_acceptance.json') != SOURCE_GATE_SHA:
        raise ValueError('Old gate identity changed')
    old.read_sealed(SOURCE / 'gate_acceptance.json')
    manifest, selected = old.verify_preparation(SOURCE)
    if old.object_hash(selected) != SOURCE_SELECTION_SHA:
        raise ValueError('Wrong previously frozen question selection')
    old.verify_hashes(manifest['source_hashes'])
    source_files = {str(SOURCE / name): old.sha256(SOURCE / name) for name in (
        'prepare_manifest.json', 'prepare_manifest.json.sha256.json', 'selected.json', 'exclusions.json',
        'gate_acceptance.json', 'gate_acceptance.json.sha256.json')}
    return manifest, selected, source_files


def prepare(root, pair):
    p = protocol(pair)
    folder = Path(root) / pair
    folder.mkdir(parents=True, exist_ok=False)
    previous, selected, source_files = load_source()
    assets = asset_helpers()
    models, hashes = {}, {}
    for label in LABELS:
        key = f'{pair}_{label}'
        hashes[label] = assets.ensure_asset(key)
        spec = assets.MODELS[key]
        models[label] = {'key': key, 'path': str(spec['path']), 'repo': spec['repo'], 'revision': spec['revision']}
    student, teacher = [old.load_tokenizer(models[label]['path']) for label in LABELS]
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError('Pair tokenizer mappings differ')
    direct = build_requests(selected, pair, 'direct', student)
    if direct != build_requests(selected, pair, 'direct', teacher):
        raise ValueError('Teacher/student native templates or token inputs differ')
    old.load_historical_grader(old.GRADER)
    old.write_json(folder / 'selected.json', selected)
    old.write_json(folder / 'direct_requests.json', direct)
    manifest = dict(protocol=p, models=models, model_hashes=hashes, source_hashes=source_files,
                    versions=runtime_versions(),
                    grader_path=str(old.GRADER), grader_sha256=old.historical.HISTORICAL_GRADER_SHA256,
                    selection_sha256=old.object_hash(selected), source_selection_sha256=previous['selection_sha256'],
                    artifacts={name: old.sha256(folder / name) for name in ('selected.json', 'direct_requests.json')},
                    invocation=invocation())
    old.seal_json(folder / 'prepare_manifest.json', manifest)
    return manifest


def load_preparation(root, pair):
    folder = Path(root) / pair
    m = old.read_sealed(folder / 'prepare_manifest.json')
    if m.get('protocol') != protocol(pair) or m.get('selection_sha256') != SOURCE_SELECTION_SHA:
        raise ValueError('Wrong pair preparation protocol/selection')
    if m.get('versions') != runtime_versions():
        raise ValueError('Pair runtime drift since preparation')
    old.verify_hashes(m['source_hashes'])
    old.verify_hashes(m['invocation']['runtime_hashes'])
    old.verify_hashes({str(folder / name): digest for name, digest in m['artifacts'].items()})
    if old.sha256(m['grader_path']) != m['grader_sha256'] or m['grader_sha256'] != old.historical.HISTORICAL_GRADER_SHA256:
        raise ValueError('Wrong historical grader')
    selected = old.read_json(folder / 'selected.json')
    if old.object_hash(selected) != SOURCE_SELECTION_SHA:
        raise ValueError('Wrong selected questions')
    return m, selected


def read_cell(root, pair, phase, label, requests, tokenizer, preparation):
    folder = Path(root) / pair / phase / label
    complete = old.read_sealed(folder / 'completion.json')
    if (complete.get('complete') is not True or complete.get('pair') != pair
            or complete.get('label') != label or complete.get('phase') != phase
            or complete.get('num_rollouts') != len(requests)):
        raise ValueError('Wrong completed cell identity/coverage')
    if set(complete.get('files', {})) != {'manifest.json', 'requests.json', 'raw.jsonl', 'results.jsonl', 'engine_events.jsonl'}:
        raise ValueError('Wrong completion file inventory')
    old.verify_hashes({str(folder / name): digest for name, digest in complete['files'].items()})
    m = old.read_sealed(folder / 'manifest.json')
    model = preparation['models'][label]
    if (m.get('protocol') != protocol(pair) or m.get('model') != model
            or m.get('phase') != phase or m.get('label') != label
            or m.get('versions') != preparation['versions']
            or m.get('prepare_manifest_sha256') != old.sha256(Path(root) / pair / 'prepare_manifest.json')
            or m.get('requests_sha256') != old.sha256(folder / 'requests.json')
            or old.read_json(folder / 'requests.json') != requests):
        raise ValueError('Wrong cell manifest/request identity')
    old.verify_hashes(m['invocation']['runtime_hashes'])
    rows = list(old.read_jsonl(folder / 'raw.jsonl'))
    old.validate_records(rows, requests, label, tokenizer, model=model['path'])
    validate_stops(rows)
    scores = list(old.read_jsonl(folder / 'results.jsonl'))
    grade, extract = old.load_historical_grader(preparation['grader_path'])
    if scores != old.score_records(rows, grade, extract):
        raise ValueError('Saved scores differ from historical regrading')
    if phase == 'smoke':
        if old.read_sealed(folder / 'non_thinking_acceptance.json') != validate_non_thinking(rows):
            raise ValueError('Incorrect GPU non-thinking acceptance')
    return rows, scores


def requests_for_cell(root, pair, phase, tokenizer, m, selected):
    student_rows = None
    if phase == 'continuation':
        direct = build_requests(selected, pair, 'direct', tokenizer)
        student_rows, _ = read_cell(root, pair, 'direct', 'student', direct, tokenizer, m)
    return build_requests(selected, pair, phase, tokenizer, student_rows=student_rows)


def generate(root, pair, phase, label, gpu=0):
    root = Path(root)
    if pair not in PAIRS or phase not in ('direct', 'continuation', 'smoke') or label not in LABELS or gpu != 0:
        raise ValueError('Unapproved pair/phase/label/GPU')
    with (root / 'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return generate_cell(root, pair, phase, label, gpu)


def generate_cell(root, pair, phase, label, gpu):
    folder = root / pair / phase / label
    folder.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    m, selected = load_preparation(root, pair)
    old.verify_hashes(m['model_hashes'][label])
    tokenizer = old.load_tokenizer(m['models']['student']['path'])
    model_tokenizer = old.load_tokenizer(m['models'][label]['path'])
    if tokenizer.get_vocab() != model_tokenizer.get_vocab():
        raise ValueError('Pair tokenizer changed')
    if pair == 'instruct' and phase != 'smoke':
        smoke = build_requests(selected, pair, 'smoke', tokenizer)
        for model_label in LABELS:
            read_cell(root, pair, 'smoke', model_label, smoke, tokenizer, m)
    requests = requests_for_cell(root, pair, phase, tokenizer, m, selected)
    if requests != requests_for_cell(root, pair, phase, model_tokenizer, m, selected):
        raise ValueError('Pair input IDs/template disagreement')
    old.write_json(folder / 'requests.json', requests)
    manifest = {'protocol': protocol(pair), 'phase': phase, 'label': label, 'model': m['models'][label],
                'prepare_manifest_sha256': old.sha256(root / pair / 'prepare_manifest.json'),
                'requests_sha256': old.sha256(folder / 'requests.json'),
                'versions': runtime_versions(),
                'gpu': gpu, 'invocation': invocation()}
    old.seal_json(folder / 'manifest.json', manifest)
    grade, extract = old.load_historical_grader(m['grader_path'])
    from vllm import LLM, SamplingParams
    model = m['models'][label]['path']
    llm = LLM(model=model, tokenizer=m['models']['student']['path'], **protocol(pair)['engine'])
    if llm.get_tokenizer().get_vocab() != tokenizer.get_vocab():
        raise ValueError('Engine tokenizer differs from canonical pair tokenizer')
    for request in requests:
        llm.llm_engine.add_request(request['request_id'], {'prompt_token_ids': request['prompt_token_ids']},
                                  SamplingParams(**request['sampling']))
    mapping, seen, rows = {r['request_id']: r for r in requests}, set(), []
    with (folder / 'raw.jsonl').open('x', encoding='utf-8') as raw, \
            (folder / 'results.jsonl').open('x', encoding='utf-8') as scored, \
            (folder / 'engine_events.jsonl').open('x', encoding='utf-8') as events:
        while llm.llm_engine.has_unfinished_requests():
            outputs = [output for output in llm.llm_engine.step() if output.finished]
            if not outputs:
                continue
            archive_engine_batch(events, outputs)
            pending = []
            for output in outputs:
                if output.request_id not in mapping or output.request_id in seen:
                    raise ValueError('Duplicate or unexpected engine output')
                row = old.engine_record(output, mapping[output.request_id], label, model, tokenizer)
                raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                seen.add(output.request_id)
                pending.append(row)
            raw.flush()
            os.fsync(raw.fileno())
            for row in pending:
                validate_stops([row])
                score = old.score_records([row], grade, extract)[0]
                scored.write(json.dumps(score, ensure_ascii=False, allow_nan=False) + '\n')
                scored.flush()
                os.fsync(scored.fileno())
                rows.append(row)
                print(json.dumps(dict(pair=pair, phase=phase, label=label, completed=len(rows), total=len(requests),
                                      response_length=len(row['response_token_ids']), finish_reason=row['finish_reason'])), flush=True)
    old.validate_records(rows, requests, label, tokenizer, model=model)
    old.verify_hashes(m['model_hashes'][label])
    load_preparation(root, pair)
    if phase == 'smoke':
        old.seal_json(folder / 'non_thinking_acceptance.json', validate_non_thinking(rows))
    completion = dict(complete=True, pair=pair, phase=phase, label=label, num_rollouts=len(rows),
                      files={name: old.sha256(folder / name) for name in ('manifest.json', 'requests.json', 'raw.jsonl', 'results.jsonl', 'engine_events.jsonl')})
    old.seal_json(folder / 'completion.json', completion)
    return completion


def summarize(root, pair):
    root = Path(root)
    if (root / pair / 'gate_acceptance.json').exists():
        raise FileExistsError('Existing pair decision; never overwrite or automatically retry')
    m, selected = load_preparation(root, pair)
    tokenizer = old.load_tokenizer(m['models']['student']['path'])
    for label in LABELS:
        old.verify_hashes(m['model_hashes'][label])
        if pair == 'instruct':
            smoke = build_requests(selected, pair, 'smoke', tokenizer)
            read_cell(root, pair, 'smoke', label, smoke, tokenizer, m)
    cells = {}
    evidence = {}
    for phase in ('direct', 'continuation'):
        requests = requests_for_cell(root, pair, phase, tokenizer, m, selected)
        cells[phase] = {}
        for label in LABELS:
            _, cells[phase][label] = read_cell(root, pair, phase, label, requests, tokenizer, m)
            path = root / pair / phase / label / 'completion.json'
            evidence[str(path)] = old.sha256(path)
    result = evaluate_pair(cells, pair)
    result.update(evidence=evidence, prepare_manifest_sha256=old.sha256(root / pair / 'prepare_manifest.json'),
                  selection_sha256=SOURCE_SELECTION_SHA, invocation=invocation())
    old.seal_json(root / pair / 'gate_acceptance.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke', 'generate', 'summarize'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pair', choices=PAIRS, required=True)
    parser.add_argument('--phase', choices=('direct', 'continuation'))
    parser.add_argument('--label', choices=LABELS)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.root, args.pair)
    elif args.action == 'summarize':
        result = summarize(args.root, args.pair)
    else:
        phase = 'smoke' if args.action == 'smoke' else args.phase
        result = generate(args.root, args.pair, phase, args.label, args.gpu)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

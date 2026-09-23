#!/usr/bin/env python3
"""Inference-only Qwen0.6 teacher screen, with exact student-input main views."""

import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path

import prepare_qwen06_screen_assets as assets
import qualify_qwen17_pairs as prior

old = prior.old
PAIRS = {'b4_b06': ('b4', 'b06'), 'i4_i06': ('i4', 'i06'),
         'g4_b06': ('g4', 'b06'), 'g4_i06': ('g4', 'i06'),
         'i8_i06': ('i8', 'i06'), 'b8_b06': ('b8', 'b06'),
         'i8_b06': ('i8', 'b06'), 'b8_i06': ('b8', 'i06')}
VERSION = 'qwen06_student_input_teacher_screen_v1'
STOP_IDS = [151643, 151645]
STUDENTS = ('b06', 'i06')


def protocol():
    p = old.frozen_protocol()
    for key in ('chat_template', 'eos_token_id', 'prompt_protocol'):
        p.pop(key)
    p.update(version=VERSION, training_authorized=False, full_benchmark_eval=False,
             input_rule='student token IDs unchanged; teacher-native direct control separate',
             stop_token_ids=STOP_IDS, native_eos='each model tokenizer EOS, recorded per cell',
             source_selection_sha256=prior.SOURCE_SELECTION_SHA,
             source_scope='Previously selected DAPO questions, not newly unseen data; GRPO teacher trained on DAPO',
             smoke_questions=4, smoke_max_tokens=256, bootstrap_multiplicity_corrected=False,
             shared_cells_are_not_independent_replicates=True)
    return p


def main_views():
    views = {f'{s}_{s}': dict(model=s, input=s) for s in STUDENTS}
    views.update({p: dict(model=t, input=s) for p, (t, s) in PAIRS.items()})
    return views


def input_audit(source, target, ids, vocab_size):
    sv, tv = source.get_vocab(), target.get_vocab()
    si, ti = {v: k for k, v in sv.items()}, {v: k for k, v in tv.items()}
    if (any(sv[k] != tv[k] for k in sv.keys() & tv.keys()) or
            any(si[k] != ti[k] for k in si.keys() & ti.keys())):
        raise ValueError('Shared tokenizer mapping conflict')
    if any(type(i) is not int or i < 0 or i >= vocab_size for i in ids):
        raise ValueError('Input ID outside teacher embedding range')
    return dict(shared_mapping_consistent=True, teacher_embedding_vocab_size=vocab_size,
                teacher_unmapped_input_ids=sorted(set(ids) - set(ti)),
                input_ids_sha256=old.object_hash(ids))


def for_target(requests, tokenizer, vocab_size):
    rows = copy.deepcopy(requests)
    for row in rows:
        if any(type(i) is not int or i < 0 or i >= vocab_size for i in row['prompt_token_ids']):
            raise ValueError('Input ID outside target embedding range')
        row['eos_token_id'] = tokenizer.eos_token_id
        row['sampling']['stop_token_ids'] = list(STOP_IDS)
    return rows


def prompt_input(tokenizer, question, key):
    if key.startswith('b'):
        return prior.prompt_input(tokenizer, question, 'base')
    messages = [{'role': 'user', 'content': question.strip() +
                 '\n\nPlease solve the problem step by step and put the final answer in \\boxed{}.'}]
    kwargs = dict(add_generation_prompt=True, enable_thinking=False)
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if not text.endswith('<think>\n\n</think>\n\n') or old.decode_ids(tokenizer, ids) != text:
        raise ValueError('Native chat non-thinking template/ID mismatch')
    if not ids or len(ids) > protocol()['max_prompt_length']:
        raise ValueError('Prompt exceeds budget')
    return dict(base_prompt_token_ids=ids, completion_prompt_text=text, messages=messages)


def build_requests(selected, source_key, phase, tokenizer, student_rows=None):
    if len(selected) != 64 or len({r['question_id'] for r in selected}) != 64:
        raise ValueError('Wrong frozen question coverage')
    prefixes = {}
    if phase == 'continuation':
        if source_key not in STUDENTS or not student_rows:
            raise ValueError('Continuation requires original student direct outputs')
        old.validate_records(student_rows, build_requests(selected, source_key, 'direct', tokenizer),
                             'student', tokenizer)
        prefixes = {r['question_id']: r for r in student_rows if r['sample_index'] == 0}
    count = {'direct': 64, 'continuation': 32, 'smoke': 4}[phase]
    requests = []
    for selected_row in selected[:count]:
        qid = selected_row['question_id']
        prompt = prompt_input(tokenizer, selected_row['question'], source_key)
        prefix = dict(prefix_token_ids=[], prefix_text='', prefix_source_label=None,
                      prefix_source_sample_index=None, prefix_source_record_sha256=None)
        if phase == 'continuation':
            source = prefixes[qid]
            prefix.update(old.causal_prefix(source['response_token_ids'], tokenizer),
                          prefix_source_label=source_key, prefix_source_sample_index=0,
                          prefix_source_record_sha256=old.object_hash(source))
        ids = prompt['base_prompt_token_ids'] + prefix['prefix_token_ids']
        for sample in range(1 if phase == 'smoke' else 2):
            seed = old.request_seed(qid, sample, 'direct' if phase == 'smoke' else phase)
            budget = (256 if phase == 'smoke' else 16384) - len(prefix['prefix_token_ids'])
            sampling = old.sampling_parameters(seed, budget)
            sampling['stop_token_ids'] = list(STOP_IDS)
            requests.append(dict(request_id=f'{phase}:{qid}:{sample}', question_id=qid,
                                 question=selected_row['question'], answer=selected_row['answer'],
                                 phase=phase, input_source=source_key, sample_index=sample, seed=seed,
                                 protocol=VERSION, enable_thinking=False, eos_token_id=tokenizer.eos_token_id,
                                 **prompt, **prefix, prompt_token_ids=ids,
                                 prompt_text=old.decode_ids(tokenizer, ids), sampling=sampling))
    if len({r['seed'] for r in requests}) != len(requests):
        raise ValueError('Seed collision')
    return requests


def execution_signature(requests):
    return [{k: r[k] for k in ('request_id', 'prompt_token_ids', 'sampling', 'eos_token_id')}
            for r in requests]


def invocation():
    result = prior.invocation()
    paths = [Path(__file__), Path(assets.__file__), assets.CONFIG,
             Path(__file__).with_name('run_qwen06_teacher_screen.py')]
    result['runtime_hashes'].update({str(p.resolve()): old.sha256(p) for p in paths})
    result['runtime_sha256'] = old.object_hash(result['runtime_hashes'])
    result['script_sha256'] = old.sha256(__file__)
    return result


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    _, selected, sources = prior.load_source()
    models = {}
    for key in assets.specifications():
        print('Verifying model', key, flush=True)
        models[key] = assets.ensure_asset(key)
    tokens = {k: old.load_tokenizer(m['path']) for k, m in models.items()}
    for key, tok in tokens.items():
        if tok.eos_token_id != models[key]['config']['eos_token_id']:
            raise ValueError(f'Native EOS/config mismatch: {key}')
    views, cells, native = main_views(), {}, {}
    direct = {}
    for view, spec in views.items():
        source, target = spec['input'], spec['model']
        direct[view] = for_target(build_requests(selected, source, 'direct', tokens[source]),
                                  tokens[target], models[target]['config']['vocab_size'])
    for key, tok in tokens.items():
        own = for_target(build_requests(selected, key, 'direct', tok), tok, models[key]['config']['vocab_size'])
        aliases = [v for v in views if views[v]['model'] == key and
                   execution_signature(own) == execution_signature(direct[v])]
        if aliases:
            native[key] = aliases[0] + '_direct'
        else:
            view = key + '_native'
            views[view] = dict(model=key, input=key)
            direct[view] = own
            native[key] = view + '_direct'
    for key in models:
        cells[key + '_smoke'] = dict(model=key, input=key, phase='smoke', depends=[])
    for view, spec in views.items():
        target, source = spec['model'], spec['input']
        cells[view + '_direct'] = dict(**spec, phase='direct', depends=[target + '_smoke'])
        if view in main_views():
            cells[view + '_continuation'] = dict(**spec, phase='continuation',
                                                 depends=[target + '_smoke', source + '_' + source + '_direct'])
    old.write_json(root / 'selected.json', selected)
    audits = {}
    for view, requests in direct.items():
        target, source = views[view]['model'], views[view]['input']
        audits[view] = input_audit(tokens[source], tokens[target],
                                  [i for r in requests for i in r['prompt_token_ids']],
                                  models[target]['config']['vocab_size'])
    manifest = dict(protocol=protocol(), pairs=PAIRS, models=models, cells=cells, native_direct=native,
                    direct_requests=direct, direct_input_audits=audits, source_hashes=sources,
                    selected_sha256=old.sha256(root / 'selected.json'), selection_sha256=old.object_hash(selected),
                    grader_path=str(old.GRADER), grader_sha256=old.historical.HISTORICAL_GRADER_SHA256,
                    versions=prior.runtime_versions(), invocation=invocation(),
                    formal_rollouts=sum(128 if c['phase'] == 'direct' else 64 for c in cells.values() if c['phase'] != 'smoke'),
                    smoke_rollouts=4 * len(models))
    old.load_historical_grader(old.GRADER)
    old.seal_json(root / 'prepare_manifest.json', manifest)
    return dict(cells=len(cells), formal_rollouts=manifest['formal_rollouts'], native_direct=native)


def load_preparation(root):
    root = Path(root)
    m = old.read_sealed(root / 'prepare_manifest.json')
    if m['protocol'] != protocol() or m['versions'] != prior.runtime_versions():
        raise ValueError('Protocol/runtime drift')
    old.verify_hashes(m['invocation']['runtime_hashes'])
    old.verify_hashes(m['source_hashes'])
    old.verify_hashes({str(root / 'selected.json'): m['selected_sha256'], m['grader_path']: m['grader_sha256']})
    selected = old.read_json(root / 'selected.json')
    if old.object_hash(selected) != prior.SOURCE_SELECTION_SHA:
        raise ValueError('Question selection drift')
    return m, selected


def cell_requests(root, key, m, selected, tokens):
    c = m['cells'][key]
    source, target, phase = c['input'], c['model'], c['phase']
    if phase == 'direct':
        rows = m['direct_requests'][key.removesuffix('_direct')]
    else:
        prefix_rows = None
        if phase == 'continuation':
            prefix_rows, _ = read_cell(root, source + '_' + source + '_direct', m, selected, tokens)
        rows = for_target(build_requests(selected, source, phase, tokens[source], prefix_rows),
                          tokens[target], m['models'][target]['config']['vocab_size'])
    return rows


def model_label(key):
    return 'student' if key in STUDENTS else 'teacher'


def read_cell(root, key, m, selected, tokens):
    folder = Path(root) / 'cells' / key
    completion = old.read_sealed(folder / 'completion.json')
    if completion.get('complete') is not True or completion.get('cell') != key:
        raise ValueError('Incomplete/wrong cell')
    expected_files = {'manifest.json', 'requests.json', 'raw.jsonl', 'results.jsonl', 'engine_events.jsonl'}
    if set(completion['files']) != expected_files:
        raise ValueError('Incomplete artifact inventory')
    old.verify_hashes({str(folder / n): h for n, h in completion['files'].items()})
    cm = old.read_sealed(folder / 'manifest.json')
    requests = cell_requests(root, key, m, selected, tokens)
    c = m['cells'][key]
    if (old.read_json(folder / 'requests.json') != requests or cm['cell_spec'] != c or
            cm['prepare_sha256'] != old.sha256(Path(root) / 'prepare_manifest.json') or
            cm['model'] != m['models'][c['model']] or completion['num_rollouts'] != len(requests)):
        raise ValueError('Cell provenance mismatch')
    old.verify_hashes(cm['invocation']['runtime_hashes'])
    rows = list(old.read_jsonl(folder / 'raw.jsonl'))
    old.validate_records(rows, requests, model_label(c['model']), tokens[c['input']],
                         model=m['models'][c['model']]['path'], response_tokenizer=tokens[c['model']])
    prior.validate_stops(rows)
    grade, extract = old.load_historical_grader(m['grader_path'])
    scores = list(old.read_jsonl(folder / 'results.jsonl'))
    if scores != old.score_records(rows, grade, extract):
        raise ValueError('Historical regrading mismatch')
    return rows, scores


def generate(root, key, gpu):
    root = Path(root)
    if gpu not in range(4):
        raise ValueError('Only ml2 GPU0-3 approved')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    with (root / f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m, selected = load_preparation(root)
        c = m['cells'][key]
        target, source = c['model'], c['input']
        model = m['models'][target]
        old.verify_hashes(model['hashes'])
        tokens = {k: old.load_tokenizer(v['path']) for k, v in m['models'].items()}
        requests = cell_requests(root, key, m, selected, tokens)
        audit = input_audit(tokens[source], tokens[target], [i for r in requests for i in r['prompt_token_ids']],
                            model['config']['vocab_size'])
        folder = root / 'cells' / key
        folder.mkdir(parents=True, exist_ok=False)
        old.write_json(folder / 'requests.json', requests)
        old.seal_json(folder / 'manifest.json', dict(cell_spec=c, model=model, gpu=gpu,
            prepare_sha256=old.sha256(root / 'prepare_manifest.json'), input_audit=audit,
            versions=prior.runtime_versions(), invocation=invocation()))
        grade, extract = old.load_historical_grader(m['grader_path'])
        from vllm import LLM, SamplingParams
        llm = LLM(model=model['path'], tokenizer=model['path'], **protocol()['engine'])
        if llm.get_tokenizer().get_vocab() != tokens[target].get_vocab():
            raise ValueError('Engine/native output tokenizer mismatch')
        mapping = {r['request_id']: r for r in requests}
        for r in requests:
            llm.llm_engine.add_request(r['request_id'], {'prompt_token_ids': r['prompt_token_ids']},
                                      SamplingParams(**r['sampling']))
        rows, seen = [], set()
        with (folder / 'raw.jsonl').open('x') as raw, (folder / 'results.jsonl').open('x') as scored, \
                (folder / 'engine_events.jsonl').open('x') as events:
            while llm.llm_engine.has_unfinished_requests():
                outputs = [o for o in llm.llm_engine.step() if o.finished]
                if not outputs:
                    continue
                prior.archive_engine_batch(events, outputs)
                pending = []
                for output in outputs:
                    if output.request_id not in mapping or output.request_id in seen:
                        raise ValueError('Unexpected/duplicate engine request')
                    row = old.engine_record(output, mapping[output.request_id], model_label(target), model['path'], tokens[target])
                    raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    pending.append(row)
                    seen.add(output.request_id)
                raw.flush()
                os.fsync(raw.fileno())
                for row in pending:
                    prior.validate_stops([row])
                    score = old.score_records([row], grade, extract)[0]
                    scored.write(json.dumps(score, ensure_ascii=False, allow_nan=False) + '\n')
                    scored.flush()
                    os.fsync(scored.fileno())
                    rows.append(row)
                    print(json.dumps(dict(cell=key, completed=len(rows), total=len(requests),
                        length=len(row['response_token_ids']), finish=row['finish_reason'])), flush=True)
        old.validate_records(rows, requests, model_label(target), tokens[source], model=model['path'], response_tokenizer=tokens[target])
        old.verify_hashes(model['hashes'])
        load_preparation(root)
        if c['phase'] == 'smoke':
            # A scientific health failure is retained, not silently retried or excluded.
            old.seal_json(folder / 'smoke_health.json', dict(num_rollouts=len(rows),
                generated_think_tags=sum('<think>' in r['response_text'] or '</think>' in r['response_text'] for r in rows),
                length_stops=sum(r['finish_reason'] == 'length' for r in rows),
                scope='256-token native-input probe, not a truncation estimate for 16K formal diagnostics'))
        old.seal_json(folder / 'completion.json', dict(complete=True, cell=key, num_rollouts=len(rows),
            files={n: old.sha256(folder / n) for n in ('manifest.json', 'requests.json', 'raw.jsonl', 'results.jsonl', 'engine_events.jsonl')}))


def ready_cells(cells, statuses, running):
    return [k for k, c in cells.items() if k not in statuses and k not in running and
            all(statuses.get(d) == 'complete' for d in c['depends'])]


def blocked_cells(cells, statuses):
    return [k for k, c in cells.items() if k not in statuses and
            any(statuses.get(d) in ('failed', 'blocked') for d in c['depends'])]


def gate_status(checks):
    failed = {k for k, v in checks.items() if not v}
    return ('passed' if not failed else 'rejected' if failed -
            {'direct_gain', 'direct_ci', 'continuation_gain'} else 'inconclusive')


def evaluate_pair(cells):
    result = {}
    for phase, n in [('direct', 64), ('continuation', 32)]:
        stats = {label: old._cell_statistics(cells[phase][label], n) for label in ('student', 'teacher')}
        qs = sorted(stats['student']['per_question'])
        if set(qs) != set(stats['teacher']['per_question']):
            raise ValueError('Unpaired question coverage')
        result[phase] = dict(models=stats, teacher_minus_student=old.paired_question_bootstrap(
            [stats['student']['per_question'][q] for q in qs], [stats['teacher']['per_question'][q] for q in qs]))
    direct, continuation = [result[p]['teacher_minus_student'] for p in ('direct', 'continuation')]
    if not set(result['continuation']['models']['student']['per_question']).issubset(
            result['direct']['models']['student']['per_question']):
        raise ValueError('Continuation questions must be a subset of direct questions')
    checks = dict(direct_gain=direct['difference'] >= .05, direct_ci=direct['ci95'][0] > 0,
                  continuation_gain=continuation['difference'] >= 0)
    for phase in ('direct', 'continuation'):
        t, s = [result[phase]['models'][label] for label in ('teacher', 'student')]
        checks.update({phase + '_length_stop': t['length_stop_rate'] <= .1,
                       phase + '_length_excess': t['length_stop_rate'] - s['length_stop_rate'] <= .05,
                       phase + '_periodic': t['periodic_rate'] <= .05,
                       phase + '_missing_boxed': t['missing_boxed_rate'] <= .25,
                       phase + '_missing_boxed_excess': t['missing_boxed_rate'] - s['missing_boxed_rate'] <= .05,
                       phase + '_no_generated_thinking': t['think_tag_count'] == s['think_tag_count'] == 0})
    result.update(checks=checks, passed=all(checks.values()), status=gate_status(checks),
                  failures=[k for k, v in checks.items() if not v], training_authorized=False)
    return result


def summarize(root):
    root = Path(root)
    m, selected = load_preparation(root)
    tokens = {k: old.load_tokenizer(v['path']) for k, v in m['models'].items()}
    cache, errors, reports = {}, {}, {}
    for key in m['cells']:
        try:
            cache[key] = read_cell(root, key, m, selected, tokens)[1]
        except (OSError, ValueError, KeyError) as exc:
            errors[key] = str(exc)
    for pair, (teacher, student) in PAIRS.items():
        required = [v + '_' + p for v in (pair, student + '_' + student) for p in ('direct', 'continuation')]
        required += [m['native_direct'][teacher]]
        if any(k not in cache for k in required):
            reports[pair] = dict(status='incomplete', training_authorized=False,
                                  errors={k: errors[k] for k in required if k not in cache})
            continue
        cells = {p: dict(student=cache[student + '_' + student + '_' + p], teacher=cache[pair + '_' + p])
                 for p in ('direct', 'continuation')}
        result = evaluate_pair(cells)
        native = old._cell_statistics(cache[m['native_direct'][teacher]], 64)
        qs = sorted(native['per_question'])
        main = result['direct']['models']['teacher']['per_question']
        result.update(teacher=teacher, student=student, native_direct=native,
                      native_control_cell=m['native_direct'][teacher],
                      native_minus_student_input_teacher=old.paired_question_bootstrap(
                          [main[q] for q in qs], [native['per_question'][q] for q in qs]))
        reports[pair] = result
    result = dict(reports=reports, cell_errors=errors, protocol=protocol(), training_started=False,
                  prepare_sha256=old.sha256(root / 'prepare_manifest.json'),
                  evidence={k: old.sha256(root / 'cells' / k / 'completion.json') for k in cache},
                  invocation=invocation())
    old.seal_json(root / 'pair_summary.json', result)
    with (root / 'pair_summary.csv').open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['pair', 'status', 'phase', 'student_correct_rate', 'teacher_correct_rate',
                         'gain', 'ci95_low', 'ci95_high', 'teacher_length_stop_rate', 'teacher_missing_boxed_rate',
                         'native_direct_correct_rate'])
        for pair, r in reports.items():
            if r['status'] == 'incomplete':
                writer.writerow([pair, 'incomplete'])
                continue
            for phase in ('direct', 'continuation'):
                s, t = [r[phase]['models'][label] for label in ('student', 'teacher')]
                delta = r[phase]['teacher_minus_student']
                writer.writerow([pair, r['status'], phase, s['correct_rate'], t['correct_rate'], delta['difference'],
                    *delta['ci95'], t['length_stop_rate'], t['missing_boxed_rate'], r['native_direct']['correct_rate']])
    return {k: v['status'] for k, v in reports.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'generate', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cell')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.root)
    elif args.action == 'summarize':
        result = summarize(args.root)
    else:
        result = generate(args.root, args.cell, args.gpu)
    print(json.dumps(result, ensure_ascii=False), flush=True)

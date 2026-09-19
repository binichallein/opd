#!/usr/bin/env python3
"""Audit the fixed Qwen completion cohort and grade using the historical grader."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import signal

from diagnose_qwen4_initial_rollouts import save_json, sha256, summarize
from diagnose_token_truncation import analyze_tokens
from audit_llama_prompt_probe import GRADER_SHA, timeout_handler

CELLS = ('historical', 'plain', 'candidate')


def validate_records(rows, requests, cap):
    expected = {r['request_id']: r for r in requests}
    if (len(rows) != len(requests) or len(expected) != len(requests)
            or len({r['request_id'] for r in rows}) != len(rows)):
        raise ValueError('Incomplete or duplicate outputs')
    for row in rows:
        reference = expected.get(row['request_id'])
        if reference is None or any(row[k] != reference[k] for k in ('index', 'seed', 'prompt_token_ids')):
            raise ValueError('Changed input or seed')
        ids, logps = row['response_token_ids'], row['sampled_logprobs']
        if (not len(ids) == len(logps) == row['response_length'] <= cap
                or not all(math.isfinite(v) for v in logps) or 151643 in ids[:-1]
                or row['finish_reason'] not in ('length', 'stop')
                or (row['finish_reason'] == 'length' and len(ids) != cap)):
            raise ValueError('Invalid lengths, logprobs or EOS termination')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--grader', type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.grader) != GRADER_SHA:
        raise ValueError('Wrong historical grader')
    spec = importlib.util.spec_from_file_location('pinned_grader', args.grader)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    signal.signal(signal.SIGALRM, timeout_handler)
    if not grader.grade_answer_verl(r'\boxed{2}', '2') or grader.grade_answer_verl(r'\boxed{3}', '2'):
        raise ValueError('Grader self-test failed')
    from transformers import AutoTokenizer
    report = {'scope': '16 fixed training questions x2; not benchmark scores', 'models': {},
              'grader_sha256': GRADER_SHA, 'audit_script_sha256': sha256(__file__), 'passed': True}
    paired = {}
    for label in ('student', 'teacher'):
        root = args.root/label
        manifest = json.loads((root/'manifest.json').read_text())
        summaries = json.loads((root/'summary.json').read_text())
        for path, digest in manifest['code'].items():
            if sha256(path) != digest:
                raise ValueError('Changed frozen inference code')
        if len(manifest['sources']) != 16:
            raise ValueError('Unexpected question count')
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
        cells = {}
        for cell in CELLS:
            folder = root/cell
            inputs = json.loads((folder/'inputs.json').read_text())
            config, requests = inputs['config'], inputs['requests']
            rows = [json.loads(line) for line in (folder/'raw.jsonl').read_text().splitlines()]
            validate_records(rows, requests, 16384)
            if len(rows) != 32 or {(r['index'], r['seed']) for r in rows} != {
                    (s['index'], seed) for s in manifest['sources'] for seed in (21, 22)}:
                raise ValueError('Wrong fixed cohort coverage')
            if sha256(folder/'raw.jsonl') != summaries[cell]['raw_sha256']:
                raise ValueError('Raw output hash mismatch')
            signature = (config, sorted((r['index'], r['seed'], r['prompt_token_ids']) for r in requests))
            if label == 'student':
                paired[cell] = signature
            elif paired[cell] != signature:
                raise ValueError('Unmatched teacher/student conditions')
            graded = []
            for row in rows:
                for key, value in analyze_tokens(row['response_token_ids'], 16384).items():
                    if row[key] != value:
                        raise ValueError('Repetition/length metric mismatch')
                raw = tokenizer.decode(row['response_token_ids'], skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
                if raw != row['response_text']:
                    raise ValueError('Raw decode mismatch')
                text = tokenizer.decode(row['response_token_ids'], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                error, correct = None, None
                try:
                    signal.setitimer(signal.ITIMER_REAL, 15)
                    correct = bool(grader.grade_answer_verl(text, str(row['answer'])))
                except Exception as exc:
                    error = type(exc).__name__+': '+str(exc)
                    report['passed'] = False
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                graded.append({'request_id': row['request_id'], 'index': row['index'], 'seed': row['seed'],
                    'answer': row['answer'], 'correct': correct, 'grade_error': error,
                    'format_error': '\\boxed' not in text, 'length_stop': row['finish_reason'] == 'length'})
            for key, value in summarize(rows).items():
                if summaries[cell][key] != value:
                    raise ValueError('Summary mismatch')
            cells[cell] = {**summaries[cell], 'diagnostic_correct': sum(r['correct'] is True for r in graded),
                          'grader_errors': sum(r['grade_error'] is not None for r in graded),
                          'historical_format_errors': sum(r['format_error'] for r in graded),
                          'question_seed_results': sorted(graded, key=lambda r: (r['index'], r['seed']))}
        report['models'][label] = {'model': manifest['model'], 'cells': cells}
    save_json(args.root/'audit_and_grades.json', report)
    print(json.dumps({label: {cell: {k: value[k] for k in ('diagnostic_correct', 'length_stops',
        'periodic_tails', 'historical_format_errors', 'grader_errors')} for cell, value in model['cells'].items()}
        for label, model in report['models'].items()}), flush=True)
    if not report['passed']:
        raise RuntimeError('Grader errors retained; not scored as incorrect')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Validate the finished fixed-cohort probe and score with the pinned grader."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import signal

from diagnose_llama_prompt import LLAMA_STOP_IDS, response_stats
from diagnose_qwen4_initial_rollouts import save_json, sha256, summarize

GRADER_SHA = '04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f'
CELLS = ('historical', 'no_think', 'completion', 'eval')


def validate_records(rows, requests, cap):
    expected = {r['request_id']: r for r in requests}
    if (len(rows) != len(requests) or len(expected) != len(requests)
            or len({r['request_id'] for r in rows}) != len(rows)):
        raise ValueError('Incomplete or duplicated requests')
    for row in rows:
        reference = expected.get(row['request_id'])
        if reference is None or any(row[k] != reference[k] for k in ('index', 'seed', 'prompt_token_ids')):
            raise ValueError('Changed input or seed')
        ids, logps = row['response_token_ids'], row['sampled_logprobs']
        if not (len(ids) == len(logps) == row['response_length'] <= cap):
            raise ValueError('Inconsistent lengths')
        if not all(math.isfinite(p) for p in logps):
            raise ValueError('Non-finite log probability')
        if any(t in LLAMA_STOP_IDS for t in ids[:-1]):
            raise ValueError('Generated past a native stop token')
        if row['finish_reason'] == 'length' and len(ids) != cap:
            raise ValueError('Inconsistent length stop')
        if row['finish_reason'] not in ('stop', 'length'):
            raise ValueError('Unexpected termination')


def timeout_handler(signum, frame):
    raise TimeoutError('Diagnostic grader exceeded 15 seconds')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--grader', type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.grader) != GRADER_SHA:
        raise ValueError('Wrong historical grader')
    spec = importlib.util.spec_from_file_location('pinned_diagnostic_grader', args.grader)
    grader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grader)
    signal.signal(signal.SIGALRM, timeout_handler)
    from transformers import AutoTokenizer

    report = {'scope': '16 fixed training questions x 2 samples; NOT benchmark evaluation',
              'grader_sha256': GRADER_SHA, 'audit_script_sha256': sha256(__file__),
              'models': {}, 'passed': True}
    paired = {}
    for label in ('student', 'teacher'):
        root = args.root / label
        manifest = json.loads((root / 'manifest.json').read_text())
        recorded = json.loads((root / 'summary.json').read_text())
        for path, digest in manifest['code'].items():
            if sha256(path) != digest:
                raise ValueError('Changed inference code')
        if len(manifest['sources']) != 16:
            raise ValueError('Unexpected diagnostic question count')
        tokenizer = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True)
        model_report = {'model': manifest['model'], 'revision': manifest['modelscope_revision'], 'cells': {}}
        for cell in CELLS:
            folder = root / cell
            inputs = json.loads((folder / 'inputs.json').read_text())
            config, requests = inputs['config'], inputs['requests']
            rows = [json.loads(line) for line in (folder / 'raw.jsonl').read_text().splitlines()]
            validate_records(rows, requests, config['max_tokens'])
            if len(rows) != 32 or {r['seed'] for r in rows} != {21, 22}:
                raise ValueError('Unexpected diagnostic sample count/seeds')
            expected = {(s['index'], seed) for s in manifest['sources'] for seed in (21, 22)}
            if {(r['index'], r['seed']) for r in rows} != expected:
                raise ValueError('Wrong diagnostic cohort')
            if sha256(folder / 'raw.jsonl') != recorded[cell]['raw_sha256']:
                raise ValueError('Raw SHA mismatch')
            signature = (config, sorted((r['index'], r['seed'], r['prompt_token_ids']) for r in requests))
            if label == 'student':
                paired[cell] = signature
            elif signature != paired[cell]:
                raise ValueError('Unmatched teacher/student prompt or sampling')
            graded = []
            for row in rows:
                for key, value in response_stats(row['response_token_ids'], row['response_text'], 16384).items():
                    if row[key] != value:
                        raise ValueError('Diagnostic metric mismatch')
                text = tokenizer.decode(row['response_token_ids'], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                error, correct = None, None
                try:
                    signal.setitimer(signal.ITIMER_REAL, 15)
                    correct = bool(grader.grade_answer_verl(text, str(row['answer'])))
                except Exception as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
                    report['passed'] = False
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                graded.append({'request_id': row['request_id'], 'index': row['index'], 'seed': row['seed'],
                               'answer': row['answer'], 'correct': correct, 'grade_error': error,
                               'historical_format_error': '\\boxed' not in text,
                               'length_stop': row['finish_reason'] == 'length'})
            for key, value in summarize(rows).items():
                if recorded[cell][key] != value:
                    raise ValueError('Summary count mismatch')
            model_report['cells'][cell] = {**recorded[cell],
                'diagnostic_correct': sum(r['correct'] is True for r in graded),
                'grader_errors': sum(r['grade_error'] is not None for r in graded),
                'historical_format_errors': sum(r['historical_format_error'] for r in graded),
                'question_seed_results': sorted(graded, key=lambda r: (r['index'], r['seed']))}
        report['models'][label] = model_report
    save_json(args.root / 'audit_and_grades.json', report)
    print(json.dumps({label: {cell: {k: value[k] for k in ('length_stops', 'periodic_tails',
          'high_4gram_repeat_tails', 'diagnostic_correct', 'historical_format_errors', 'grader_errors')}
          for cell, value in model['cells'].items()} for label, model in report['models'].items()}), flush=True)
    if not report['passed']:
        raise RuntimeError('Audit retained grader errors; do not treat them as incorrect answers')


if __name__ == '__main__':
    main()

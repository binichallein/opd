#!/usr/bin/env python3
"""Check built bilingual manuscripts without treating formatting as scientific validation."""

import argparse
import json
import math
from pathlib import Path
import re
import subprocess


def validate_evaluation(report):
    tasks = ('math500', 'aime24', 'aime25', 'amc23')
    expected = {(arm, step, task) for arm in ('token_opd', 'block3_mean')
                for step in (50, 100, 150, 200) for task in tasks}
    rows = report.get('rows', [])
    observed = {(r['variant'], r['step'], r['task']) for r in rows}
    if report.get('complete') is not True or report.get('missing_checkpoints'):
        raise ValueError('Current evaluation is incomplete; cannot finalize the paper')
    if observed != expected or len(rows) != len(expected):
        raise ValueError('Expected exactly two methods by four checkpoints by four benchmarks')
    for row in rows:
        if row.get('status') != 'accepted':
            raise ValueError('A checkpoint result has not passed acceptance')
        for metric in ('avg_at_8', 'pass_at_8', 'format_error_rate', 'engine_truncation_ratio'):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid final metric: {metric}')
    if {r['task'] for r in report.get('bootstrap', []) if r['step'] == 200} != set(tasks):
        raise ValueError('Missing primary-endpoint paired-question intervals')


def validate_instruct_evaluation(report):
    tasks = {'math500': 500, 'aime24': 30, 'aime25': 30, 'amc23': 83}
    steps = (50, 100, 150, 200)
    names = {'student_base'} | {f'{arm}_step{s}' for arm in ('token_opd', 'block3_mean') for s in steps}
    if (report.get('complete') is not True or report.get('student') != 'Qwen/Qwen3-1.7B'
            or report.get('teacher') != 'Qwen/Qwen3-8B'
            or report.get('protocol') != 'qwen3_native_chat_no_thinking_boxed_v1'
            or report.get('training_seed') != 21 or set(report.get('per_model', {})) != names):
        raise ValueError('Incomplete or wrong Instruct comparison')
    for result in report['per_model'].values():
        if set(result) != set(tasks):
            raise ValueError('Missing Instruct benchmark')
        for task, n in tasks.items():
            row = result[task]
            if row.get('num_examples') != n or row.get('num_rollouts') != 8*n:
                raise ValueError('Wrong Instruct question/response count')
            for metric in ('avg_at_8', 'pass_at_8', 'format_error_rate', 'engine_truncation_ratio'):
                value = row.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid Instruct metric')
    if set(report.get('bootstrap', {})) != {str(s) for s in steps}:
        raise ValueError('Missing Instruct checkpoint intervals')
    for step in steps:
        cis = report['bootstrap'][str(step)]
        if set(cis) != set(tasks):
            raise ValueError('Missing Instruct benchmark intervals')
        for task, values in cis.items():
            for metric in ('avg_at_8', 'pass_at_8'):
                ci = values.get(metric, {})
                nums = [ci.get(k) for k in ('delta_pp', 'lower_95_pp', 'upper_95_pp')]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums):
                    raise ValueError('Nonfinite or missing Instruct interval')
                delta = 100*(report['per_model'][f'block3_mean_step{step}'][task][metric]
                             - report['per_model'][f'token_opd_step{step}'][task][metric])
                if not math.isclose(nums[0], delta, abs_tol=1e-10) or nums[1] > nums[2]:
                    raise ValueError('Instruct interval disagrees with accepted scores')


def inspect_paper(paper, *, require_evaluation=True):
    reports = {}
    for lang in ('en', 'zh'):
        folder = paper / 'build' / lang
        log = (folder / f'main_{lang}.log').read_text(errors='replace')
        aux = (folder / f'main_{lang}.aux').read_text(errors='replace')
        text = (folder / f'main_{lang}.txt').read_text()
        pdf = folder / f'main_{lang}.pdf'
        label = re.search(r'\\newlabel\{mainend\}\{\{[^}]*\}\{(\d+)\}', aux)
        if not label:
            raise ValueError(f'{lang}: main-text boundary label missing')
        main_pages = int(label.group(1))
        errors = []
        if lang == 'en' and main_pages > 9:
            errors.append('Initial English submission exceeds nine main-text pages')
        if re.search(r'(Citation|Reference) .+ undefined|There were undefined references', log):
            errors.append('Unresolved bibliography or cross-reference')
        if re.search(r'(?i)\b(TODO|TBD|FIXME|placeholder)\b|待填|待补', text):
            errors.append('Unfinished manuscript marker')
        if require_evaluation and re.search(r'\bPARTIAL\b', text):
            errors.append('Stale partial-evaluation figure or table remains in the final PDF')
        for forbidden in ('/home/tyf', '/limx_embap', 'binichallein', '2928210292', 'Yaleon', 'yaofeng tu'):
            if forbidden.lower() in text.lower():
                errors.append(f'Potential anonymity leak: {forbidden}')
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        author = re.search(r'^Author:[ \t]*(.*)$', info, re.M)
        if author and author.group(1).strip() not in ('', 'Anonymous authors'):
            errors.append('Non-anonymous PDF author metadata')
        if pdf.stat().st_size >= 50_000_000:
            errors.append('PDF exceeds 50 MB')
        if not re.search(r'AI\s+Use\s+Statement' if lang == 'en' else r'AI\s*使用声明', text):
            errors.append('AI disclosure heading missing')
        reports[lang] = {'main_text_pages': main_pages, 'pdf_bytes': pdf.stat().st_size,
                         'errors': errors,
                         'overfull_boxes': len(re.findall(r'Overfull \\[hv]box', log))}
    evidence_errors = []
    if require_evaluation:
        try:
            assets = (paper / 'asset_paths.tex').read_text()
            match = re.search(r'\\newcommand\{\\qwenresults\}\{([^}]+)\}', assets)
            if not match:
                raise ValueError('Missing current-result directory in manuscript inputs')
            directory = (paper / match.group(1)).resolve()
            if not directory.is_relative_to((paper / 'generated').resolve()):
                raise ValueError('Result directory must be inside the manuscript generated directory')
            validate_evaluation(json.loads((directory / 'qwen4_summary.json').read_text()))
            if (paper / 'instruct_appendix_en.tex').exists():
                match = re.search(r'\\newcommand\{\\qweninstructresults\}\{([^}]+)\}', assets)
                if not match:
                    raise ValueError('Missing Instruct result declaration')
                directory = (paper / match.group(1)).resolve()
                if not directory.is_relative_to((paper / 'generated').resolve()):
                    raise ValueError('Instruct evidence must be inside generated')
                validate_instruct_evaluation(json.loads((directory / 'qwen17_summary.json').read_text()))
        except (OSError, ValueError, KeyError) as exc:
            evidence_errors.append(str(exc))
    result = {'passed': not evidence_errors and not any(row['errors'] for row in reports.values()),
              'draft_only': not require_evaluation, 'evidence_errors': evidence_errors,
              'papers': reports, 'scope': 'Mechanical and evidence-completeness checks; scientific and visual review still required.'}
    (paper / 'build' / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise ValueError(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper', type=Path)
    parser.add_argument('--draft', action='store_true', help='Layout only; do not assert evaluation completion')
    args = parser.parse_args()
    print(json.dumps(inspect_paper(args.paper, require_evaluation=not args.draft), indent=2))

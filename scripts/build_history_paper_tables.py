#!/usr/bin/env python3
"""Render all archived experiment views separately, excluding identifying paths."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


SERIES = {
    'early_weights50': 'Early weighting pilots',
    'early_outcome200': 'Outcome weighting and long-run pilots',
    'early_anchor100': 'Reference-anchor pilots',
    'early_blocksum50': 'Naive block-sum pilot',
    'early_blockadv50': 'Block sum, mean and mixed pilot',
    'qwen17_sweep': 'Qwen 1.7B block-size sweep',
    'block10_collapse': 'Block10 two-host diagnostic repetitions',
    'qwen17_ml2': 'Qwen 1.7B same-host paired study',
    'qwen17_reeval': 'Qwen 1.7B checkpoint re-evaluation',
    'deepseek_justrl': 'DeepSeek/JustRL paired study',
    'window_random_sliding': 'Random3 and Sliding3 paired study',
    'qwen06_old_prompt': 'Qwen 0.6B old-prompt interrupted attempt',
    'qwen06_nonthinking': 'Qwen 0.6B non-thinking paired study',
    'llama_native_stopped': 'Llama 3.2 original-prompt interrupted attempt',
    'llama_historical': 'Llama 3.2 historical-prompt paired study',
    'qwen4_chatml_stopped': 'Qwen 4B ChatML interrupted attempt',
    'qwen4_completion': 'Qwen 4B completion study',
}
VIEWS = {
    'historical_external': 'Historical external grader: Avg@8 / Pass@8',
    'historical_external_audited': 'Audited historical external grader: Avg@8 / Pass@8',
    'full_n1_online_simple': 'Full n=1: original simple grader, accuracy',
    'full_n1_primary_regrade': 'Full n=1: post-hoc primary grader, accuracy',
    'builtin_verl': 'Built-in grader: Avg@8 / Pass@8 (separate view)',
    'pilot100_original': '100-question pilot: mean score / Pass@k, k unrecorded',
    'full_n1_primary_reported_rounded': 'Full n=1: documented rounded accuracy only',
    'legacy_external_no_explicit_eval_seeds': 'Legacy external grader: Avg@8 / Pass@8, evaluation seeds unspecified',
    'planned_verl_incomplete': 'Incomplete planned evaluation: no full benchmark claim',
}
TASKS = ('gsm8k', 'math500', 'aime24', 'aime25', 'amc23')
NAMES = {'gsm8k': 'GSM8K', 'math500': 'MATH500', 'aime24': 'AIME24', 'aime25': 'AIME25', 'amc23': 'AMC23'}
ARM_NAMES = {'block10_mean:ml2-A100': 'Block10: A100 host',
             'block10_mean:train-A800': 'Block10: A800 host'}


def escape(text):
    replacements = {'\\': r'\textbackslash{}', '_': r'\_\allowbreak{}', '%': r'\%', '&': r'\&',
                    '#': r'\#', '{': r'\{', '}': r'\}', '$': r'\$', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(text))


def group_evaluations(evaluations):
    rows = {}
    for e in evaluations:
        key = tuple(e.get(k) for k in ('series', 'view', 'arm', 'checkpoint_step', 'training_seed', 'n'))
        if key not in rows:
            rows[key] = {k: e.get(k) for k in ('series', 'view', 'arm', 'checkpoint_step', 'training_seed', 'n')}
            rows[key].update(per_task={}, evidence_ids=[], statuses=[])
        row = rows[key]
        row['evidence_ids'].append(e['id'])
        row['statuses'].append(e['status'])
        for task, record in e['per_task'].items():
            if task in row['per_task']:
                raise ValueError(f'Duplicate task within an experimental view: {key}, {task}')
            row['per_task'][task] = record['metrics']
    return list(rows.values())


def number(value):
    return '--' if value is None else f'{100 * value:.2f}'


def metric_cell(metrics):
    if not metrics or all(v is None for v in metrics.values()):
        return '--'
    if 'accuracy' in metrics:
        return number(metrics['accuracy'])
    if 'avg_at_8' in metrics:
        return f"{number(metrics['avg_at_8'])} / {number(metrics.get('pass_at_8'))}"
    if 'mean_score' in metrics:
        return f"{number(metrics['mean_score'])} / {number(metrics.get('pass_at_k'))}"
    raise ValueError(f'Unknown archived metric definition: {metrics}')


def archive_tex(rows, series, lang):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['series']][row['view']].append(row)
    output = []
    for index, s in enumerate(series, 1):
        sid = s['id']
        title = SERIES[sid] if lang == 'en' else s['title']
        output += [f'\\subsection{{H{index:02d}: {escape(title)}}}', f'\\label{{history:{sid}}}']
        seeds = ', '.join(str(n) for n in (s.get('training_seeds') or [])) or ('unrecorded' if lang == 'en' else '未记录')
        steps = str(s.get('training_steps') or '--')
        label = f'Training seeds: {seeds}; recorded training steps: {steps}.' if lang == 'en' else f'训练seed：{seeds}；记录训练步数：{steps}。'
        output.append(escape(label))
        if sid == 'qwen4_completion':
            output.append(r'See Appendix~\ref{app:current} for the final current-pair results.' if lang == 'en'
                          else r'该师生对的最终结果见附录\ref{app:current}。')
            continue
        if not grouped[sid]:
            output.append('No completed benchmark result is recorded for this attempt.' if lang == 'en' else '此尝试无已完成benchmark结果记录。')
        for view, entries in grouped[sid].items():
            tasks = [t for t in TASKS if any(t in row['per_task'] for row in entries)]
            if view == 'builtin_verl' and 'amc23' in tasks:
                output.append(r'\emph{Historical built-in AMC23 grading has a compatibility caveat; zeros are not independently validated capability measurements.}' if lang == 'en'
                              else r'\emph{历史内置AMC23判分存在兼容性问题；零分不等于独立验证的能力测量。}')
            output += [r'\paragraph{' + escape(VIEWS[view]) + '}', r'{\footnotesize\setlength{\tabcolsep}{3.5pt}',
                       r'\begin{longtable}{@{}p{1.4in}rr' + 'r' * len(tasks) + '@{}}', r'\toprule',
                       'Method & Step & Seed & ' + ' & '.join(NAMES[t] for t in tasks) + r' \\', r'\midrule\endhead']
            for row in sorted(entries, key=lambda e: (e['arm'], e['checkpoint_step'] or 0, str(e['training_seed']))):
                output.append(' & '.join([escape(ARM_NAMES.get(row['arm'], row['arm'])), str(row['checkpoint_step']),
                              str(row['training_seed']) if row['training_seed'] is not None else '--'] +
                              [metric_cell(row['per_task'].get(t)) for t in tasks]) + r' \\')
            output += [r'\bottomrule\end{longtable}', '}']
    return '\n'.join(output) + '\n'


def endpoint_table(rows):
    pairs = [('qwen17_ml2', 'historical_external_audited', 'Qwen 1.7B'),
             ('deepseek_justrl', 'historical_external', 'DeepSeek 1.5B'),
             ('qwen06_nonthinking', 'historical_external', 'Qwen 0.6B'),
             ('llama_historical', 'historical_external', 'Llama 1B')]
    lines = [r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
             r'Student & Method & MATH500 & AIME24 & AIME25 & AMC23 \\', r'\midrule']
    for series, view, label in pairs:
        for arm, name in [('token_opd', 'Token'), ('block3_mean', 'Block3')]:
            matches = [r for r in rows if r['series'] == series and r['view'] == view
                       and r['checkpoint_step'] == 200 and r['arm'] == arm]
            if len(matches) != 1:
                raise ValueError(f'Expected one endpoint: {series} {view} {arm}; got {len(matches)}')
            lines.append(' & '.join([label, name] + [number(matches[0]['per_task'][t]['avg_at_8'])
                                                   for t in TASKS[1:]]) + r' \\')
        lines.append(r'\addlinespace')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}', ''])


def probe_rows(evidence):
    rows = []
    for study in evidence:
        if study['id'] in ('llama_prompt_16', 'qwen4_completion_prompt_16'):
            label = 'Llama prompt' if study['id'] == 'llama_prompt_16' else 'Qwen4 prompt'
            for role, model in study['models'].items():
                for cell, data in model['cells'].items():
                    rows.append({'study': label, 'model': role, 'condition': cell,
                                 'n': data['responses'], 'length_stops': data.get('length_stops'),
                                 'periodic_tails': data.get('periodic_tails'),
                                 'missing_box': data.get('historical_format_errors'),
                                 'correct': data.get('diagnostic_correct')})
        elif study['id'] == 'qwen06_qwen17_truncation_192':
            for group in ('primary32', 'prompt_ablation_first16'):
                for cell, data in study['data'][group].items():
                    rows.append({'study': 'Qwen 0.6/1.7 ' + group, 'model': cell,
                                 'condition': '--', 'n': data['n'],
                                 'length_stops': data.get('length_stop_count'),
                                 'periodic_tails': data.get('periodic_tail_count'),
                                 'missing_box': None, 'correct': data.get('correct_count')})
    for row in rows:
        for key in ('length_stops', 'periodic_tails', 'missing_box', 'correct'):
            value = row[key]
            if value is not None and (not isinstance(value, int) or not 0 <= value <= row['n']):
                raise ValueError(f'Invalid diagnostic count: {row}')
    return rows


def probe_tex(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['study']].append(row)
    output = []
    for study, records in groups.items():
        label = 'Qwen3-4B prompt' if study == 'Qwen4 prompt' else study
        output += [r'\paragraph{' + escape(label) + '}', r'{\small\setlength{\tabcolsep}{4pt}',
                   r'\begin{longtable}{@{}llrrrrr@{}}', r'\toprule',
                   r'Model & Condition & $n$ & Length & Periodic & No box & Correct \\', r'\midrule\endhead']
        for row in records:
            output.append(' & '.join([escape(row['model']), escape(row['condition'])] +
                          ['--' if row[k] is None else str(row[k]) for k in
                           ('n', 'length_stops', 'periodic_tails', 'missing_box', 'correct')]) + r' \\')
        output += [r'\bottomrule\end{longtable}', '}']
    return '\n'.join(output) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=Path(__file__).resolve().parents[1] / 'paper/iclr2027')
    args = parser.parse_args()
    source = args.paper / 'internal/historical_evidence.json'
    evidence = json.loads(source.read_text())
    rows = group_evaluations(evidence['evaluations'])
    probes = probe_rows(evidence['diagnostic_evidence'])
    collapse_source = args.paper.parents[1] / 'results/2026-07-11-block10-collapse-dual-host.json'
    collapse = json.loads(collapse_source.read_text())
    generated = args.paper / 'generated'
    generated.mkdir(exist_ok=True)
    data = args.paper / 'data'
    data.mkdir(exist_ok=True)
    for lang in ('en', 'zh'):
        (generated / f'history_all_{lang}.tex').write_text(archive_tex(rows, evidence['series'], lang))
    (generated / 'history_endpoints.tex').write_text(endpoint_table(rows))
    (generated / 'history_probes.tex').write_text(probe_tex(probes))
    collapse_lines = [r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
                      r'Hardware & $H_S$ & $H_T$ & Overlap & $M_S$ & $M_T$ & Max. grad \\', r'\midrule']
    collapse_public = []
    for host, record in collapse['hosts'].items():
        d = record['step_200_diagnostics']
        row = {'hardware': host.split('-')[-1], 'step_200': d,
               'max_preclip_grad_norm': record['maximum_preclip_grad_norm']}
        collapse_public.append(row)
        values = [d[k] for k in ('student_entropy', 'teacher_entropy', 'top16_overlap_ratio',
                                'student_overlap_mass', 'teacher_overlap_mass')]
        values.append(record['maximum_preclip_grad_norm']['value'])
        collapse_lines.append(' & '.join([row['hardware']] + [f'{v:.3f}' for v in values]) + r' \\')
    (generated / 'history_block10_diagnostics.tex').write_text('\n'.join(collapse_lines + [r'\bottomrule\end{tabular}', '']))
    public_rows = [{**row, 'arm': ARM_NAMES.get(row['arm'], row['arm'])} for row in rows
                   if row['series'] != 'qwen4_completion']
    public = {'schema_version': 1, 'source_inventory_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'rows': public_rows, 'probes': probes, 'units': 'benchmark fractions; tables convert to percentages; probe counts are integers',
              'block10_diagnostics': collapse_public,
              'block10_diagnostic_source_sha256': hashlib.sha256(collapse_source.read_bytes()).hexdigest(),
              'current_pair': 'Stored separately in the qwen4 artifacts; not the earlier inventory snapshot.',
              'note': 'Views and series are kept separate; missing values are never imputed as zero.'}
    (data / 'history_public.json').write_text(json.dumps(public, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'input_evaluations': len(evidence['evaluations']), 'grouped_rows': len(rows),
                      'series': len(evidence['series']), 'source_sha256': public['source_inventory_sha256']}))


if __name__ == '__main__':
    main()

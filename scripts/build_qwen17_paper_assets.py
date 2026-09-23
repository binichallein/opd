#!/usr/bin/env python3
"""Offline, hash-checked bilingual manuscript assets for the completed Instruct pair.

Reads existing graded responses without inference, regrading, or remote writes.
Reuses the paired-question bootstrap and strict graded-record parser of the 4B
paper pipeline. Output directories must be new; accepted exports are immutable.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import archive_qwen17_results as archive
import build_qwen4_paper_assets as shared

TASKS = shared.TASKS
STEPS = (50, 100, 150, 200)
ARMS = ('token_opd', 'block3_mean')
PROTOCOL = 'qwen3_native_chat_no_thinking_boxed_v1'
COMMIT = 'be736b5fac4f26ff59f4e2c21abafc456c2503f6'
MODELS = ['student_base'] + [f'{v}_step{s}' for s in STEPS for v in ARMS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report):
    require(report.get('complete') is True and report.get('pending_models') == [],
            'Only a complete accepted pair can enter the paper')
    require(set(report.get('per_model', {})) == set(MODELS), 'All nine model views required')
    for arm in ARMS:
        card = report['training'][arm]['run_card']
        expected = dict(seed=21, data_seed=21, total_training_steps=200,
                        train_batch_size=4, rollout_group_size=8, learning_rate=2e-6,
                        opd_prompt_protocol=PROTOCOL, source_commit=COMMIT,
                        diagnostic_save_steps='50,100,150,200',
                        student_model_revision='4855588ea1a12789f2e965e5f52a9e4a24c94b2a',
                        teacher_model_revision='26028140be3ee69b82b1d1450179ab71bb1121b9')
        require(all(card.get(k) == v for k, v in expected.items()), 'Wrong training contract')
        require(Path(card['student_model']).name == 'Qwen3-1.7B'
                and Path(card['teacher_model']).name == 'Qwen3-8B', 'Wrong Instruct pair')
    for result in report['per_model'].values():
        require(set(result) == set(TASKS), 'Missing benchmark')
        for task, count in TASKS.items():
            row = result[task]
            require(row['num_examples'] == count and row['num_rollouts'] == 8 * count,
                    'Wrong benchmark sample count')
            for key in shared.METRICS:
                value = row[key]
                require(isinstance(value, (int, float)) and math.isfinite(value)
                        and 0 <= value <= 1, 'Invalid final metric')


def verify_question_scores(rows, expected, task):
    require(len(rows) == TASKS[task] and len({r['id'] for r in rows}) == len(rows),
            'Incomplete or duplicate question identities')
    values = (sum(r['correct_count'] for r in rows) / (8 * len(rows)),
              sum(r['correct_count'] > 0 for r in rows) / len(rows),
              sum(r['missing_box_count'] for r in rows) / (8 * len(rows)),
              sum(r['length_stop_count'] for r in rows) / (8 * len(rows)))
    require(all(math.isclose(v, expected[k], rel_tol=0, abs_tol=1e-12)
                for k, v in zip(shared.METRICS, values)), 'Question/summary disagreement')


def all_checkpoint_table(report, language, metric):
    names = {'student_base': 'Original student' if language == 'en' else '原始学生'}
    names.update({f'{v}_step{s}': f'{"Token" if v == "token_opd" else "Block3"} {s}'
                  for s in STEPS for v in ARMS})
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model / Step & MATH500 & AIME24 & AIME25 & AMC23 \\', r'\midrule']
    for name in MODELS:
        scores = [f"{100 * report['per_model'][name][task][metric]:.2f}" for task in TASKS]
        lines.append(' & '.join([names[name], *scores]) + r' \\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def endpoint_table(report, lang):
    caption = (r'Official Qwen3-8B to Qwen3-1.7B Instruct: Step200 scores (\%) and '
               'Block3 minus Token differences (percentage points).'
               if lang == 'en' else
               r'官方Qwen3-8B到Qwen3-1.7B指令版：Step200分数（\%）及Block3减Token差值（百分点）。')
    note = ('Original denotes the untrained-for-this-experiment Instruct student, not a Base model. '
            r'Brackets are pointwise 95\% paired-question bootstrap intervals (10,000 resamples); '
            'one training seed, no correction for checkpoint or task selection.' if lang == 'en' else
            r'Original为本实验训练前的指令版学生，不是Base模型。方括号为10,000次配对题目重采样的逐点95\%区间；'
            '仅一个训练seed，未校正checkpoint或任务选择。')
    lines = [r'\begin{table}[t]', r'\centering\small', '\\caption{' + caption + '}',
             r'\label{tab:qwen17endpoint}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Benchmark & Original & Token & Block3 & $\Delta$ [95\% CI] \\']
    for metric, label in [('avg_at_8', 'Avg@8'), ('pass_at_8', 'Pass@8')]:
        lines += [r'\midrule', r'\multicolumn{5}{c}{' + label + r'} \\']
        for task in TASKS:
            scores = [report['per_model'][name][task][metric] * 100
                      for name in ('student_base', 'token_opd_step200', 'block3_mean_step200')]
            ci = report['bootstrap']['200'][task][metric]
            delta = f"{ci['delta_pp']:+.2f} [{ci['lower_95_pp']:.2f}, {ci['upper_95_pp']:.2f}]"
            lines.append(' & '.join([shared.TASK_LABELS[task], *[f'{v:.2f}' for v in scores], delta]) + r' \\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}',
                      r'\par\smallskip\footnotesize ' + note, r'\end{table}', ''])


def behavior_table(report):
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Benchmark & Token missing box & Block3 missing box & Token length & Block3 length \\',
             r'\midrule']
    for task in TASKS:
        values = [100 * report['per_model'][f'{arm}_step200'][task][metric]
                  for metric in ('format_error_rate', 'engine_truncation_ratio') for arm in ARMS]
        lines.append(' & '.join([shared.TASK_LABELS[task], *[f'{v:.3f}' for v in values]]) + r' \\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', ''])


def make_figures(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'pdf.fonttype': 42})
    for metric, label in [('avg_at_8', 'Avg@8'), ('pass_at_8', 'Pass@8')]:
        fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6), layout='constrained')
        for ax, task in zip(axes.flat, TASKS):
            ax.axhline(100 * report['per_model']['student_base'][task][metric],
                       color='#666666', linestyle='--', linewidth=1, label='Original student')
            for arm in ARMS:
                ys = [100 * report['per_model'][f'{arm}_step{s}'][task][metric] for s in STEPS]
                ax.plot(STEPS, ys, marker='o', markersize=3, color=shared.COLORS[arm],
                        label=shared.LABELS[arm])
            ax.set(title=shared.TASK_LABELS[task], xlabel='Training step', ylabel=label + ' (%)')
            ax.set_xticks(STEPS)
            ax.grid(axis='y', alpha=.22)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=3, frameon=False)
        fig.savefig(directory / f'qwen17_{metric}.pdf')
        fig.savefig(directory / f'qwen17_{metric}.png', dpi=180)
        plt.close(fig)


def build(snapshot, source, tables, figures):
    report = json.loads(source.read_text())
    validate_report(report)
    backup = json.loads((snapshot / 'backup_acceptance.json').read_text())
    acceptance = json.loads((snapshot / 'evaluation_acceptance.json').read_text())
    archive.validate_acceptance(acceptance)
    require(acceptance['complete'] and backup['passed'], 'Unaccepted local archive')
    require(archive.sha256(snapshot / 'evaluation_acceptance.json') == report['source_acceptance_sha256'],
            'Source acceptance has changed')
    sources = {'results.json': archive.sha256(source)}

    def verified(relative):
        path = snapshot / relative
        expected = backup['files'][relative]['sha256']
        archive.verify_file(path, expected)
        sources[relative] = expected
        return path

    paired = json.loads(verified('paired_rollout_acceptance.json').read_text())
    require(paired['passed'] and paired['steps'] == 200 and paired['trajectories_per_arm'] == 6400,
            'Missing complete matched-input audit')
    questions = {}
    for name in MODELS:
        accepted = acceptance['models'][name]
        require(accepted['per_task'] == report['per_model'][name]
                and accepted.get('train_eval_prompt_verified') is True
                and accepted['grader_sha256'] == shared.GRADER_SHA, 'Evaluation evidence mismatch')
        card = json.loads(verified(f'evaluations/{name}/eval_card.json').read_text())
        expected = dict(n=8, temperature=1., top_p=.9, max_tokens=16384, eval_seed=21,
                        rollout_seeds=list(range(21, 29)), enable_thinking=False,
                        prompt_protocol=PROTOCOL, retain_rollouts=True, source_commit=COMMIT)
        require(all(card.get(k) == v for k, v in expected.items()), 'Evaluation protocol drift')
        questions[name] = {}
        for task in TASKS:
            relative = f'evaluations/{name}/outputs/{task}_graded.jsonl'
            path = verified(relative)
            hashes = [v for p, v in accepted['sha256'].items() if p.endswith('/' + relative)]
            require(hashes == [sources[relative]], 'Graded source is not the accepted artifact')
            rows = shared.compact_question_scores(path, task, sources[relative])
            verify_question_scores(rows, accepted['per_task'][task], task)
            questions[name][task] = rows
    public = {'complete': True, 'student': 'Qwen/Qwen3-1.7B', 'teacher': 'Qwen/Qwen3-8B',
              'protocol': PROTOCOL, 'training_seed': 21, 'steps': list(STEPS),
              'per_model': report['per_model'], 'bootstrap_replicates': 10000,
              'bootstrap_seed': 20260923, 'bootstrap_scope': shared.UNCERTAINTY_SCOPE,
              'bootstrap': {}, 'source_sha256': sources, 'training': {}}
    for step in STEPS:
        public['bootstrap'][str(step)] = {
            task: shared.paired_bootstrap(questions[f'block3_mean_step{step}'][task],
                                         questions[f'token_opd_step{step}'][task], seed=20260923)
            for task in TASKS}
    for arm in ARMS:
        card = json.loads(verified(f'{arm}/run_card.json').read_text())
        require(card == report['training'][arm]['run_card'], 'Training identity changed')
        stats = [json.loads(s) for s in verified(f'{arm}/diagnostics/scalars.jsonl').read_text().splitlines()]
        require([s['step'] for s in stats] == list(range(1, 201)), 'Incomplete training diagnostics')
        public['training'][arm] = {'rollout_count': 6400,
            'length_stop_rollouts': report['training'][arm]['length_stop_rollouts'],
            'generated_think_tags': report['training'][arm]['generated_think_tags'],
            'max_grad_norm': max(s['actor/grad_norm'] for s in stats)}
    require(not tables.exists() and not figures.exists(), 'Use new immutable asset directories')
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)
    (tables / 'qwen17_summary.json').write_text(json.dumps(public, indent=2) + '\n')
    with (tables / 'qwen17_results.csv').open('w', newline='') as stream:
        rows = [{'model': name, 'benchmark': task, **public['per_model'][name][task]}
                for name in MODELS for task in TASKS]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for lang in ('en', 'zh'):
        (tables / f'qwen17_step200_{lang}.tex').write_text(endpoint_table(public, lang))
        for metric in ('avg_at_8', 'pass_at_8'):
            (tables / f'qwen17_{metric}_{lang}.tex').write_text(all_checkpoint_table(public, lang, metric))
    (tables / 'qwen17_behavior.tex').write_text(behavior_table(public))
    make_figures(public, figures)
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'source', 'tables', 'figures'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.snapshot, args.source, args.tables, args.figures)
    print(json.dumps({'complete': result['complete'], 'step200': result['bootstrap']['200'],
                      'source_files': len(result['source_sha256'])}, indent=2))

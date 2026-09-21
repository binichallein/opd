#!/usr/bin/env python3
"""Plot archived paired checkpoint scores, keeping protocol strata separate."""

import argparse
import hashlib
import json
import math
from pathlib import Path


STRATA = (
    ('qwen17_ml2', 'historical_external_audited', 'Qwen 1.7B'),
    ('deepseek_justrl', 'historical_external', 'DeepSeek 1.5B'),
    ('qwen06_nonthinking', 'historical_external', 'Qwen 0.6B'),
    ('llama_historical', 'historical_external', 'Llama 1B'),
)
TASKS = (('math500', 'MATH500'), ('aime24', 'AIME24'),
         ('aime25', 'AIME25'), ('amc23', 'AMC23'))
ARMS = (('token_opd', 'Token OPD', '#1573A5', 'o', '-'),
        ('block3_mean', 'Block3 Mean', '#C34038', 's', '--'))


def select_points(rows, series, view, arm, task, metric):
    points = {}
    for row in rows:
        if (row['series'], row['view'], row['arm']) != (series, view, arm):
            continue
        step = row['checkpoint_step']
        if step in points:
            raise ValueError(f'Duplicate checkpoint in one stratum: {series} {arm} {step}')
        value = row['per_task'].get(task, {}).get(metric)
        if value is None:
            continue
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid benchmark fraction: {value}')
        if row['n'] != 8 or row['training_seed'] != 21:
            raise ValueError(f'Unexpected sampling/seed identity: {row}')
        points[step] = {'step': step, 'value': value, 'evidence_ids': row['evidence_ids']}
    return [points[s] for s in sorted(points)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing figure snapshots are immutable.')
    data = json.loads(args.input.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'axes.titlesize': 8, 'axes.labelsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    args.output.mkdir(parents=True)
    selections = {}
    for metric, label in (('avg_at_8', 'Avg@8'), ('pass_at_8', 'Pass@8')):
        fig, axes = plt.subplots(4, 4, figsize=(5.5, 6.5), sharex=True)
        fig.subplots_adjust(left=.125, right=.985, bottom=.07, top=.92, wspace=.43, hspace=.46)
        for i, (series, view, student) in enumerate(STRATA):
            for j, (task, title) in enumerate(TASKS):
                ax = axes[i, j]
                ymax = 0
                for arm, name, color, marker, linestyle in ARMS:
                    points = select_points(data['rows'], series, view, arm, task, metric)
                    if [p['step'] for p in points] != [50, 100, 200]:
                        raise ValueError(f'Incomplete historical trajectory: {series} {arm} {task}')
                    selections[f'{series}/{arm}/{task}/{metric}'] = points
                    x = [p['step'] for p in points]
                    y = [100 * p['value'] for p in points]
                    ymax = max(ymax, *y)
                    ax.plot(x, y, label=name, color=color, marker=marker,
                            linestyle=linestyle, markersize=3, linewidth=1.3)
                ax.set_ylim(0, min(100, max(2, ymax * 1.15)))
                ax.set_xlim(40, 210)
                ax.set_xticks([50, 100, 200])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.grid(axis='y', color='#DFE3E6', linewidth=.55)
                ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(title, pad=7)
                if j == 0:
                    ax.set_ylabel(f'{student}\n{label} (%)')
                if i == 3:
                    ax.set_xlabel('Training step')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
                   bbox_to_anchor=(.55, .986))
        fig.savefig(args.output / f'history_{metric}.pdf')
        fig.savefig(args.output / f'history_{metric}.png', dpi=220)
        plt.close(fig)
    provenance = {'schema_version': 1,
                  'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'units': 'percent, no task aggregation',
                  'note': 'Separate historical protocol strata. Lines connect measured checkpoints; no interpolated evaluations or training-seed intervals.',
                  'selected_points': selections}
    (args.output / 'history_figures_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'curves': len(selections), 'figures': 2, 'output': str(args.output)}))


if __name__ == '__main__':
    main()

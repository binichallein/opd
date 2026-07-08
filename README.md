# OPD Internal Experiments

This is an internal research repository for validating On-Policy Distillation
(OPD) variants for language-model post-training.

The current codebase contains clean-room OPD training/evaluation scripts,
focused tests for block supervision, curated result summaries, and the HTML
report for the block reverse-KL experiments.

## What Is Versioned

- Training, evaluation, analysis, and launch scripts in `scripts/`
- Unit tests in `tests/`
- Research notes in `manuscript/`
- Curated summaries in `results/*.json`
- Human-readable reports in `reports/`
- Figures in `figures/`
- Experiment and data policies in `docs/`
- Reproducible experiment configs in `configs/`

## What Is Not Versioned

Do not commit raw data, processed datasets, checkpoints, model weights, full
prediction files, logs, cache directories, or credentials. The `.gitignore`
is intentionally strict because the repository is meant to preserve experiment
logic and conclusions, not duplicate large artifacts.

## Current Main Result

The latest validated experiment tests block-level reverse-KL OPD with
advantage aggregation:

- `naive block-3`: sum token advantages inside each 3-token block
- `mean block-3`: average token advantages inside each 3-token block
- `mixed block-3`: interpolate token-local and block-mean advantages

On the July 8, 2026 run, `mean block-3` reduced gradient norm substantially
while slightly improving both GSM8K and MATH500 versus naive block-3. The
current conclusion is that block OPD is most promising as variance-controlled
block advantage aggregation, not naive block-size scaling.

See:

- `reports/block_opd_experiment_report.html`
- `docs/results/2026-07-08-block-advantage.md`

## Quick Checks

```bash
python -m py_compile scripts/clean_opd_train.py scripts/summarize_block_advantage_experiment.py
python -m pytest tests/test_block_supervision.py -q
```

The current local environment used for these checks was the `vllm` conda
environment on the experiment machine.

## Remote Training Policy

Long-running training must be launched detached on the training server, must
write a complete checkpoint state, and must pass a small resume gate before
using real GPU budget. See `docs/experiment_protocol.md`.

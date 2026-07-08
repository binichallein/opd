# Repository Organization Plan

## Objective

Make the internal OPD repository useful for two workflows:

1. Preserve experiment code and conclusions without leaking or duplicating
   heavy artifacts.
2. Give AI coding agents enough structure to understand the project quickly
   and avoid using the wrong model or data.

## Chosen Layout

- `scripts/`: current executable research scripts. These remain flat for now
  because the active experiments already use this interface.
- `tests/`: targeted regression tests for fragile logic such as OPD loss units.
- `configs/`: reproducible experiment definitions.
- `docs/policies/`: data and model safety rules.
- `docs/ideas/`: research idea definitions.
- `docs/results/`: compact experiment conclusions.
- `results/*.json`: small curated result tables only.
- `reports/`: readable HTML summaries for discussion.
- `manuscript/`: longer research notes and drafts.

## Artifact Boundary

Git stores code, configs, summaries, figures, and reports. It does not store
raw data, generated JSONL predictions, checkpoints, model weights, logs, cache
directories, or credentials.

## AI Readability

`AGENTS.md` is the main entry point for future agents. It records the default
official models, forbidden artifacts, full-eval definition, and the required
checks before reporting completion.

## Future Refactor

Once the active experiments stabilize, move reusable code from `scripts/` into
`src/opd/` and leave the old scripts as thin CLI wrappers. That refactor should
be separate from experiment-result commits.

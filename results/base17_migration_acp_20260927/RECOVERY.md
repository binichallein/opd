# ACP Migration Preflight Recovery

Transfer completed at 2026-09-27 02:13:29 Beijing: all 2355 files and
101849693103 bytes passed source/destination hashes.

The v1 queue stopped at 02:22 before any GPU evaluation or training. Its source
inventory included nine Hydra-generated configuration snapshots from the three
source training invocations. These are evidence, not algorithm source files.
The 912 shared vendor source/configuration files and seven `opd_ext` files had
identical bytes. No imported checkpoint was changed or retrained.

Commit `781df3021f8b84936921e068aa2251e38e686bfd` narrowly excludes
`external/revisiting_opd/outputs/YYYY-MM-DD/HH-MM-SS/.hydra/` snapshots named
`config.yaml`, `hydra.yaml`, or `overrides.yaml` from source inventory comparison.
The original transfer manifest still covers these source snapshots. The test
rejects unexpected Python files, non-dated output configuration, and genuine
trainer configuration drift. Local regression first failed, then all65 focused
tests passed; actual ACP runtime174 tests, real source comparison and deployment
hashes passed.

Recovery uses a NEW frozen deployment and run, not a hot patch or overwrite:

- Runtime: AFS `deployments/781df3021f8b84936921e068aa2251e38e686bfd`.
- Run: AFS `runs/20260927v2_base17_grpo_migrated_n1_seed21_acp`.
- Detached controller351590 launched at02:45:32 Beijing; read live state.
- Original v1 failure, imported states, source models and raw rollouts retained.
- Data, prompts, losses, training/evaluation parameters and job order unchanged.
- At launch, full preflight was pending; neither evaluation nor Token training
  should be reported as started until its actual process and outputs are verified.

AFS root: `/mnt/afs/202609/tyf-qwen-opd`.

## Supervised Evaluation Startup

On September27, supervision followed v2 from preflight through actual inference:

- Full imported state/rollout acceptance passed: four checkpoints,16rank states,
  all3200 original training trajectories. Actual707 input contract and historical
  grader checks passed. No experiment configuration or runtime was changed.
- Step100 merge began03:07:34; full evaluation PID355094 began03:09:16.
- Additional CPU check verified identical original/merged tokenizer input IDs on
  all643 benchmark completion prompts,EOS151643. This is a scoped actual-input
  check, not a universal tokenization equivalence claim.
- All4 engines completed initialization by03:13:54; by03:14:29 GPUs showed83-85%
  utilization while MATH500 generated. No OOM or process exit was observed.
- Raw archives are retained per rollout; a newly opened empty archive does not
  count as completed answers. Each worker writes after its generation batch.
- Token remains queued after Block3 eval100/75/50/25. No completed evaluation
  score or method effectiveness result is established by this startup check.
- Structured snapshot: `20260927_first_eval_start.json`, also retained under
  the v2 run's `supervision/` directory on AFS.

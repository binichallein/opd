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

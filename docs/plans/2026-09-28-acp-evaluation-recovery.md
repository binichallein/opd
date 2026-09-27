# ACP Interrupted Evaluation Recovery

## Scope

Recover only missing Token OPD Step25 evaluation for the completed public
Qwen3-4B-Base-GRPO to Qwen3-1.7B-Base 100-step,32x1,seed21 experiment.
The original seven accepted evaluations and all training states remain intact.

## Execution

- [x] Verify replacement worker, four idle H100 GPUs, persistent AFS, existing
  migration environment and completed training/partial generation evidence.
- [x] Add a separately versioned controller and failing-then-passing tests.
  Focused suite:136 passed locally, including archive and queue regressions.
- [ ] Deploy controller separately from frozen evaluation runtime781df30.
- [ ] Verify frozen source/model/data/accepted-output hashes and package alignment.
  Copy and audit existing4000 MATH500 and240 AIME24 raw responses and archives.
- [ ] Detached generation of240 AIME25 and664 AMC23 responses, same prompt,
  historical grader,8 samples,seeds21-28,temperature1,top_p0.9,max_tokens16384.
- [ ] Grade/audit all5144 Token25 responses, produce full eight-model comparison
  and remaining imported Block3 diagnostics plots. Archive summaries locally/GitHub.

All outputs use a new recovery subdirectory. No old source files, weights,
accepted evaluations or queue states are overwritten. Unexpected partial data
or hash mismatch stops recovery. No automatic retries, pruning or training.
Worker replacement is recorded, not claimed as bitwise reproducible generation.
Block3 training was on A100; Token training was on H100.

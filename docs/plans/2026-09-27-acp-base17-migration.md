# ACP Base1.7B Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move remaining evaluation and Token OPD work from ml2 to ACP without retraining or changing the completed Block3 model.

**Architecture:** Preserve the failed ml2 queue and its artifacts. Copy verified assets into AFS, then run a new immutable ACP continuation: imported Block3 evaluation100/75/50/25, independent Token save/resume gate and100-step training, then Token evaluation100/75/50/25. Reuse the existing queue, collector, loss, grader and evaluation implementations.

**Tech Stack:** Python, existing verl/vLLM scripts, SSH, AFS, pytest.

## Fixed Contract

- Public Qwen3-4B-Base-GRPO teacher and original Qwen3-1.7B-Base student, identical source bytes.
- DAPO training data and physical-row schedule unchanged;32 prompts x1 response,100 steps,seed21,legacy per-request seed21,lr2e-6.
- Completion prompt with boxed instruction and Solution:,native EOS151643,no ChatML/think.
- Token full states25/50/75/100 preserved, no pruning; Block3 existing full states remain protected on ml2.
- Full four benchmarks, each643 questions x8 collectively, scored separately with historical grader; all raw rollouts and diagnostics retained.
- No initial-student re-evaluation and no Block3 retraining.
- Explicit hardware/environment difference: Block3 trained on ml2 A100; Token trains on ACP H100. Both new evaluations run on ACP. Do not claim same-hardware causal isolation or bitwise equivalence.
- ACP CPU quota32 rather than ml2 Ray64 is a deployment difference, not a change to batch/optimizer settings.

## Tasks

1. [x] Read live source failure and verify completed Block3 state, rollouts and idle ACP GPUs.
2. [ ] Transfer models, source evidence, diagnostics and four evaluable checkpoints with source/destination hashes; retain all source artifacts. Do not copy private SSH keys.
3. [x] Add `scripts/run_acp_base17_migration.py` and focused tests for import-only Block3, loss/config alignment, fail-closed gates, complete evaluation and independent Token initialization. Start with failing tests. Dedicated migration/waiter61 tests pass; broader CPU-only local suite148 passes. Spec and code-quality reviews completed.
4. [x] Verify isolated environment and plotting dependency;170 runtime tests pass on ACP. Source-aligned math/data packages pass the actual environment comparison. Imported full-file hashes/707 actual collector inputs/historical grader are mandatory deferred gates before GPU evaluation.
5. [x] Freeze e07997a deployment and launch detached AFS waiter339663, verified PPID1. Transfer receiver336441 also PPID1; no local SSH dependency after authentication.
6. [ ] Complete transfer, audit imported checkpoints/inputs, verify first GPU evaluation actually starts. Token save1/resume2 gate must pass before formal training. At01:12 Beijing only the transfer and waiter are running.
7. [x] Save migration manifest, live startup evidence and Chinese status locally/GitHub. Submitted paper and existing ACP experiments unchanged.

## Acceptance

Imported Block3 is never passed to a trainer. All evaluation checkpoints are byte-verified against ml2 source. Queue has an exclusive lock and stops on errors, preserves failures, and does not retry blindly. Transfer completion is distinguished from queue startup and experimental completion. All long-running jobs survive SSH disconnection after transfer and launch.

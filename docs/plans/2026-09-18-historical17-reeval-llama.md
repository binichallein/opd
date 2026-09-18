# Historical 1.7B Re-evaluation and Llama Restart Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-evaluate six historical Qwen3-1.7B checkpoints with lossless rollouts, then restart the matched Llama pair using the historical train/eval prompt difference explicitly requested by the user.

**Architecture:** Preserve the stopped Llama attempt and all historical inputs. A new immutable queue first evaluates Token/Block3 checkpoints 50/100/200, then runs initial Llama evaluation, Token training/full evaluation, and Block3 training/full evaluation. Reuse existing process-group isolation, historical grading, full trajectory archives, checkpoint/resume gates, and diagnostics; only version the prompt protocol and add evaluation archival.

**Tech Stack:** Python, vLLM, Revisiting OPD/verl, FSDP, pytest, SSH to ml2 only.

## Authorized Design

- User explicitly confirmed: reproduce old 1.7B TRAIN/EVAL DIFFERENCE, not a newly matched think prompt.
- Historical training user text: `Math problem: {question}\n\nPlease carefully reason through the math problem step by step and derive the correct answer. You must conduct reasoning inside <think> and </think> and give the final answer within \\boxed{}.\n`.
- Historical eval: existing eval JSONL `prompt`, thinking false, n8, seeds21-28, temperature1, top_p0.9, max_tokens16384, four GPUs, historical grader SHA256 `04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f`.
- Keep Qwen historical rendered inputs unchanged. For Llama use native template and EOS, not Qwen control tokens; fixed template date remains 18 Sep 2026. This is a native-template adaptation of the same user instructions, NOT byte-identical cross-family input.
- Llama training no longer requires zero generated think tags; record them accurately. Do not label historical training as explicitly non-thinking. Eval remains the existing no-think-request prompt.
- Keep DAPO pool and hashes, seed21, 4 prompts x8 rollouts, LR2e-6, 200 updates, max response16384, original token/block3 objectives, complete checkpoints50/100/200, all existing scalar/position diagnostics.
- Both Llama arms restart from original ModelScope Llama-3.2-1B-Instruct; teacher Llama-3.2-3B-Instruct. Never warm-start from the interrupted attempt or probes.
- Four benchmark results remain separate. Every evaluated checkpoint must produce 5144 responses with native tokens, rendered prompts, unfiltered decoded text, actual finish/stop reasons and seed identities. Record output metadata without altering generation RNG or decoding parameters.
- Interrupted Llama run `20260918v1_llama32_1b_3b_nonthinking_seed21_ml2` is retained. Controller PID297837 was intentionally SIGTERM'ed; state error143 is a user stop, not spontaneous failure. GPU idle verified afterwards.

## Tasks

### Task 1: Evaluation Archival

Files: `scripts/eval_qwen3_math_vllm.py`, new focused helper/tests if needed.

1. Write failing tests for lossless Qwen/Llama metadata and incremental exclusive archive files.
2. Add an explicit archival option. Preserve original prompt input type, batching, sampling and grader response text; decode a separate raw text field with special tokens retained.
3. Persist returned rollouts before task-wide completion; no overwrites or silent retries. Cross-check saved native IDs and output coverage.
4. Run evaluator/regression tests, including historical Qwen generation-contract parity.

### Task 2: Historical Llama Training Protocol

Files: `opd_ext/math_protocol.py`, launcher/environment/rollout hooks, focused protocol tests.

1. Write failing tests for exact legacy math instruction, native Llama rendering, truthful archive protocol/think metadata, and preserved non-thinking behavior for old protocols.
2. Add a separately named legacy Llama training protocol, retain native stop IDs/length masks and complete archives.
3. Do not change the existing frozen runtime or old run card. GPU gate verifies actual prompt IDs/native BOS/EOS and logs generated think behavior without forbidding it.

### Task 3: Ordered Queue and Audits

Files: new `scripts/run_historical17_reeval_llama.py`, focused tests, `AGENTS.md`.

1. Write failing tests for six checkpoint identities/order, unchanged historical eval contract, mandatory full acceptance before Llama, and fresh paired initialization.
2. Verify old configurations/model/data/grader hashes. Protect original outputs/checkpoints. Record exact source/config hashes.
3. Reuse old evaluator architecture with logging-only extension. Reuse Llama orchestration through narrowly scoped parameters/helpers rather than copying unrelated training logic.
4. Queue six Qwen evaluations first; then Llama initial full evaluation; Token two-step resume gate + formal200 + eval200/100/50; Block3 equivalents; paired report and protection checks.
5. Failure stops queue; no pruning, automatic retries, extra seeds or GPU competition.

### Task 4: Verification and Deployment

1. Run focused tests plus broad protocol/rollout/eval/queue/checkpoint regression suite.
2. Review diff, commit code, deploy new immutable release via existing Git archive/hash procedure.
3. Verify GPU idle and old controller stopped, launch one nohup controller, record PID/command/runtime.
4. Observe initial preflight and first real GPU evaluation/archival output; report actual status, never future jobs as completed.

## Evidence Limits

- Re-generated rollouts are new inference artifacts, not recovered historical training trajectories.
- Equal settings do not guarantee bit-identical sampled outputs across different engine/runtime versions. Record current engine/package versions and any historical evidence gaps.
- This intentionally reintroduces the historical train/eval instruction difference for Llama; do not describe it as matched-prompt non-thinking training.

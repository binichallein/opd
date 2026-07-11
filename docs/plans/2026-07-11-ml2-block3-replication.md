# ML2 Block3 Replication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the train-validated `block3_mean` configuration on `ml2` with memory-safe runtime settings, full diagnostics, Step 50/100/200 checkpoints, complete n=8 evaluation, and reproducible heatmaps.

**Architecture:** Reuse the existing generic VERL training and evaluation launchers. Add a Block3-specific control/sync layer, parameterize the existing audit for Block3 milestones, and add a single-run diagnostic plotting entry point that reuses the validated Block10 plotting functions.

**Tech Stack:** Bash, Python 3, PyTorch/VERL, Ray, vLLM, NumPy, Matplotlib, pytest, SSH/TAR.

---

### Task 1: Parameterize The Diagnostic Run Audit

**Files:**
- Modify: `scripts/audit_block10_run.py`
- Create: `tests/test_diagnostic_run_audit.py`

**Step 1: Write the failing tests**

Add tests that import the audit module and verify:

```python
def test_parse_step_list_accepts_sorted_unique_steps():
    assert parse_step_list("50,100,200") == [50, 100, 200]

def test_expected_run_card_can_target_block3():
    expected = expected_run_card("block3_mean")
    assert expected["variant"] == "block3_mean"
    assert expected["rollout_gpu_memory_utilization"] == 0.6
    assert expected["ref_log_prob_micro_batch_size_per_gpu"] == 1
```

**Step 2: Run the focused tests and verify RED**

Run:

```bash
/home/tyf/miniconda3/envs/vllm/bin/python -m pytest -q tests/test_diagnostic_run_audit.py
```

Expected: import or attribute failure because the helpers do not exist.

**Step 3: Implement the minimal parameterization**

- Add `--variant`, `--checkpoint-steps`, `--eval-steps`, `--world-size`, and `--skip-eval` arguments.
- Keep current Block10 defaults so existing commands remain compatible.
- Parse comma-separated milestone lists with a tested helper.
- Build the expected run-card dictionary from the requested variant.
- Make checkpoint shard count use `--world-size`.
- Preserve all existing eval and diagnostic checks.

**Step 4: Verify GREEN and regression safety**

Run the focused tests, then the complete suite:

```bash
/home/tyf/miniconda3/envs/vllm/bin/python -m pytest -q tests/test_diagnostic_run_audit.py
/home/tyf/miniconda3/envs/vllm/bin/python -m pytest -q
```

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/audit_block10_run.py tests/test_diagnostic_run_audit.py
git commit -m "feat: parameterize OPD diagnostic audit"
```

### Task 2: Add The ML2 Block3 Control And Sync Layer

**Files:**
- Create: `scripts/block3_replication_control.sh`
- Create: `scripts/sync_block3_replication_to_ml2.sh`
- Create: `tests/test_block3_replication_launcher.py`
- Modify: `scripts/launch_revisiting_block_opd_formal_train.sh`

**Step 1: Write launcher contract tests**

The tests must assert that the control script fixes:

- `VARIANT=block3_mean`.
- Seed 21, train batch 4, rollout group 8, LR `2e-6`, 200 formal steps.
- vLLM utilization 0.6 and reference log-probability micro-batch 1.
- Diagnostics interval 5, Top-16, position stride 1.
- Formal checkpoints `50,100,200` and no automatic deletion.
- Eval tasks Math500/AIME24/AIME25/AMC23 with n=8 and seeds 21-28.
- All cache paths under `/limx_embap/tos`.
- Actions for `sync`, `probe1`, `probe2`, `formal`, `status`, `eval`, and `audit`.

Also assert that the formal launcher records the actor PPO and rollout log-probability micro-batch values in `run_card.json`, removing ambiguity from the historical card.

**Step 2: Run tests and verify RED**

Run:

```bash
/home/tyf/miniconda3/envs/vllm/bin/python -m pytest -q tests/test_block3_replication_launcher.py
```

Expected: missing script or missing run-card fields.

**Step 3: Implement the control script**

- Use only host `ml2`.
- Use TOS-backed paths for run data and caches.
- Launch probe Step 1 with a complete checkpoint, then resume and advance to Step 2.
- Launch the formal 200-step run with milestones 50/100/200.
- Route eval actions through `launch_qwen3_math_eval.sh`.
- Route final audit through the parameterized audit.
- Record source commit, run lifecycle, exact commands, environment, data/model hashes, and checkpoint policy.

**Step 4: Implement deterministic sync**

- Copy only the required tracked runtime files with `tar` over SSH.
- Create required remote directories first.
- Print SHA-256 for every copied file.
- Never write caches to `/tmp` on `ml2`.

**Step 5: Verify GREEN and shell syntax**

Run:

```bash
bash -n scripts/block3_replication_control.sh
bash -n scripts/sync_block3_replication_to_ml2.sh
/home/tyf/miniconda3/envs/vllm/bin/python -m pytest -q tests/test_block3_replication_launcher.py
/home/tyf/miniconda3/envs/vllm/bin/python -m pytest -q
```

Expected: all checks pass.

**Step 6: Commit**

```bash
git add scripts/block3_replication_control.sh scripts/sync_block3_replication_to_ml2.sh \
  scripts/launch_revisiting_block_opd_formal_train.sh tests/test_block3_replication_launcher.py
git commit -m "feat: add ml2 Block3 replication control"
```

### Task 3: Add Single-Run Diagnostic Plots And Heatmaps

**Files:**
- Modify: `scripts/analyze_block10_collapse_diagnostics.py`
- Create: `scripts/analyze_single_opd_diagnostics.py`
- Create: `tests/test_single_opd_diagnostic_analysis.py`

**Step 1: Write failing plotting tests**

Add tests that verify:

- Heatmap and entropy-segment plotting functions accept exactly one host without indexing errors.
- The single-run CLI declares scalar alignment, credit, optimization, position heatmap, and entropy-segment outputs.
- The generated HTML uses relative paths and names all monitored metrics accurately.

Use small synthetic diagnostic fixtures; do not depend on remote run data.

**Step 2: Run tests and verify RED**

Run:

```bash
/home/tyf/miniconda3/envs/vllm/bin/python -m pytest -q tests/test_single_opd_diagnostic_analysis.py
```

Expected: single-host plotting or missing-script failure.

**Step 3: Implement single-host support**

- Use `squeeze=False` for subplot arrays whose host dimension may be one.
- Reuse the existing scalar groups and diagnostic snapshot loader.
- Generate:
  - `scalar_alignment.png`
  - `scalar_credit.png`
  - `scalar_optimization.png`
  - `position_heatmaps.png`
  - `entropy_segments.png`
  - `diagnostics.html`
- Label the run as Block3 and avoid Block10-specific causal classification.

**Step 4: Verify GREEN**

Run focused and full tests. Render the synthetic HTML with Playwright at desktop and mobile widths.

**Step 5: Commit**

```bash
git add scripts/analyze_block10_collapse_diagnostics.py scripts/analyze_single_opd_diagnostics.py \
  tests/test_single_opd_diagnostic_analysis.py
git commit -m "feat: plot single-run OPD diagnostics"
```

### Task 4: Sync And Execute The Resume Probe

**Files:**
- Remote run artifacts only; no source edits expected.

**Step 1: Push the implementation branch**

```bash
git push -u origin feat/ml2-block3-replication
```

**Step 2: Sync the exact source state**

```bash
bash scripts/block3_replication_control.sh sync
```

Verify local and remote SHA-256 outputs match.

**Step 3: Launch probe Step 1**

```bash
bash scripts/block3_replication_control.sh probe1
```

Wait for exit code 0. Verify Step-1 checkpoint shards, `data.pt`, diagnostics NPZ/JSONL, source commit, and no CUDA OOM/non-finite diagnostics.

**Step 4: Resume to Step 2**

```bash
bash scripts/block3_replication_control.sh probe2
```

Verify the log explicitly loads `global_step_1`, advances to Step 2, writes a complete Step-2 checkpoint, and exits 0.

**Step 5: Record probe evidence**

Save a concise local/remote probe acceptance summary. Do not delete the probe artifacts.

### Task 5: Launch And Supervise The Formal Run

**Files:**
- Remote run artifacts only.

**Step 1: Recheck launch gates**

- All four GPUs idle.
- TOS filesystem writable with sufficient space.
- No stale Ray/vLLM process.
- Formal run directory absent or explicitly empty.
- Probe acceptance passed.

**Step 2: Launch the formal run**

```bash
bash scripts/block3_replication_control.sh formal
```

Record PID, command, run root, source commit, and start time.

**Step 3: Monitor every diagnostic interval**

At each new diagnostic point, inspect entropy, overlap mass, WeightedSignFlipRate, NormalizedLeakage, block ratio outside-clip fraction, PG loss, grad norm, response length/truncation, and non-finite counts. Report meaningful changes rather than raw logs.

**Step 4: Verify completion**

Require 200/200, exit code 0, 41 diagnostic points, and complete Step 50/100/200 checkpoints. Preserve any shutdown warning separately from the main status.

### Task 6: Evaluate, Plot, Audit, And Report

**Files:**
- Create after results exist: `results/2026-07-*-block3-ml2-replication.json`
- Create after results exist: `reports/block3_ml2_replication.html`
- Create after results exist: `reports/block3_ml2_replication_assets/*.png`

**Step 1: Evaluate all milestones**

Run the control script's eval action for Step 50, 100, and 200, one at a time. Require exit code 0 and complete 5,144-rollout summaries for each.

**Step 2: Run final audit**

```bash
bash scripts/block3_replication_control.sh audit
```

Require no issues and no non-finite warnings.

**Step 3: Fetch curated artifacts and generate plots**

Fetch run cards, manifests, diagnostics, acceptance, and eval summaries. Run the single-run analyzer and verify all five PNGs render.

**Step 4: Build the comparison report**

Compare:

- Historical train token OPD Step 200.
- Historical train Block3 Step 200.
- New ml2 Block3 Steps 50/100/200.

State the vLLM/reference micro-batch difference and one-seed limitation. Do not claim statistical replication.

**Step 5: Verify and commit**

- Validate result JSON against authoritative summaries.
- Render HTML in desktop and mobile Playwright screenshots.
- Open the final HTML in Windows Chrome.
- Run the full test suite and `git diff --check`.
- Commit and push the result artifacts.

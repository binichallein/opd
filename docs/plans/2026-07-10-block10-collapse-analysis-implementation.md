# Block10 Collapse Analysis HTML Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone offline Chinese HTML page that explains the best-supported cause of the `block10_mean` collapse and opens correctly in Windows Chrome.

**Architecture:** Create one self-contained semantic HTML file with embedded CSS and no JavaScript or remote assets. Reuse the repository report's restrained visual language while presenting equations, a CSS timeline, a feedback loop, evidence grades, and falsification tests as separate full-width sections.

**Tech Stack:** HTML5, embedded CSS, Windows Chrome headless rendering, shell-based structural checks.

---

### Task 1: Build the Evidence Narrative

**Files:**
- Create: `reports/block10_collapse_analysis.html`

**Step 1: Add the page shell and metadata**

Create a Chinese HTML5 document with:

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Block10 OPD 崩塌原因分析</title>
</head>
```

The header must identify the student, teacher, data, seed, step, and evidence
status.

**Step 2: Add the executive verdict and observed facts**

Include the exact findings:

- all 200 pre-clip gradient norms exceed the 1.0 clipping threshold;
- median 4.141, P95 47.143, maximum 420.710;
- no NaN/Inf and no gradient disappearance;
- all 5,144 evaluation rollouts are format errors;
- collapse onset is approximately steps 50-60.

**Step 3: Add equations and the cross-credit expansion**

Render the following as styled HTML equations:

```text
A_t = log pi_T(y_t | h_t) - log pi_old(y_t | h_t)
A_bar_B = (1/k) sum A_t
log r_B = sum [log pi_theta(y_t | h_t) - log pi_old(y_t | h_t)]
grad L_B proportional to -A_bar_B sum grad log pi_theta(y_t | h_t)
```

Visually distinguish diagonal token-credit terms from cross-token terms and
show the ordered cross-term counts `6`, `20`, and `90` for k=3/5/10.

**Step 4: Add evidence boundaries**

State that the mechanism is the best-supported explanation, not a proven causal
claim, because block10 ran on A100 while the other variants ran on A800 and only
one seed is available.

**Step 5: Run content assertions**

Run:

```bash
rg -n "420.710|90 个|5,144|A100|A800|不是.*单次.*梯度" reports/block10_collapse_analysis.html
```

Expected: every required claim is present.

### Task 2: Add Visual Explanations

**Files:**
- Modify: `reports/block10_collapse_analysis.html`

**Step 1: Add the training timeline**

Create a CSS-only timeline covering selected steps 40, 48, 53, 58, 60, 66,
70, 76, and 90. Each point must show entropy, gradient norm, average response
length, and outcome reward where relevant.

**Step 2: Add the feedback loop**

Use flex/grid blocks and arrows for:

```text
cross-token credit leakage -> normal-token suppression -> entropy growth ->
OOD student prefixes -> more negative sampled-token feedback -> longer/random
rollouts -> continued clipped updates
```

**Step 3: Add the four-method gradient comparison**

Show post-step-50 values:

| Method | Mean | P95 | Steps > 10 |
|---|---:|---:|---:|
| token | 1.233 | 1.622 | 0 |
| block3 | 2.118 | 2.228 | 1 |
| block5 | 2.451 | 2.824 | 1 |
| block10 | 7.289 | 20.735 | 21 |

**Step 4: Add responsive behavior**

At widths below 760px, stack multi-column layouts, make tables horizontally
scrollable, and keep all labels inside their containers.

### Task 3: Verify Chrome Rendering

**Files:**
- Verify: `reports/block10_collapse_analysis.html`

**Step 1: Run structural checks**

Run:

```bash
xmllint --html --noout reports/block10_collapse_analysis.html
```

Expected: only possible legacy-parser warnings for HTML5 semantic tags; no
unclosed-tag or malformed-table errors.

**Step 2: Render desktop in Windows Chrome**

Run Chrome headless at `1440x5000`, save a PNG in the Windows temporary
directory, and inspect it for overlap, clipping, blank sections, and unreadable
text.

**Step 3: Render mobile in Windows Chrome**

Run Chrome headless at `430x3000`, inspect the screenshot, and correct any text
overflow or table/layout overlap.

**Step 4: Open the final page interactively**

Use Windows PowerShell `Start-Process` with the Chrome executable and the WSL
UNC path to the HTML file.

### Task 4: Synchronize and Commit

**Files:**
- Add: `reports/block10_collapse_analysis.html`
- Add: `docs/plans/2026-07-10-block10-collapse-analysis-design.md`
- Add: `docs/plans/2026-07-10-block10-collapse-analysis-implementation.md`

**Step 1: Copy the HTML into the cleanroom report directory**

```bash
cp reports/block10_collapse_analysis.html ../opd_cleanroom/reports/
```

**Step 2: Verify hashes**

```bash
sha256sum reports/block10_collapse_analysis.html \
  ../opd_cleanroom/reports/block10_collapse_analysis.html
```

Expected: identical hashes.

**Step 3: Commit and push**

```bash
git add reports/block10_collapse_analysis.html \
  docs/plans/2026-07-10-block10-collapse-analysis-implementation.md
git commit -m "Add block10 collapse analysis report"
git push origin main
```

**Step 4: Final verification**

Confirm the worktree is clean, `HEAD == origin/main`, Chrome is running with the
page, and both desktop/mobile screenshots exist.

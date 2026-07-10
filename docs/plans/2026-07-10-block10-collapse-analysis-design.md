# Block10 Collapse Analysis Page Design

## Purpose

Create a standalone Chinese HTML report explaining why the `block10_mean`
experiment collapsed. The page must connect the implemented objective to the
observed training dynamics without overstating a single-seed, cross-hardware
result as definitive causality.

## Audience

The primary audience is the experiment owner and future research agents. The
page should be readable without opening source code or parsing training logs,
while retaining enough mathematical detail to guide the next experiment.

## Content Structure

1. Executive verdict: not gradient disappearance or a single numerical
   explosion, but a block-length-dependent credit-assignment instability.
2. Implementation formula: show token OPD, block mean advantage, block PPO
   ratio, and the desired diagonal terms versus cross-token terms.
3. Scaling argument: compare 6, 20, and 90 ordered cross-token interactions for
   block sizes 3, 5, and 10.
4. Training timeline: visualize steps 40-90 with selected entropy, gradient,
   response length, and outcome reward observations.
5. Feedback loop: credit leakage -> distribution drift -> OOD prefixes -> more
   negative sampled-token feedback -> entropy/length inflation -> continued
   destructive updates.
6. Gradient interpretation: distinguish pre-clip spikes from actual clipped
   updates and explain why clipping controls magnitude but not direction.
7. Evidence grading: separate directly observed facts, mechanism-supported
   interpretations, secondary hypotheses, and unresolved confounders.
8. Falsification plan: specify the smallest same-machine diagnostic runs and
   instrumentation needed to distinguish the proposed mechanisms.

## Visual Design

- Reuse the established OPD report typography and restrained work-focused
  styling, but make this a separate file.
- Use a dark header, white unframed sections, compact evidence panels, and a
  mixed blue/green/amber/red palette.
- Render equations as styled HTML, not raw Markdown or external MathJax.
- Use CSS-only timelines and flow diagrams so the page works offline.
- Keep cards at 8px radius or less and avoid nested cards.
- Support desktop and mobile widths without clipped text or overlapping tables.

## Evidence Rules

- State that `actor/grad_norm` is the pre-clip norm returned by
  `clip_grad_norm_`, with a configured threshold of 1.0.
- Do not call the collapse a proven consequence of `k=10`; label the mechanism
  as the best-supported explanation from code and one run.
- Record the A800/A100, driver, and Python patch-version confounders.
- Note that isolated gradient spikes are insufficient: block5 had a large
  single spike without collapsing.
- Treat multiplicative block-ratio variance as secondary because observed PPO
  clip fractions were modest near the onset.

## Acceptance Criteria

- The page includes the exact four-method gradient comparison and collapse
  timeline values used in the analysis.
- No stale claim says block10 is still running.
- HTML parses without structural errors and renders without overlap at 1440px
  desktop and a narrow mobile viewport.
- Windows Chrome opens the final local file.
- The page and design/plan documents are committed and pushed to the internal
  repository.

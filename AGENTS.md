# Agent Instructions

This repository is an internal OPD research workspace. Optimize for
reproducibility, data safety, and clear experiment lineage.

## Authorized Qwen8 Teacher Qualification (2026-09-21)

- FINAL: Qwen8-to1.7 diagnostics completed21:40:04 Beijing; controller exited,
  all four GPUs idle22:10, no training. Read
  `docs/results/2026-09-21-qwen8-to17-final.md` BEFORE proposing any next run.
  Base gate inconclusive: direct13/128 for both, continuation13/64 vs16/64.
  Instruct non-thinking gate passed: direct38/128 vs60/128, continuation18/64
  vs32/64 (student vs teacher). Teacher continuation cap6/64, periodic3/64 are
  near frozen limits, not a clean bill of health. All768 diagnostic +8 smoke
  rollouts retained; this is not full benchmark evaluation or OPD validation.
  Post-run CPU audit validated all10 cells, sealed hashes, historical regrading
  and recomputed decisions: diagnostics/20260921_qwen17_pair_postrun_audit/report.json.
  No new training authorization. Below are historical launch snapshots.
- NEW authorization: test 8B-Base versus original1.7B-Base, then official
  Qwen3-8B versus Qwen3-1.7B post-trained models in non-thinking mode. Read
  `docs/plans/2026-09-21-qwen8-to17-diagnostics.md`. This is INFERENCE ONLY,
  no training even when capability passes. New controller
  `scripts/run_qwen17_pair_diagnostics.py`, root
  `runs/20260921v1_qwen8_to17_diagnostics_ml2`; inspect LIVE state before launching.
  Models from pinned official ModelScope; Base completion, instruct native chat
  with explicit enable_thinking=False and GPU smoke on each model before main eval.
  Reuse original64 questions, n2,32 student-prefix questions, same sampling/grader;
  keep all768 diagnostic +8 smoke rollouts. No replacement of old runtime/results.
- New queue STARTED19:39:17 Beijing, controller1872503, PPID1/SID1872503;
  immutable control ce653fbe5a9aebceadd37d6f8cfd33e46a1bd11d. At19:43 Base1.7
  assets completed, read-only8B-Base verification running, no model scores yet.
  Read `docs/results/2026-09-21-qwen8-to17-diagnostics-startup.md` and LIVE state.
  Do not duplicate launch or redeploy over this release. Actual ml2 CPU tests264pass.
- Old8B-versus4B queue ENDED17:36 Beijing, capability_not_accepted; gate status
  inconclusive, failures direct_gain/direct_ci, no training. All six cells completed.
  OldGRPO continuation16/64; gate SHA256
  ef60b0889ff0041432ddea91c59bf9bf5641dde716f3df62081bd8bd9bc396f8.
  Below are historical snapshots, not permission to restart the old training queue.
- Supervision snapshot17:25 Beijing: five of six diagnostic cells complete.
  Direct student16/128,8B13/128,oldGRPO42/128; continuation student11/64,8B12/64.
  Direct gain/CI prerequisite not met; do NOT bypass the capability gate or train.
  OldGRPO continuation PID1852491 started17:23:10; final summary still pending.
  Read `docs/results/2026-09-21-qwen8-teacher-supervision.md` and LIVE state; the
  bounded foreground watch ended, original remote queue continues unchanged.
- Latest user selects official ModelScope Qwen/Qwen3-8B-Base, first qualify its
  ability, then train original Qwen3-4B-Base. Read
  `docs/plans/2026-09-21-qwen8-teacher-acceptance.md` before any launch.
- New root `runs/20260921v1_qwen4_from8b_completion_seed21_ml2`, controller
  `scripts/run_qwen8_teacher_pair.py`; ml2 only. Check live state to avoid duplicate
  queues. Never replace old experiments or frozen training runtime0f9161f.
- A fixed64-question/n2 capability diagnostic and fixed32-question/n2 student
  continuation test must pass before GPU updates. Original4B-GRPO is reference
  only. Rejection/inconclusive stops automatically, no automatic threshold tuning.
- Both methods must pass a NEW Step1/save/Step2/resume gate with the8B teacher.
  Then Token200 followed by original Block3Mean200, from the original student
  independently; keep all50/100/150/200 states, every rollout and diagnostic.
- Same DAPO bytes/seed21/completion prompt/16K responses/historical losses and
  normalization. No new full-benchmark eval autostart in this controller.
- ModelScope teacher revision is pinned and hashed by the new asset manifest;
  the old launcher's teacher_model_revision field stays empty because its Qwen
  branch assumes HF_REVISION. Never create a fake HF revision or alter old code.
- Below are historical authorizations, not permissions to restart old queues.
- Controller1839519 started16:14:23 Beijing, PPID1/SID1839519 verified; control
  release55e24cfc2a4a3891d348a8247b712e4ead4e0f6a. Read
  `docs/results/2026-09-21-qwen8-teacher-startup.md` and LIVE queue_state before
  reporting or acting. At16:15 downloading8B shards, no capability result/training.
  Never duplicate this queue. Old failed CPU asset init33d883d is archived; NFS
  hardlink publication fixed and tested. Short training cache=/dev/shm/q48.

## Completed Qwen4 Evaluation and Paper (2026-09-21)

- Token evaluation completed03:43:15 Beijing, all200/150/100/50 accepted; root
  acceptance passed/complete and protected inputs verified. Four GPUs were idle
  at14:52 Beijing. Do not relaunch evaluation or infer authorization for new jobs.
- Current paper: `paper/iclr2027/`, final results `generated/qwen4_publication/`.
  Read `docs/results/2026-09-21-qwen4-evaluation-and-paper-final.md` first.
  Current code/docs commits never replace immutable training/evaluation deployments.
- Preserve all historical results including failures. Block3 changes advantages,
  joint ratios and reduction, not pure advantage smoothing. Report each benchmark
  separately and all checkpoints; one training seed is not multiple replicates.
- Dataset overlap/exposure audits are retrospective; full scores remain primary.
  The496-question MATH sensitivity is not a new benchmark or proof of decontamination.
- English/Chinese paper and anonymous LaTeX sources require strict build acceptance.
  `internal/`, raw outputs, infrastructure paths and private checkpoints must not
  enter the anonymous package. No automatic OpenReview submission.

## Authorized Qwen4 Completion Token Control (2026-09-20)

- NEW authorization2026-09-21: Token training completed and passed all four
  checkpoint audits plus all6400 paired rollout input checks. Launch full Token
  evaluation NOW, order200,150,100,50, using `run_qwen4_completion_eval.py --variant
  token_opd`, new root `20260921v1_qwen4_completion_token_eval_seed21_ml2`.
  Same frozen0f runtime, completion prompt, n8, four separate benchmarks,
  historical grader and raw rollout archives as the completed Block3 evaluation.
  No training or initial-student evaluation autostart; never overwrite old runs.
  After evaluation, prepare English and Chinese ICLR2027 manuscripts using all
  archived experiments, distinguish controlled evidence, pilots and confounds.
  Follow official2027 template, nine-page submission main text, anonymity and
  mandatory AI disclosure; never submit to OpenReview automatically.

- Latest user authorizes preparing Token OPD now and starting after the current
  Block3 four-checkpoint evaluation completes and passes acceptance. This overrides
  earlier no-Token authorization for this NEW queue only. Do not interrupt eval.
- Read `docs/plans/2026-09-20-qwen4-completion-token.md`. New controller
  `scripts/run_qwen4_completion_token.py`, run `20260920v1_qwen4_completion_token_seed21_ml2`.
  Always inspect live state before launching; never duplicate or overwrite a queue.
- Same original Qwen3-4B-Base/public GRPO teacher/data/seed21/completion prompt,
  independent request seeds, 200steps, all50/100/150/200 full checkpoints and every
  rollout/diagnostic. Frozen training remains0f9161f02f08287fb07f0375ad0a6bda81133ff0.
  No warm start or algorithm rewrite. Accepted Token GPU update/resume gate reused.
- Strictly compare run card, command and model/data/runtime hashes to completed
  Block3. Wait for all4 full eval acceptances plus GPU idle; fail closed otherwise.
  New tmpfs cache `/dev/shm/opd-q4-t1`. No automatic Base/Token benchmark evaluation
  in this training-only controller; it audits full state/rollouts and draws figures.
- Waiting controller1529721 launched14:34:46 Beijing with control release
  `7131bc181c0bcddb6f5f95d3c5d379297c361b79`;14:35 state waiting_for_evaluation,
  PPID1/SID1529721 verified, no train.pid yet. Do NOT launch it again. Read
  `docs/results/2026-09-20-qwen4-completion-token-startup.md` and live queue state.
  Later documentation commits must not replace either immutable control/runtime.

## Authorized Qwen4 Completion Formal Block3 (2026-09-20)

- NEW user authorization: start FULL evaluation now, Step200 first. Read
  `docs/plans/2026-09-20-qwen4-completion-block3-eval.md`. Evaluation order200,150,100,50,
  new root `runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2`, controller
  `scripts/run_qwen4_completion_eval.py`. Do not launch Token training or Base eval.
  This supersedes no-eval authorization for this new independent queue only.
  Always pass completion prompt explicitly; legacy evaluator defaults use ChatML.
- Evaluation controller1502108 launched11:58:46 Beijing; immutable control
  `645a2f391f96311fe3ca3d85d0af44c0e877acbd`, inference runtime remains0f9161f.
  At13:02 Step200 exited0 and passed all643-question/5144-rollout/32-archive
  checks; Step150 PID1511019 running, then100,50. Do NOT relaunch this queue.
  Read `docs/results/2026-09-20-qwen4-completion-evaluation-progress.md` and live
  queue state. These are absolute scores, not gain evidence without matched
  initial-student and Token baselines. Later docs commits do not replace runtime.
- Formal training completed10:44 Beijing; checkpoint audit and figures exited0,
  queue complete10:45. Four FSDP checkpoints50/100/150/200,200 diagnostics/archives
  retained. All scalar metrics finite; capped247/6400. Shutdown MathMultiProcessEnv
  destructor/DataLoader warning was ignored, main exit0 and checkpoint audit passed.
  This is not benchmark gain evidence. Original run state remains immutable.

- Latest user authorized restarting the Qwen4 pair's Block3 Mean with the accepted
  new configuration and FOUR retained checkpoints:50,100,150,200. This supersedes
  the earlier no-formal-launch instruction for this NEW run only.
- Read `docs/plans/2026-09-20-qwen4-completion-block3-formal.md`.
  Current retry root `runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2`, cache `/dev/shm/opd-q4-c3`.
  Controller `scripts/run_qwen4_completion_block3.py` uses the exact accepted
  TRAINING runtime `0f9161f02f08287fb07f0375ad0a6bda81133ff0`; controller code has
  a separate immutable analysis deployment. Never replace training code mid-run.
- Only Block3 runs automatically; no Token or benchmark autostart. Start from
  original public Qwen3-4B-Base, not stopped v1 or accepted probe checkpoints.
  All checkpoint states retained; every step keeps32 raw rollouts and diagnostics.
  All common settings match the accepted probe, except200 steps, four save nodes,
  no probe stop, and names/output paths. New completion prompt and seed rule stay.
- Before reporting status read live queue_state, train.pid and logs/nohup.log.
  Never duplicate the new controller or restart either old v1 queue/gate.
- v2 controller1173330/train1173572 failed01:10:37 Beijing before the FIRST update:
  Triton TemporaryDirectory cleanup OSError39 on NFS4. First32 rollouts preserved.
  Only its orphan worker1178906/PGID1173572 was terminated. No GPU OOM in this trace.
  Control95678957abd0ae56ba9a6f015b13d065b2242d05 and v2 artifacts remain immutable.
  A 4-process/16-compilation test passed on BOTH NFS and executable tmpfs; exact
  intermittent NFS trigger not reproduced. v3 changes cache location and scoped
  failure cleanup only; same TRAIN_COMMIT and four milestones. Read live progress.
- v3 controller1185643 launched01:19:02 Beijing, train1185912 started01:20:04;
  control commit02ab842bcb9bf0930837ddc1abe25c7d0ee7bfbf. Preflight passed and
  actual driver1185921 uses /dev/shm/opd-q4-c3/train for all temporary/compiler
  caches, also verified in GPU workers. By01:36:47 Beijing formal Steps1-3 completed;
  preclipgrad5.42415/1.30626/3.24619, cap0/32,4/32,0/32. All recorded scalars finite.
  First32 raw trajectories passed exact eval-input/seed21 data-order/mask/logprob
  audit:32 independent request seeds,8 distinct outputs per question,0 periodic
  tails,1 missing boxed,0 think tags. Step1 position NPZ exists; actual Hydra
  confirms all4 save milestones, no pruning, model/optimizer/extra checkpoint
  contents. No milestone checkpoint exists yet. See launch-v3 JSON and remote
  startup_acceptance.json. First real update crossed v2's failure boundary with
  tmpfs; this is not a long-run stability or benchmark gain claim. Local48 tests
  passed. Freshly check progress before reporting; training intentionally continues.
- Read `docs/results/2026-09-20-qwen4-completion-supervision-step11.md` for the
  subsequent603-second live watch (01:45-01:55 Beijing,11 samples,Step6->11).
  All352 Step1-11 raw trajectories and11 NPZ/scalars passed integrity/protocol
  checks. Caps11/352, strict periodic4/352, missing-boxed21/352, think0. All scalars
  finite, no nonzero numerical counters; no claim of zero repetition or gains.
  Watch finished; training continues unchanged. Figures and immutable CPU
  snapshot under run/supervision/snapshot_step_000011. No new GPU job launched.
  Final live check advanced to12: grad0.979,cap2/32,all scalars finite,no fatal
  log errors. Step12 is not included in the Step1-11 raw audit/figures.

## Qwen Completion Acceptance (2026-09-20)

- User asked to VERIFY the three pending Qwen points: original teacher suitability,
  exact boxed completion prompt on BOTH original models, and independently derived
  per-request seeds wired into actual training with GPU update/resume acceptance.
- Read `docs/plans/2026-09-20-qwen-completion-acceptance.md`. Old v1 remains stopped.
  No automatic formal200/full-eval launch. The bounded gate runs Block3 then Token,
  each original->Step1 save->resumeStep2; keep every checkpoint and rollout.
- Inference diagnostic:16 archived Qwen v1 training questions, n2 seeds21/22,
  historical/plain/candidate prompts, both models,192 outputs. Immutable inference
  code `analysis_deployments/048f0f8acfaa74e9679e41c1d1c4acfdedf8a471`;
  outputs `diagnostics/20260920_qwen_completion_acceptance`. Both jobs completed.
- Training code `0f9161f02f08287fb07f0375ad0a6bda81133ff0` is deployed separately.
  Controller `scripts/run_qwen_completion_gate.py` targets ONLY
  `runs/20260920v1_qwen4_completion_gate_seed21_ml2`, short cache `/limx_embap/tos/q4/c1`.
  Controller1128081 completed00:43:19 Beijing Sep20. Both arms passed actual
  Step1-save-exit-resume-Step2, all4 training processes exited0. Do not relaunch
  this gate or old v1 queue; no formal200/full-eval was started.
- New protocol `qwen3_completion_boxed_v1` bypasses ChatML, uses the shared boxed
  completion function for train/eval. New request seed rule uses stable global21,
  step/source index/question hash/sample index/turn, with per-request SamplingParams.
  Neither change alters historical Block3 joint ratio/reduction or Qwen EOS mask.
- Local164 related tests passed. Actual remote collector/eval input IDs matched
  all643 benchmark questions; this is a CPU input check, not formal evaluation.
  All192 inference outputs and pinned historical grading passed; candidate
  student4/32 correct,1/32 cap,0/32 missing boxed; teacher8/32 correct,3/32 cap,
  2/32 missing boxed. Only16 training questions; NOT benchmark scores, superiority
  significance, or proof of teacher reliability on arbitrary student prefixes.
  See `docs/results/2026-09-20-qwen-completion-acceptance.md`.
- Actual Block3 Step1:32 seeds,8 distinct outputs for EACH of4 questions,0 cap,
  no periodic tails,1 missing boxed; original seed21 data order and exact eval
  input IDs verified. Preclipgrad5.42485, entropy student0.20898/teacher0.17316,
  top16overlap0.91432, signflip0.16348. No nonfinite diagnostics. Probe only,
  not formal method validation. Block3 Step2 cap2/32, periodic1/32, grad1.25157;
  Token Step1 cap0/32, grad4.42339; Token Step2 cap4/32, periodic4/32, grad1.28878.
  All128 rollouts audited, each batch32 unique seeds and8 different outputs per
  question. Both arms/all4 ranks' Adam37states, scheduler, RNG and sampler
  directly inspected at BOTH steps: optimizer1->2, sampler4->8. No nonfinite
  diagnostics; residual repetition remains, not proof of long-run safety/gains.
- First pre-update rollout: same32 input IDs, sources, seed identities, sampling,
  output token IDs and masks across arms. Only28/32 logprob arrays bitwise equal;
  six positions differ, max0.0142758. No bitwise floating/replay guarantee.
  Read curated gate JSON beside the result MD. Standard and min-count1 position
  PNGs retained; low-count tails are NOT population trends. Old plot HTML has a
  hard-coded Block3 intro even for Token; use correctly titled PNGs, not that intro.
  DataLoader worker-exit warnings remain in shutdown logs; all jobs returned0
  and resume audits passed. Kernel logs were inaccessible; don't claim their
  root cause was proven. Fresh relevant local tests:101 passed.

## Approved Base Protocol Revision and Llama Diagnosis (2026-09-19)

- User accepted completion prompts WITHOUT ChatML for Base models, independent
  per-trajectory request seeds derived deterministically from global seed21,
  step/question/sample identity, and inference validation before formal training.
  New Token/Block3 runs must share the revised train/eval prompt and seed rule;
  start independently from original weights. Preserve the old run and checkpoints.
- This supersedes the historical prompt difference for the NEXT Base experiment,
  not for already completed or stopped attempts. Do not restart the stopped v1
  queue. The candidate boxed-answer completion prompt subsequently received
  student AND teacher GPU checks; see Sep20 section above for results/limits.
- User also requested checking whether old Llama shared the prompt problem.
  Llama is 1B-Instruct/3B-Instruct with native Llama chat, not Qwen ChatML or Base.
  Diagnose original models with archived questions and native EOS; do not strip
  native chat from Instruct models by assumption. No Llama retraining authorized
  by this diagnosis. Keep all inference outputs and existing training history.
- Llama inference diagnosis completed23:05 Beijing:16 archived training questions,
  n2 seeds21/22,4 prompt cells,2 original models,256 outputs; audit/grading passed.
  See `docs/results/2026-09-19-llama-native-prompt-diagnosis.md`. Student/teacher
  caps: historical1/32,2/32; remove think1/32,4/32; completion4/32,4/32;
  native eval1/32,0/32. Do NOT remove native chat from Llama Instruct as a fix.
  Original-model diagnostic correct counts are only0-1/32 student,1-3/32 teacher;
  these are fixed training questions, not benchmarks. Request-seed duplication is
  shared with old Qwen; Llama's later high-entropy Block3 failure is not established
  to have the same cause as Base/ChatML pre-update noise. No new training launched.
  Raw/logs/manifests are in `diagnostics/20260919_llama_prompt_check`; both jobs exited,
  all4 GPUs idle at final check. Recheck live state before using GPUs.

## Qwen4 User Stop and Initial-Loop Diagnosis (2026-09-19)

- Latest user explicitly STOPPED this pair to investigate pre-update repetition.
  At21:32 Beijing controller1047903 was SIGTERM'ed;21:33 all4 GPUs idle.
  Formal completed Step4,4 raw batches preserved; no formal checkpoint before50.
  Queue failed/error143 is this intentional stop. Do NOT resume or start queued
  eval/Token jobs. Keep all probes, logs, weights and original frozen runtime.
- See `docs/results/2026-09-19-qwen4-initial-loop-incident.md`.
  Inference-only interventions authorized by this diagnosis, separate remote root
  `diagnostics/20260919_qwen4_initial_loop`: original4B/1.7B, same first4 actual
  prompts, fixed vs independent request seeds, prompt/marker changes. Preserve
  raw outputs and code versions; these are NOT benchmark scores or training arms.
- All diagnostic jobs completed by22:08;22:09 all4 GPUs idle.664 saved vLLM
  outputs passed SHA/input/seed/logprob/completeness checks. Full16384 independent
  seeds21..28: historical ChatML cap4B8/32,1.7B8/32; remove think13/32,11/32;
  remove only3 ChatML markers4/32,2/32; plain+Solution2/32,1/32. These are the
  same4 training questions, NOT benchmarks, and do not eliminate all math loops.
- Base/ChatML mismatch is an experimentally supported trigger, not Block3's
  first update. HF CPU and GPU confirm first-token distribution shift when
  markers are removed; independent HF GPU continues both original loop tails
  with selected-token probability>0.9999. EOS was not emitted before truncation.
  Fixed per-request seed amplifies duplicates but is not the sole cause. Both
  models have near-identical special-token embedding rows; do not claim knowledge
  of their initialization/training history from this geometry alone. See full
  incident note and machine-readable summary for caveats and exact protocols.

## Authorized Qwen4 Block3-First Pair (2026-09-19)

- Latest user approved Qwen3-4B-Base student with the unchanged historical public
  Qwen3-4B-Base-GRPO teacher. Models from ModelScope; no private weights or train access.
- New order supersedes earlier Token-first defaults for THIS pair: Block3 formal
  training/full eval, original zero-step student eval, then Token formal training/full eval.
  Both start independently from the SAME original student, never from each other.
- Save and fully evaluate50/100/150/200 for BOTH arms, retaining all state and raw
  training/evaluation rollouts; four benchmarks separately, exact historical grader.
- Read `docs/plans/2026-09-19-qwen4-blockfirst.md`. Preserve historical1.7B prompt
  difference and original fixed Block3 joint ratio/reduction. No silent loss/mask/seed changes.
- Historical Llama v4 finished all evaluations at13:42 Beijing Sep19; fresh state
  and GPU-idle verification required before launching new queue. Do not restart old queues.
- New queue launched via server nohup at20:28 Beijing, PID1047903; immutable
  runtime `57ae7dfec10e206dd441a4c70571308c09f64a03`, run
  `20260919v1_qwen4_blockfirst_seed21_ml2`. ModelScope student download verified;
  paired preflight passed20:31. Read live state and startup note before action,
  never duplicate or replace the frozen runtime with later documentation commits.
  Block3 probe1 PID1049174 started20:31:58; Ray init passed20:32:36. Formal training
  was not yet started at this snapshot; probes must pass first.
- User requested supervision. Read `docs/results/2026-09-19-qwen4-training-supervision.md`.
  Probe1 completed update/save/audit; probe2 PID1066486 started20:50:19 Beijing.
  Four-rank optimizer/scheduler1, LR2e-6, full RNG, sampler4 inspected. Raw32
  archive passed frozen protocol/mask/logprob audit. First PRE-UPDATE generation
  already had repetition/multilingual noise and16/32 length stops (4 unique
  responses from4 prompts). Do not attribute this initial quality risk to Block3
  updates or call finite grad4.98 proof of healthy output. No parameters changed.
- Resume/raw/storage gates passed21:07; formal Block3 PID1079900 launched21:07:37
  from original Base, resume disabled. Step2 optimizer/scheduler2, sampler8 and
  all RNG states inspected. Probe2 grad1.19,24/32 truncated; quality risk persists.
  Ignored worker-killed exception occurred AFTER probe2 final metrics during
  teardown, exit0/audits passed; same signature in completed Llama arms. Preserve
  warning, do not call it a training CUDA OOM or hot-edit frozen runtime. At21:08
  formal was initializing; inspect live state before claiming actual progress.
- At21:21:31 formal Step1 completed, Step2 generation ongoing. Grad4.982413,
  scalar nonfinite checks clean; raw32 protocol audit passed. Actual first input/
  output IDs, mask, sampling/order match probe1; max logprob difference2.15e-5.
  Initial16/32 truncation/repetition persists; this is not a benchmark result.
  No scientific settings changed. Save50/100/150/200 remains mandatory.

## User-Directed Historical Re-evaluation (2026-09-18)

- Sep19 morning update supersedes the early Token-only snapshot below: Token200
  and all50/100/200 full evaluations passed. Block3 formal PID815640 completed200
  at11:01:44 Beijing, exit0; checkpoint/41diagnostics/6400raw audits passed. Four
  rank optimizer/scheduler200, LR2e-6, RNG and sampler800 inspected. Step200 full
  evaluation PID961434 started11:03:39;100/50 queued. Read live state and
  `docs/results/2026-09-19-llama-block3-quality-diagnosis.md`. Block3 raw output
  degenerates by at leastStep28; student entropy9.24atStep35, before maxpreclipgrad
  161.621atStep75. Finite gradients/lower truncation do NOT mean healthy output.
  All200steps actual prompt/order/sampling/protocol matched; firstrollouts equal.
  No runtime or parameters changed. Keep original queue and full milestone evals;
  do not silently repair scientific failure by changing one comparison arm.
  Current Block3 objective also uses joint PPO ratios and block reduction; it is
  NOT only advantage broadcasting with token-normalized PPO. CPU toy verifies3x
  logprob-input gradient for equal advantages (not3x Adam update). Preserve the
  historical objective; any normalization-only control is a new experiment.
- User explicitly requested active training supervision and repair on Sep19.
  Read `docs/results/2026-09-19-llama-historical17-supervision.md` and live v4 state.
  Both Token probes completed; four-rank model/optim/extra_state loading and
  advancement to Step2 confirmed, acceptance passed with no issues/warnings.
  Formal Token PID647058 launched00:10:44 Beijing with resume disabled from the
  original student. At01:19:04 formal Step52 completed. Step50 full four-rank
  optimizer/scheduler/RNG and dataloader states inspected; scheduler/optimizer50,
  LR2e-6, sampler200 prompts. First50 raw archives/1600 outputs passed SHA/mask/
  finite-logprob checks; 11 diagnostics and Step50 plots verified. Cumulative
  truncation272/1600=17%; repetitive/off-topic outputs exist even without truncation.
  Historical per-request seed yields identical8 responses per prompt (200 groups,
  200 unique within-group outputs), not1600 independent samples. Preserve paired
  conditions; do not silently change sampling mid-run or claim bit-exact vLLM
  resume. Actual restored Step2 prompts match uninterrupted Step2. Block3 not yet
  started. Read live state for progress; supervision is not method validation.
- Latest live audit: all six Qwen evaluations and the Llama initial evaluation/GPU
  prompt gates completed. v2 stopped at23:04:54 Beijing in token_opd_probe1 BEFORE
  training because its Ray AF_UNIX socket path exceeded107 bytes. No Llama training
  checkpoint/rollout was produced. Preserve the failed attempt; do not rerun six evals.
- Read `docs/results/2026-09-18-llama-historical17-recovery.md`. Reviewed recovery
  controller `recover_historical17_llama.py` targets a distinct v4 run, short cache
  `/limx_embap/tos/lh/r1`, and unchanged frozen94be7ea training/eval runtime. It reuses
  verified initial evaluation, then executes the original matched Token/Block3 plan.
  v3 bootstrap failed before manifest/Ray due to NFS exclusive flock on a read-only
  descriptor; logs retained, corrected with non-truncating writable locks in v4.
  Check live v4 state before any launch; do not duplicate it or auto-retry failures.
- v4 launched23:43:15 Beijing, controller623151, controller release
  `b00ad93385ef367cc88c499d3c89d9feaa515327`, frozen training/eval94be7ea.
  Runtime hashes passed and queue Ray gate started; a separate actual CPU Ray
  worker test already passed with102-byte socket budget. Do not replace runtime
  with later documentation commits. Formal training was not yet started at launch.
- Latest user explicitly stopped the active Llama queue and confirmed reproducing
  the OLD 1.7B train/eval instruction difference: training requests think tags,
  evaluation does not. This narrowly overrides the non-thinking policy for the
  new Llama attempt; it does not authorize modifying historical experiments.
- Old Llama run `20260918v1_llama32_1b_3b_nonthinking_seed21_ml2` was deliberately
  stopped at Step146, controller297837 SIGTERM, state error143. Full checkpoints
  50/100 and all146 rollout archives remain. Block3 never started. Do not resume
  this attempt or label the user stop a spontaneous failure.
- Read `docs/plans/2026-09-18-historical17-reeval-llama.md`. New controller:
  `scripts/run_historical17_reeval_llama.py`, run identity
  `20260918v2_historical17_reeval_llama_seed21_ml2`. Check live state before launch.
- First re-evaluate original Qwen3-1.7B Token and Block3 checkpoints50/100/200,
  six complete evaluations, exact old prompts/decoding/seeds and the historical
  external grader used for the accepted comparison. Add lossless eval archives
  without changing generation inputs/RNG. Preserve original results and weights.
- Only after six complete accepted evaluations: initial Llama1B-Instruct eval,
  Token resume gate/formal200/eval200/100/50, Block3 equivalents, paired report.
  Both start from original ModelScope student, not the interrupted/probe weights.
- Llama legacy protocol `llama32_historical17_v1` uses exact old math user
  instructions with native Llama chat template/date/EOS; never inject Qwen control
  tokens. Eval remains `llama32_nonthinking_v1`. Explicitly document this intended
  train/eval difference. Generated think tags are measured, not forbidden here.
- Preserve original data/seed21/hyperparameters/objectives/full n8 four-task
  grading, all training and evaluation rollouts, diagnostics/heatmaps, and complete
  checkpoints. New immutable deployment only; failure stops queue, no auto retry,
  extra seeds, private models, pruning, or access to train.
- New controller launched at 2026-09-18 12:32:57 Beijing, PID434226, runtime
  `94be7ea1d659309256c8356681925bb9710895c4`. Preflight passed; Token Step50
  evaluation PID435131 launched at12:39:43, all four workers loaded the correct
  historical weights. The six-checkpoint evaluation precedes all new Llama jobs. Read live
  state and `docs/results/2026-09-18-historical17-reeval-startup.md`; do not start
  a duplicate or replace its runtime with later documentation commits.

## Authorized Llama 3.2 Pair (2026-09-18)

- User selected Llama-3.2-1B-Instruct student / Llama-3.2-3B-Instruct teacher,
  obtained only through ModelScope. Read `docs/plans/2026-09-18-llama32-paired-validation.md`.
- New queue must wait for the current Sep17v2 Qwen06 Block3 FULL evaluation
  (Steps200/100/50) and protection checks to succeed, then all GPUs idle.
- Order: zero-step Instruct student evaluation, Token training/evaluation,
  original Block3 mean training/evaluation. Both formal arms initialize from the
  same pinned original student. Preserve data/seed21/hyperparameters/grader/full
  n8 four-task evaluation; native Llama template/EOS replaces Qwen-specific control.
- Keep full rollout archives, existing diagnostics/heatmaps, complete50/100/200
  checkpoints and per-benchmark results. Require actual GPU prompt/output and
  two-step four-rank resume gates. Do not claim success before these gates run.
- Only ml2; no train, private models, extra seed, automatic retry or pruning.
- Prepare a new immutable runtime; never hot-edit an existing deployment or
  interrupt the predecessor. Read live state before launching a duplicate queue.
- Waiting controller launched at 2026-09-18 02:32 Beijing, PID297837, runtime
  `f2d148e74c06283617a878b1a03fabdd5ef390da`, run
  `20260918v1_llama32_1b_3b_nonthinking_seed21_ml2`. State verified
  `waiting_predecessor`; no Llama GPU work has started. Read
  `docs/results/2026-09-18-llama32-preparation.md` and live state. Never duplicate
  or replace its runtime with later documentation commits.

## Authorized Block3 Full Evaluation (2026-09-17)

- User explicitly requested starting evaluation without further questions. This
  overrides the prior no-autostart boundary, not the completed queue's identity.
- New evaluation-only queue: `20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2`,
  controller `scripts/run_nonthinking_block3_eval.py`. Read live state before launch.
- Evaluate completed v1 Block3 checkpoints200/100/50, all four tasks, full n8,
  explicit non-thinking and exact historical grader. Use frozen evaluator0ce73aa,
  not a newly modified evaluator. No new training, private models, or extra seed.
- Token reference is the accepted v1 evaluations of Sep13v4 Token checkpoints;
  verify originals and reuse their scores, not rerun them. Report each benchmark's
  Avg@8/Pass@8 and matched-step Block3-minus-Token percentage-point differences.
- New merges/results only in v2. Protect all old checkpoints/results, no pruning,
  no automatic retry or restarting v1. Failure or incomplete output stops v2.
- Plan: `docs/plans/2026-09-17-qwen06-block3-full-eval.md`. Execution acceptance
  does not remove the documented Block3 output-quality warning.
- v2 is launched: PID252025, controller831c0af58f34ad90ac6776f1a5444622a5b82262.
  Preflight and Step200 merge/prompt checks passed; eval PID252473 entered
  four-GPU MATH500 generation. Steps100/50 remain queued. Read live state and
  `docs/results/2026-09-17-qwen06-block3-eval-startup.md`; do not launch a duplicate.

## Authorized Token Eval Then Block3 (2026-09-17)

- Latest user explicitly authorizes full evaluation of the completed non-thinking
  Qwen06 Token arm, followed by matched Block3 mean training. This overrides the
  historical eval/Block3 pause below, but NEVER restarts the old thinking queue.
- Read `docs/plans/2026-09-17-qwen06-token-eval-then-block3.md`. New ordered queue:
  `20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2`, controller
  `scripts/run_nonthinking_eval_block3.py`. Check live state before launching.
- Token source is the completed Sep13 v4 non-thinking run. Evaluate Steps200/100/50,
  official 0.6B Base and existing 4B GRPO teacher with pinned historical grader,
  full n8, all four benchmarks scored independently. No pooled headline score.
- Only after ALL five model evaluations pass raw/graded/coverage checks may the
  queue launch Block3's two-step resume probe and formal200. No weight warm-start
  from Token/probes. No new seed, sliding/random variant, SFT or private model.
- Block3 uses EXACT Token training runtime `0ce73aa42d7f734b9d34c379f474458bb6d48771`;
  controller has its own release. Keep explicit non-thinking, full rollout archives,
  original diagnostics and all50/100/200 complete checkpoints. Do not hot-edit
  either runtime or overwrite Token outputs. Merges/evals go to the NEW run root.
- Queue completed at 2026-09-17 20:10:49 Beijing: all five full evaluations,
  Block3 resume probe, formal200, full rollout/checkpoint audits and final figures.
  Protected inputs verified unchanged. Controller release remains
  `788a2dca21902fda25777fca1a6863780973a8c1`; later documentation commits must not
  replace either runtime. Read live state before any new launch; never restart it.
- Execution acceptance is NOT method validation: Block3 showed severe gibberish,
  off-topic output and repetition; full-run truncation63.75% vs Token40.625%.
  All6400 nominal training trajectories/41 diagnostics retained; actual prompt
  IDs and ordering match Token for every step. Full Block3 benchmarks NOT run.
- User requested continuous supervision. Accepted per-benchmark scores and
  monitoring evidence are in `docs/results/2026-09-17-qwen06-evaluation-progress.md`.
  Read live state first; phase snapshots are not proof that subsequent jobs ran.
- Any eval failure or paired mismatch stops the queue. No auto retry/cleanup.
  Block3 full evaluation is NOT auto-started by this controller.

## Authorized Non-Thinking Token Run (2026-09-13)

- User now authorizes a NEW official Qwen3-0.6B-Base Token OPD run on ml2, with
  explicit thinking disable, training/evaluation prompt agreement, lossless
  training trajectories and all existing OPD diagnostics. Read
  `docs/plans/2026-09-13-qwen06-nonthinking-token.md` before acting.
- The active attempt is `20260913v4_qwen06_nonthinking_token_seed21_ml2`;
  controller `scripts/run_qwen06_nonthinking.py`. It gates training on real GPU
  prompt/output checks and a two-step training-state/rollout-retention resume test.
- Attempt v2 stopped in the GPU gate's CPU input construction (missing batch on a
  test stub), before any generation or training. Its runtime 8a0a2d3 and artifacts
  remain intact. v3 uses a real DataProto fixture; do not restart v2.
- v3 / 7724141 passed prompt checks but stopped before generation because its
  standalone vLLM gate forked after CUDA initialization. v4 explicitly uses spawn
  for that gate; no old runtime or failed output is overwritten.
- Keep original seed21, data, teacher, 200 steps and Step50/100/200 state retention.
  Start formal Token from the official Base, never from old/probe weights. New
  protocol also matches evaluation EOS stops and uses actual generation lengths
  for masks. Do not silently attribute protocol changes to Block3.
- The OLD Qwen06 queue remains stopped. Do not launch Block3 or revive old eval.
  This standalone controller trains Token and builds diagnostic figures only;
  full benchmark evaluation is not automatically resumed by this authorization.
- Inspect live state before launching; never duplicate a running controller.
- v4 controller PID3844168 uses frozen runtime
  `0ce73aa42d7f734b9d34c379f474458bb6d48771`. Read
  `docs/results/2026-09-13-qwen06-nonthinking-startup.md` and live queue state.
  Later documentation commits must not replace this runtime in audit commands.
- GPU gate and two-step four-rank resume/rollout audits passed. Formal Token
  PID3867901 started at 18:06:07 Beijing, and its first update/archive/diagnostic
  figures are verified. Training is ongoing, not complete. Both the GPU gate and
  actual first training batch produced no new think tags, but length truncation
  persists (formal Step1: 16/32). Do not claim non-thinking solved truncation.

## Non-Thinking Training Policy (2026-09-13)

- User requires all subsequent training to disable thinking mode. This overrides
  historical prompt defaults; it does not authorize restarting the paused queue.
- Explicitly pass `enable_thinking=False` to supported chat templates, including
  the active rollout path (`data.apply_chat_template_kwargs` in Revisiting OPD).
  An empty kwargs mapping is not an explicit disable. Remove environment/prompt
  instructions requiring reasoning inside `<think>...</think>` as well.
- Inspect the actual rendered prompt and token IDs in preflight, not just the
  run card. Qwen's tokenizer may express disabled thinking with an EMPTY,
  already-closed think block in the assistant prefix. Do not delete that control
  prefix just to make a string search find no think tags. This is not a guarantee
  that a Base model can never generate such tags or ordinary reasoning text.
- Preserve historic deployments, configs and trajectories unchanged. The paused
  Qwen06 runtime is not yet amended for this policy or full rollout retention;
  it must not be resumed blindly. Use a new versioned protocol and run identity.
- Apply the same protocol to Token and Block3 comparison arms. Do not compare a
  newly non-thinking Block3 arm against the historical thinking-unspecified,
  think-instructed Token arm as a controlled method-only ablation.
- Historical audit: ml2 1.7B Token/Block3 and current 0.6B Token training used
  empty template kwargs plus an environment instruction requiring think tags;
  their evaluation path explicitly passes false. The recent `no_think` diagnostic
  only removed an instruction; it did NOT set `enable_thinking=False`.
  See `docs/results/2026-09-13-token-truncation-comparison.md`.

## Active User Pause (2026-09-13)

- User stopped the Qwen06 evaluation queue to diagnose 0.6B truncation against
  1.7B. Controller PID3547500 and Step50 eval PID3822051 were terminated at
  2026-09-13 14:04 Beijing. The queue records `failed/error=143` and eval exit -15
  because of this deliberate stop, not a spontaneous training failure.
- Do not restart full evaluation, dispatch Block3, or relaunch the old controller
  until the user redirects from this diagnosis. All training checkpoints remain.
- Small matched inference-only diagnostics are authorized. Preserve raw tokens,
  special-token text and finish reasons; keep them separate from historical
  training trajectories and benchmark scores. Never call regenerated responses
  the original training rollouts.
- The bounded diagnosis is complete: 192 raw responses, all eight generation
  jobs exited 0. Read `docs/results/2026-09-13-qwen06-truncation-diagnosis.md`.
  The four GPUs were verified idle afterwards. Recommendations in that note are
  not launched experiments; the evaluation/Block3 pause still applies.

## Training Rollout Retention Policy (2026-09-13)

- User requires future training runs to retain their actual on-policy rollout
  trajectories. Scalar metrics, position heatmaps, and evaluation predictions
  are not substitutes for training trajectories.
- Save every training step and every sampled response, not only checkpoint or
  diagnostic steps. Retain run/attempt ID, step, prompt/sample/group identity,
  unmodified prompt and response token IDs, response length/mask, and decoded
  text that preserves special tokens. Link records to the run's sampling seeds,
  generation settings, tokenizer identity, and source revision.
- Preserve the generation engine's actual finish/stop reason when available,
  together with EOS/stop configuration. A length-at-cap indicator is a separate
  measurement, not proof of the engine's finish reason. Never invent missing
  reasons or reconstruct original trajectories by sampling a checkpoint again.
- Make retention part of preflight and the resume gate: verify readable records,
  step/sample coverage, special-token preservation, and no overwrite on resume.
  Logging must not consume sampling RNG or change loss, batches, or decoding.
- Keep raw trajectories in the experiment artifact store, never in Git. Preserve
  prior attempts and use distinct files/segments on resume; no silent pruning.
- This is a requirement for subsequent new launches, not evidence that the
  existing frozen Qwen06 queue already saves trajectories. Its completed Token
  run has no raw rollout dump. Do not hot-edit that runtime; any logging amendment
  to an already queued job must be explicit, versioned, and verified first.
- Historical truncation comparison and evidence limits are recorded in
  `docs/results/2026-09-13-token-truncation-comparison.md`.

## Benchmark Reporting Policy (2026-09-13)

- User requires independent scores for MATH500, AIME24, AIME25, and AMC23.
  Report each benchmark's Avg@8, Pass@8 and Block3-minus-Token differences at
  Step50/100/200. Never combine benchmarks into a total score or substitute a
  macro/pooled score or overall verdict for these separate results.
- Existing evaluation already retains task-specific graded files and `per_task`
  comparisons. Preserve historical raw artifacts and aggregate fields for lineage,
  but do not use their aggregate scores as new user-facing results.
- Frozen queue/report scripts still produce historical macro summaries. Before
  delivering a new final report, render the per-task results separately instead
  of publishing that old macro-first report unchanged. Do not interrupt training
  or regrade unchanged predictions merely to change presentation.
- A macro bootstrap interval is not a per-benchmark interval. Any task-specific
  uncertainty analysis must be calculated using that task's paired examples.

## Active Window Experiment (2026-09-11)

- New user-approved follow-on (2026-09-13): read
  `docs/plans/2026-09-13-qwen06-paired-validation.md`. Only official Qwen3-0.6B-Base
  with the existing public 4B GRPO teacher; Token OPD before original Block3 mean,
  identical data/seed/full eval. User subsequently removed the 48-hour time cap on
  2026-09-13; keep both runs at 200 steps with full eval. No 4B student
  or extra seeds. Wait for the current window queue to complete; never preempt it.
  Read `docs/results/2026-09-13-qwen06-pair-startup.md` before any action.
  Its queue is already running as PID3547500; do not launch a duplicate.
  On 2026-09-13 04:21 Beijing, the predecessor completed. Token resume gate
  passed, and formal Token training started at 04:47 as PID3594635, from Base
  with resume disabled. Check live state before acting; Block3 remains queued.
  Read the startup note's night-supervision section, including probe teardown
  warnings and the distinction between probe and formal diagnostics.
  Qwen06 runtime is frozen at
  `ec0a7a950540d7f4753c08b18bc26c544f6a7cde` (no time cap). The earlier waiting
  controller PID3544990 was retired before any GPU job; its evidence is archived.
  Later docs commits do not replace the runtime.

- Read `docs/plans/2026-09-11-sliding-window-opd-validation-v2-seed21.md`
  and `docs/results/2026-09-11-window-seed21-startup.md` before acting.
- This approved experiment uses only `ml2`; never connect to `train`.
- Only `random3` and `sliding3`, training seed21, 200 steps each. No automatic
  additional seeds, token/fixed baselines, or model pairs.
- Its explicit model/data/eval protocol supersedes the historical clean-room
  defaults below: student Qwen3-1.7B-Base, teacher Qwen3-4B-Base-GRPO, the recorded DAPO pool,
  all four math tasks with n=8 at Step50/100/200. Do not substitute other assets.
- Runtime is frozen at `fbad852a638de18e20d571a061b2be0437942a38`; pass that value
  as `SOURCE_COMMIT` to window control commands after documentation-only commits.
- Inspect live queue state before any launch. Do not duplicate a queued run or
  present resume-gate measurements as formal training/evaluation results.
- For the reviewed JSONL failure, read `docs/results/2026-09-12-window-eval-recovery.md`.
  Recovery uses a separate committed analysis release; never overwrite the frozen
  training runtime, original failed job, or completed raw/graded predictions.

## Hard Rules

- Do not use user-trained or private checkpoints unless the user explicitly
  requests that exact model in the current task.
- Default student for current clean-room experiments: official `Qwen/Qwen3-0.6B`.
- Default teacher for current clean-room experiments: official `Qwen/Qwen3-4B`.
- Never commit checkpoints, model weights, raw datasets, processed datasets,
  full prediction JSONL files, logs, cache directories, SSH keys, tokens, or
  `.env` files.
- Do not put access keys, tokens, private keys, or passwords into committed
  files. Internal run paths may appear in result notes/configs when needed for
  lineage.
- Every real experiment should have a config, command, data manifest, run
  summary, and result note.
- Full eval means GSM8K 1319 examples and MATH500 500 examples unless a result
  note explicitly says it is a pilot or partial eval.

## Repository Map

- `scripts/clean_opd_train.py`: current OPD trainer and loss implementation.
- `scripts/eval_math_batched.py`: batched math evaluation.
- `scripts/regrade_predictions.py`: post-hoc regrading.
- `scripts/launch_*`: launch wrappers for remote experiments.
- `scripts/summarize_*`: compact experiment summaries.
- `tests/test_block_supervision.py`: unit tests for token/block supervision.
- `reports/`: human-readable HTML reports.
- `docs/results/`: concise experiment conclusions.
- `results/*.json`: small curated summaries only.

## Before Running Expensive Training

1. Confirm the exact model paths are official models unless told otherwise.
2. Confirm data file and manifest identity.
3. Run or inspect a resume gate.
4. Launch with `nohup` or another detached server-side mechanism.
5. Save complete training state: model, optimizer, scheduler, global step,
   sampler or consumed offset, RNG, args, and data identity.
6. Preserve checkpoints unless the user explicitly approves deletion.

## Before Reporting Completion

Run fresh verification. At minimum:

```bash
python -m py_compile scripts/clean_opd_train.py
python -m pytest tests/test_block_supervision.py -q
git status --short
```

If the task involved a remote run, also verify the run summary, output counts,
and that no training/eval process is left running unintentionally.

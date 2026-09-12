#!/usr/bin/env python3
"""Fixed-scope ml2 queue: public Qwen06 Token OPD, then original Block3 mean."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_paired_opd_evals as paired
import prepare_qwen06_assets as assets
import run_window_queue as jobs

ROOT = assets.ROOT
RUN_ROOT = ROOT / "runs/20260913v1_qwen06_pair_seed21_ml2"
PREDECESSOR = ROOT / "runs/20260911v2r1_sliding_window_seed21_ml2"
HISTORICAL = ROOT / "runs/20260712v1_token_opd_replication_seed21_ml2/token_opd"
VENV = Path("/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl")
PYTHON = VENV / "bin/python"
PLOT_PYTHON = ROOT / "envs/window_plot_20260911/bin/python"
GRADER = HISTORICAL / "grading/historical_utils_sha04f7.py"
DATA = ROOT / "data/math_opd_dapo17k_hf_full_eval4"
CACHE = Path("/limx_embap/tos/q06/0913v1")
VARIANTS = ("token_opd", "block3_mean")
STEPS = (50, 100, 200)
BUDGET_SECONDS = None  # User removed the time cap on 2026-09-13; the two-run scope is unchanged.


def read_json(path):
    return json.loads(path.read_text())


def predecessor_ready(root):
    state = read_json(root / "queue_state.json")
    if state.get("status") == "running":
        return False
    if state.get("status") != "complete":
        raise ValueError("predecessor is not running or complete; manual review required")
    proof = root / "recoveries" / state.get("recovery_id", "missing") / "protected_inputs_verified.json"
    exits = [root / v / f"eval_step_{s}_n8/exit_code.txt" for v in ("random3", "sliding3") for s in STEPS]
    if (not proof.is_file() or read_json(proof).get("passed") is not True
            or not (root / "paired_comparison.json").is_file()
            or any(not p.is_file() or p.read_text().strip() != "0" for p in exits)):
        raise ValueError("predecessor completion evidence is incomplete")
    return True


def validate_prepared_pair(cards):
    if set(cards) != set(VARIANTS):
        raise ValueError("Only the approved Token/Block3 pair is allowed")
    allowed = {"variant", "opd_block_size", "opd_block_advantage_mode", "experiment_name", "diagnostic_output_dir"}
    left, right = (cards[v] for v in VARIANTS)
    for key in (set(left) | set(right)) - allowed:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f"Unapproved paired difference: {key}")
    for variant, size, mode in (("token_opd", 1, "sum"), ("block3_mean", 3, "mean")):
        expected = {**paired.EXPECTED_TRAINING_VALUES, "variant": variant, "opd_block_size": size,
                    "opd_block_advantage_mode": mode, "opd_window_mode": "fixed", "ppo_epochs": 1,
                    "student_model": str(assets.STUDENT), "student_model_revision": assets.REVISION,
                    "teacher_model": str(assets.TEACHER)}
        for key, value in expected.items():
            if cards[variant].get(key) != value:
                raise ValueError(f"{variant}: {key} must equal {value!r}")


def launcher_env(runtime, commit, root, variant, probe_step=None):
    if variant not in VARIANTS or probe_step not in (None, 1, 2):
        raise ValueError("Unapproved variant/probe")
    values = {
        "LAUNCH_TRANSPORT": "local", "PREPARE_ONLY": "true", "REMOTE": "ml2", "REMOTE_ROOT": runtime,
        "RUN_ROOT": root, "VARIANT": variant, "VENV": VENV, "SOURCE_COMMIT": commit,
        "SUBMODULE_BASE_COMMIT": "f32f284f25bae5b16d2d44ee336b52851dccc736",
        "PROJECT_NAME": "opd_qwen06_pair", "EXP_NAME": f"qwen06-{variant}",
        "STUDENT_MODEL": assets.STUDENT, "STUDENT_MODEL_REVISION": assets.REVISION,
        "MATH_TEACHER": assets.TEACHER, "TEACHER_MODEL_REVISION": "",
        "HF_HOME_DIR": VENV.parents[1] / "hf_home", "LOCAL_CACHE_ROOT": CACHE / "train",
        "BASELINE_ALIGNMENT": "Qwen06 public Base capacity transfer; same teacher/data/protocol as seed21 Qwen1.7",
        "DATA_DIR": DATA, "TRAIN_DATA": DATA / "train.parquet", "VAL_DATA": DATA / "test.parquet",
        "N_GPUS_PER_NODE": 4, "RAY_NUM_CPUS": 64, "ENV_SEED": 21, "OPD_WINDOW_SEED": 910021,
        "TRAIN_BATCH_SIZE": 4, "PPO_MINI_BATCH_SIZE": 32, "ROLLOUT_GROUP_SIZE": 8,
        "MAX_PROMPT_LENGTH": 2048, "MAX_RESPONSE_LENGTH": 16384, "LEARNING_RATE": "2e-6",
        "TOTAL_TRAINING_STEPS": 2 if probe_step else 200, "SAVE_FREQ": -1, "TEST_FREQ": -1,
        "VAL_N": 1, "TRAINER_LOGGER": "['console']", "ROLLOUT_GPU_MEMORY_UTILIZATION": 0.6,
        "ROLLOUT_MAX_NUM_BATCHED_TOKENS": 18432, "ROLLOUT_TEMPERATURE": 1.0, "ROLLOUT_TOP_P": 0.9,
        "ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU": 1, "ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": 4,
        "REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": 1, "OPD_DIAGNOSTICS": "true",
        "OPD_DIAG_OUTPUT_DIR": root / variant / "diagnostics",
        "OPD_DIAG_INTERVAL": 1 if probe_step else 5, "OPD_DIAG_TOPK": 16, "OPD_DIAG_POSITION_BIN": 128,
        "OPD_DIAG_POSITION_STRIDE": 1, "OPD_DIAG_SIGN_EPS": "1e-4",
        "DIAGNOSTIC_SAVE_STEPS": {None: "50,100,200", 1: "1", 2: "1,2"}[probe_step],
        "STOP_AFTER_STEP": 1 if probe_step == 1 else -1, "FILTER_OVERLONG_PROMPTS": "false",
        "EXPECTED_TRAIN_SHA256": jobs.TRAIN_SHA, "RESUME_MODE": "resume_path" if probe_step == 2 else "disable",
        "RESUME_FROM_PATH": root / variant / "checkpoints/global_step_1" if probe_step == 2 else "",
    }
    return {k: str(v) for k, v in values.items()}


def prepare(root, runtime, commit, variant, runner, probe_step=None):
    run_root = root / "probes" if probe_step else root
    env = launcher_env(runtime, commit, run_root, variant, probe_step)
    runner(["bash", runtime / "scripts/launch_revisiting_block_opd_formal_train.sh"],
           root / "queue_jobs" / f"prepare_{variant}_{probe_step or 'formal'}", job_env=env)


def check_pair_artifacts(root):
    cards = {v: read_json(root / v / "run_card.json") for v in VARIANTS}
    validate_prepared_pair(cards)
    for variant, card in cards.items():
        if card["diagnostic_output_dir"] != str(root / variant / "diagnostics"):
            raise ValueError("Unapproved diagnostics destination")
    entries = [paired.load_sha256_manifest(root / v / "artifact_hashes.sha256") for v in VARIANTS]
    if set(entries[0]) != set(entries[1]):
        raise ValueError("Paired model/data bytes differ")
    jobs.write_json(root / "paired_preflight.json", {"passed": True, "cards": cards,
                    "allowed_differences": ["method", "experiment_name", "diagnostic_output_dir"]})


def audit_command(runtime, run, commit, full=False, probe_step=None):
    cmd = [PYTHON, runtime / "scripts/audit_block10_run.py", "--run-dir", run, "--variant", run.name,
           "--checkpoint-steps", {None: "50,100,200", 1: "1", 2: "1,2"}[probe_step],
           "--expected-source-commit", commit, "--expected-train-sha256", jobs.TRAIN_SHA,
           "--expected-eval-data-dir", DATA / "eval_jsonl",
           "--expected-student-model-suffix", "Qwen3-0.6B-Base",
           "--expected-teacher-model-suffix", "Qwen3-4B-Base-GRPO",
           "--expected-student-model-revision", assets.REVISION]
    cmd += ["--eval-steps", "50,100,200"] if full else ["--skip-eval"]
    if probe_step:
        cmd += ["--expected-total-training-steps", "2", "--expected-diag-interval", "1",
                "--expected-diagnostic-steps", "1" if probe_step == 1 else "1,2",
                "--expected-resume-mode", "disable" if probe_step == 1 else "resume_path",
                "--expected-resume-from-path", "" if probe_step == 1 else run / "checkpoints/global_step_1"]
    return cmd


def validate_resume_state(extra, data, step):
    if not {"cpu", "cuda", "numpy", "random"}.issubset(extra.get("rng", {})):
        raise ValueError("Incomplete rank RNG state")
    if extra.get("lr_scheduler", {}).get("last_epoch") != step:
        raise ValueError("Incorrect scheduler step")
    snapshot = data.get("_snapshot", {})
    consumed = (snapshot.get("_snapshot_step", 0) + data.get("_steps_since_snapshot", 0)
                if snapshot else data.get("_num_yielded"))
    if consumed != step:
        raise ValueError("Missing or incorrect dataloader consumed offset")


def audit_resume(run):
    import torch

    log = (run / "logs/nohup.log").read_text()
    checkpoint1 = run / "checkpoints/global_step_1"
    if f"Resuming from {checkpoint1}" not in log or "No dataloader state found" in log:
        raise ValueError("Training state resume was not confirmed")
    evidence = []
    for step in (1, 2):
        checkpoint = run / f"checkpoints/global_step_{step}"
        data = torch.load(checkpoint / "data.pt", map_location="cpu", weights_only=False)
        for rank in range(4):
            extra = torch.load(checkpoint / "actor" / f"extra_state_world_size_4_rank_{rank}.pt",
                               map_location="cpu", weights_only=False)
            validate_resume_state(extra, data, step)
            if f"[rank-{rank}]: Loading from {checkpoint1}/actor/model_world_size_4_rank_{rank}.pt" not in log:
                raise ValueError(f"Missing rank {rank} model/optimizer/extra-state load evidence")
            for prefix in ("model", "optim", "extra_state"):
                path = checkpoint / "actor" / f"{prefix}_world_size_4_rank_{rank}.pt"
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Incomplete training state: {path}")
            evidence.append({"step": step, "rank": rank, "scheduler_step": extra["lr_scheduler"]["last_epoch"]})
    jobs.write_json(run / "resume_gate.json", {"passed": True, "states": evidence,
                    "limitation": "Checks saved rank RNG and restored training state, not bitwise vLLM replay"})


def execute_pair(root, runtime, commit, runner):
    for variant in VARIANTS:
        prepare(root, runtime, commit, variant, runner)
    check_pair_artifacts(root)
    for variant in VARIANTS:
        run, probe = root / variant, root / "probes" / variant
        for probe_step in (1, 2):
            prepare(root, runtime, commit, variant, runner, probe_step)
            job = root / "queue_jobs" / f"{variant}_probe{probe_step}"
            runner(["bash", probe / "command.sh"], job, gpu=True)
            (probe / "logs").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(job / "logs/job.log", probe / "logs/nohup.log")
            runner(audit_command(runtime, probe, commit, probe_step=probe_step),
                   root / "queue_jobs" / f"{variant}_probe{probe_step}_audit")
        runner([PYTHON, runtime / "scripts/run_qwen06_pair.py", "--audit-resume", probe],
               root / "queue_jobs" / f"{variant}_resume_gate")
        runner(["bash", run / "command.sh"], run, pid_name="train.pid", log_name="nohup.log", gpu=True)
        runner(audit_command(runtime, run, commit), run / "queue_jobs/checkpoints")
        runner([PLOT_PYTHON, runtime / "scripts/analyze_single_opd_diagnostics.py", "--run-dir", run,
                "--output-dir", run / "figures", "--label", f"Qwen06 {variant} seed21"], run / "queue_jobs/figures")
        for step in STEPS:
            checkpoint = run / f"checkpoints/global_step_{step}"
            actor, model = checkpoint / "actor", checkpoint / "actor/huggingface"
            runner([PYTHON, runtime / "external/revisiting_opd/scripts/model_merger.py", "merge",
                    "--backend", "fsdp", "--local_dir", actor, "--target_dir", model], run / f"queue_jobs/merge_{step}")
            evaluation = run / f"eval_step_{step}_n8"
            evaluation.mkdir(parents=True)
            card = {"variant": variant, "step": step, "checkpoint_dir": str(checkpoint),
                    "actor_dir": str(actor), "model_dir": str(model), "eval_data_dir": str(DATA / "eval_jsonl"),
                    "n": 8, "temperature": 1.0, "top_p": 0.9, "max_tokens": 16384, "eval_seed": 21,
                    "grader": "verl", "enable_thinking": False, "gpus": "0,1,2,3",
                    "tasks": " ".join(jobs.TASKS), "source_commit": commit}
            jobs.write_json(evaluation / "eval_card.json", card)
            # Hash generation is a job so its output is audited and charged to the budget.
            runner(["sha256sum", *[DATA / "eval_jsonl" / f"{t}.jsonl" for t in jobs.TASKS]],
                   run / f"queue_jobs/eval_hashes_{step}")
            hashlog = run / f"queue_jobs/eval_hashes_{step}/logs/job.log"
            if hashlog.is_file():
                shutil.copyfile(hashlog, evaluation / "eval_data_hashes.sha256")
            runner(jobs.eval_command(runtime, PYTHON, model, DATA / "eval_jsonl", evaluation / "outputs", assets.STUDENT),
                   evaluation, pid_name="eval.pid", log_name="eval.log", gpu=True)
        runner(audit_command(runtime, run, commit, full=True), run / "queue_jobs/full_audit")
        runner([PYTHON, runtime / "scripts/regrade_opd_eval_external.py", "--run-dir", run,
                "--grader-source", GRADER, "--steps", "50,100,200"], run / "queue_jobs/regrade")
    for view, name in (("historical_external_grader", "paired_comparison"), ("outputs", "paired_builtin")):
        runner([PYTHON, runtime / "scripts/compare_paired_opd_evals.py", "--left-run", root / VARIANTS[0],
                "--right-run", root / VARIANTS[1], "--pair-kind", "token-block3", "--view", view,
                "--bootstrap-seed", "20260913", "--bootstrap-replicates", "10000",
                "--output-json", root / f"{name}.json"], root / "queue_jobs" / name)
    runner([PYTHON, runtime / "scripts/build_paired_validation_report.py",
            "--external-json", root / "paired_comparison.json", "--builtin-json", root / "paired_builtin.json",
            "--output-html", root / "report.html", "--token-diagnostics", root / "token_opd/figures",
            "--block-diagnostics", root / "block3_mean/figures", "--title", "Qwen3-0.6B Token OPD vs Block3 mean"],
           root / "queue_jobs/report")


def preflight_assets():
    manifest = read_json(assets.STUDENT / "asset_manifest.json")
    if manifest.get("revision") != assets.REVISION or manifest.get("tokenizer_alignment") is not True:
        raise ValueError("Student asset preparation incomplete")
    protected = {str(assets.STUDENT / name): value for name, value in manifest["sha256"].items()}
    historical = paired.load_sha256_manifest(HISTORICAL / "artifact_hashes.sha256")
    teacher_files = [(sha, path) for sha, path in historical if path.parent == assets.TEACHER]
    data_files = [(sha, path) for sha, path in historical if DATA in path.parents]
    if not any(path.suffix == ".safetensors" for _, path in teacher_files) or len(data_files) != 6:
        raise ValueError("Historical teacher/data identities are missing")
    protected.update({str(path): sha for sha, path in teacher_files + data_files})
    protected[str(GRADER)] = paired.HISTORICAL_GRADER_SHA256
    for name, sha in protected.items():
        if assets.sha256(Path(name)) != sha:
            raise ValueError(f"Changed protected model/data/grader: {name}")
    assets.tokenizer_alignment(assets.STUDENT, assets.TEACHER, assets.REFERENCE)
    if shutil.disk_usage(ROOT).free < 100_000_000_000:
        raise ValueError("Less than 100 GB free; do not prune historical checkpoints")
    return protected


def phase_deadline(start):
    return None if BUDGET_SECONDS is None else start + BUDGET_SECONDS


def require_budget(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Machine-time budget exhausted; cannot declare completion")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-resume", type=Path)
    parser.add_argument("--preflight-assets", action="store_true")
    parser.add_argument("--verify-protected", action="store_true")
    args = parser.parse_args()
    if args.audit_resume:
        audit_resume(args.audit_resume)
        return
    if args.preflight_assets:
        jobs.write_json(RUN_ROOT / "protected_inputs.json", preflight_assets())
        return
    if args.verify_protected:
        for name, sha in read_json(RUN_ROOT / "protected_inputs.json").items():
            if assets.sha256(Path(name)) != sha:
                raise ValueError(f"Protected input changed: {name}")
        return
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime / "DEPLOYED_COMMIT").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or runtime != ROOT / "deployments" / commit:
        raise ValueError("Only an immutable approved ml2 runtime may launch this queue")
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state_path = RUN_ROOT / "queue_state.json"
    with (RUN_ROOT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / "queue_manifest.json").exists():
            raise ValueError("Existing phase requires manual review; no automatic retry or new budget")
        (RUN_ROOT / "queue.pid").write_text(str(os.getpid()) + "\n")
        try:
            jobs.write_json(state_path, {"status": "waiting_for_predecessor", "predecessor": str(PREDECESSOR),
                            "order": VARIANTS, "phase_started": False, "budget_seconds": BUDGET_SECONDS,
                            "updated_at": jobs.now()})
            while not predecessor_ready(PREDECESSOR):
                time.sleep(60)
            with (PREDECESSOR / "queue.lock").open("a") as previous_lock:
                fcntl.flock(previous_lock, fcntl.LOCK_EX)
                if not predecessor_ready(PREDECESSOR):
                    raise ValueError("predecessor changed after lock acquisition")
                start = time.time()
                deadline = phase_deadline(start)
                jobs.write_json(RUN_ROOT / "queue_manifest.json", {
                    "source_commit": commit, "runtime": str(runtime), "started_at": jobs.now(),
                    "start_epoch": start, "deadline_epoch": deadline, "budget_seconds": BUDGET_SECONDS,
                    "variants": VARIANTS, "student_revision": assets.REVISION, "training_seed": 21,
                    "eval_steps": STEPS, "cleanup_reserve_seconds": 60, "predecessor": str(PREDECESSOR)})
                env = dict(os.environ, PATH=f"{VENV / 'bin'}:{os.environ.get('PATH', '')}",
                           PYTHONPATH=f"{runtime}:{runtime / 'external/revisiting_opd'}", CUDA_VISIBLE_DEVICES="0,1,2,3",
                           PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                           RAY_DEDUP_LOGS="0", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", ENGINE="vllm")
                for variable, suffix in {"TMPDIR": "tmp", "VLLM_CACHE_ROOT": "vllm", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                                         "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "OUTLINES_CACHE_DIR": "outlines"}.items():
                    (CACHE / suffix).mkdir(parents=True, exist_ok=True)
                    env[variable] = str(CACHE / suffix)
                def runner(argv, job, job_env=None, **kwargs):
                    jobs.run_job(argv, job, runtime, state_path, {**env, **(job_env or {})},
                                 deadline_epoch=deadline, **kwargs)
                runner(["sha256sum", "-c", ".expected.sha256"], RUN_ROOT / "queue_jobs/runtime_verification")
                runner([PYTHON, runtime / "scripts/run_qwen06_pair.py", "--preflight-assets"],
                       RUN_ROOT / "queue_jobs/asset_verification")
                execute_pair(RUN_ROOT, runtime, commit, runner)
                runner([PYTHON, runtime / "scripts/run_qwen06_pair.py", "--verify-protected"],
                       RUN_ROOT / "queue_jobs/final_input_verification")
                require_budget(deadline)
                jobs.write_json(state_path, {"status": "complete", "updated_at": jobs.now(), "variants": VARIANTS,
                                "elapsed_seconds": time.time() - start, "protected_inputs_verified": True})
        except BaseException as error:
            previous = read_json(state_path) if state_path.exists() else {}
            status = "budget_exhausted" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "failed"
            jobs.write_json(state_path, {**previous, "status": status, "error": str(error), "updated_at": jobs.now()})
            raise


if __name__ == "__main__":
    main()

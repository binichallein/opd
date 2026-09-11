#!/usr/bin/env python3
"""Run the two approved seed21 experiments serially on ml2, failing closed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

VARIANTS = ("random3", "sliding3")
STEPS = (50, 100, 200)
TASKS = ("math500", "aime24", "aime25", "amc23")
TRAINING_SEED = 21
TRAIN_SHA = "cf359f257a320aecb6448e824b7cc34f70e694583be3df7177b14f359b7959cf"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def completed_or_new(job):
    started, exit_file = job / "started_at.txt", job / "exit_code.txt"
    if exit_file.is_file():
        if exit_file.read_text().strip() == "0":
            return True
        raise RuntimeError(f"Failed job requires manual review, not automatic retry: {job}")
    if started.exists():
        raise RuntimeError(f"Interrupted/active job requires manual review: {job}")
    return False


def validate_card(card, variant, commit):
    expected = {
        "variant": variant, "seed": TRAINING_SEED, "total_training_steps": 200,
        "opd_window_mode": {"random3": "random", "sliding3": "sliding"}[variant],
        "opd_window_seed": 910021, "source_commit": commit,
    }
    for key, value in expected.items():
        if card.get(key) != value:
            raise ValueError(f"{variant}: {key} expected {value!r}, got {card.get(key)!r}")


def eval_command(runtime, python, model, data, output, student):
    return list(map(str, [
        python, runtime / "scripts/eval_qwen3_math_vllm.py", "--model-path", model,
        "--eval-jsonl-dir", data, "--output-dir", output, "--tasks", *TASKS,
        "--n", 8, "--temperature", 1.0, "--top-p", 0.9, "--max-tokens", 16384,
        "--gpus", "0,1,2,3", "--eval-seed", 21, "--grader", "verl",
        "--length-tokenizer-path", student,
    ]))


def wait_for_idle(timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits",
        ], check=True, capture_output=True, text=True)
        rows = [[int(v.strip()) for v in row.split(",")] for row in result.stdout.splitlines()]
        if len(rows) != 4:
            raise RuntimeError("This queue requires exactly four GPUs")
        if all(memory < 1024 and utilization == 0 for memory, utilization in rows):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("GPUs are not idle; no competing process will be terminated")
        time.sleep(5)


def run_job(argv, job, runtime, state_path, env, *, pid_name="job.pid", log_name="job.log", gpu=False):
    job.mkdir(parents=True, exist_ok=True)
    if completed_or_new(job):
        return
    if gpu:
        wait_for_idle()
    logs = job / "logs"
    logs.mkdir(exist_ok=True)
    (job / "queued_command.txt").write_text(shlex.join(list(map(str, argv))) + "\n")
    (job / "started_at.txt").write_text(now() + "\n")
    with (logs / log_name).open("ab") as output:
        process = subprocess.Popen(list(map(str, argv)), cwd=runtime, env=env,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        (job / pid_name).write_text(str(process.pid) + "\n")
        write_json(state_path, {"status": "running", "job": str(job), "pid": process.pid, "updated_at": now()})
        print(f"{now()} started {job} pid={process.pid}", flush=True)
        try:
            code = process.wait()
        except BaseException:
            # Only signal the process group created by this queue, never a global Ray/GPU kill.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
            except ProcessLookupError:
                code = process.wait()
            (job / "exit_code.txt").write_text(str(code or 130) + "\n")
            (job / "finished_at.txt").write_text(now() + "\n")
            raise
    (job / "exit_code.txt").write_text(str(code) + "\n")
    (job / "finished_at.txt").write_text(now() + "\n")
    if code:
        raise RuntimeError(f"Job exited {code}: {job}; queue stopped, artifacts retained")


def audit_command(runtime, python, run, commit, data, *, full):
    command = [python, str(runtime / "scripts/audit_block10_run.py"),
               "--run-dir", str(run), "--variant", run.name, "--checkpoint-steps", "50,100,200",
               "--expected-source-commit", commit, "--expected-train-sha256", TRAIN_SHA,
               "--expected-eval-data-dir", str(data),
               "--expected-student-model-suffix", "Qwen3-1.7B-Base",
               "--expected-teacher-model-suffix", "Qwen3-4B-Base-GRPO"]
    return command + (["--eval-steps", "50,100,200"] if full else ["--skip-eval"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--grader-source", type=Path, required=True)
    parser.add_argument("--plot-python", type=Path, required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    runtime = Path(__file__).resolve().parents[1]
    root = args.run_root.resolve(strict=True)
    if not str(root).startswith("/limx_embap/tos/user/Yaleon/"):
        raise ValueError("Only the approved ml2 asset root is supported")
    commit = (runtime / "DEPLOYED_COMMIT").read_text().strip()
    state_path = root / "queue_state.json"
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "queue.pid").write_text(str(os.getpid()) + "\n")
        try:
            with (root / "runtime_verification.log").open("a") as log:
                subprocess.run(["sha256sum", "-c", ".expected.sha256"], cwd=runtime, stdout=log, check=True)
            cards = {v: json.loads((root / v / "run_card.json").read_text()) for v in VARIANTS}
            for variant, card in cards.items():
                validate_card(card, variant, commit)
            card = cards[VARIANTS[0]]
            python = str(Path(card["venv"]) / "bin/python")
            data = Path(card["val_data"]).parent / "eval_jsonl"
            env = dict(os.environ)
            env.update(PATH=f"{Path(python).parent}:{env.get('PATH', '')}",
                       PYTHONPATH=f"{runtime}:{runtime / 'external/revisiting_opd'}:{env.get('PYTHONPATH', '')}",
                       CUDA_VISIBLE_DEVICES="0,1,2,3", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            cache = Path("/limx_embap/tos/swv/0911v2_eval")
            for variable, suffix in {"TMPDIR": "tmp", "VLLM_CACHE_ROOT": "vllm", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                                     "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "OUTLINES_CACHE_DIR": "outlines"}.items():
                (cache / suffix).mkdir(parents=True, exist_ok=True)
                env[variable] = str(cache / suffix)
            runs = [root / variant for variant in VARIANTS]
            for run in runs:
                run_job(["bash", run / "command.sh"], run, runtime, state_path, env,
                        pid_name="train.pid", log_name="nohup.log", gpu=True)
                run_job(audit_command(runtime, python, run, commit, data, full=False),
                        run / "queue_jobs/checkpoints", runtime, state_path, env)
                acceptance = run / "checkpoint_acceptance.json"
                if not acceptance.exists():
                    acceptance.write_bytes((run / "acceptance.json").read_bytes())
                run_job([python, runtime / "scripts/audit_window_run.py", "--run-dir", run,
                         "--variant", run.name, "--checkpoint-steps", "50,100,200"],
                        run / "queue_jobs/window_audit", runtime, state_path, env)
                run_job([args.plot_python, runtime / "scripts/analyze_single_opd_diagnostics.py", "--run-dir", run,
                         "--output-dir", run / "figures", "--label", f"ml2 {run.name} seed21"],
                        run / "queue_jobs/figures", runtime, state_path, env)
            run_job([args.plot_python, runtime / "scripts/plot_window_diagnostics.py", "--run-root", root,
                     "--output-dir", root / "paired_heatmaps"],
                    root / "queue_jobs/paired_heatmaps", runtime, state_path, env)
            for run in runs:
                for step in STEPS:
                    checkpoint = run / f"checkpoints/global_step_{step}"
                    actor = checkpoint / "actor"
                    model = actor / "huggingface"
                    run_job([python, runtime / "external/revisiting_opd/scripts/model_merger.py", "merge",
                             "--backend", "fsdp", "--local_dir", actor, "--target_dir", model],
                            run / f"queue_jobs/merge_{step}", runtime, state_path, env)
                    evaluation = run / f"eval_step_{step}_n8"
                    evaluation.mkdir(exist_ok=True)
                    eval_card = {
                        "variant": run.name, "step": step, "checkpoint_dir": str(checkpoint),
                        "actor_dir": str(actor), "model_dir": str(model), "eval_data_dir": str(data),
                        "n": 8, "temperature": 1.0, "top_p": 0.9, "max_tokens": 16384,
                        "eval_seed": 21, "grader": "verl", "enable_thinking": False,
                        "gpus": "0,1,2,3", "tasks": " ".join(TASKS), "source_commit": commit,
                    }
                    if not (evaluation / "eval_card.json").exists():
                        write_json(evaluation / "eval_card.json", eval_card)
                    hashes = [f"{hashlib.sha256((data / (task + '.jsonl')).read_bytes()).hexdigest()}  {data / (task + '.jsonl')}\n" for task in TASKS]
                    (evaluation / "eval_data_hashes.sha256").write_text("".join(hashes))
                    run_job(eval_command(runtime, python, model, data, evaluation / "outputs", Path(card["student_model"])),
                            evaluation, runtime, state_path, env, pid_name="eval.pid", log_name="eval.log", gpu=True)
                run_job(audit_command(runtime, python, run, commit, data, full=True),
                        run / "queue_jobs/full_audit", runtime, state_path, env)
                run_job([python, runtime / "scripts/regrade_opd_eval_external.py", "--run-dir", run,
                         "--grader-source", args.grader_source, "--steps", "50,100,200"],
                        run / "queue_jobs/regrade", runtime, state_path, env)
            run_job([python, runtime / "scripts/compare_paired_opd_evals.py", "--left-run", runs[0],
                     "--right-run", runs[1], "--left-label", "random3", "--right-label", "sliding3",
                     "--pair-kind", "random-sliding", "--bootstrap-seed", "20260911",
                     "--output-json", root / "paired_comparison.json"],
                    root / "queue_jobs/comparison", runtime, state_path, env)
            write_json(state_path, {"status": "complete", "variants": VARIANTS, "seed": 21, "updated_at": now()})
        except BaseException as error:
            previous = json.loads(state_path.read_text()) if state_path.exists() else {}
            write_json(state_path, {**previous, "status": "failed", "error": str(error), "updated_at": now()})
            raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reviewed, post-training-only recovery of the September window eval queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_window_queue as queue
from scripts.regrade_opd_eval_external import HISTORICAL_GRADER_SHA256, validate_grader_hash

TRAINING_COMMIT = "fbad852a638de18e20d571a061b2be0437942a38"
ASSET_ROOT = Path("/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_recovery(root, training, analysis, recovery_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", recovery_id):
        raise ValueError("Invalid recovery identifier")
    if (training / "DEPLOYED_COMMIT").read_text().strip() != TRAINING_COMMIT:
        raise ValueError("Unexpected training runtime; weights and rollout code must stay frozen")
    analysis_commit = (analysis / "ANALYSIS_COMMIT").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", analysis_commit):
        raise ValueError("Invalid analysis release identity")
    state = json.loads((root / "queue_state.json").read_text())
    failed_job = root / "random3/queue_jobs/full_audit"
    if state.get("status") != "failed" or state.get("job") != str(failed_job):
        raise ValueError("Recovery applies only to the reviewed Random3 full-audit failure")
    if (failed_job / "exit_code.txt").read_text().strip() != "1":
        raise ValueError("Missing original failed audit evidence")
    for variant in queue.VARIANTS:
        run = root / variant
        queue.validate_card(json.loads((run / "run_card.json").read_text()), variant, TRAINING_COMMIT)
        if (run / "exit_code.txt").read_text().strip() != "0":
            raise ValueError(f"{variant}: training must already be complete; recovery never trains")
        for name in ("checkpoint_acceptance.json", "window_acceptance.json"):
            record = json.loads((run / name).read_text())
            if record.get("passed") is not True or record.get("issues"):
                raise ValueError(f"{variant}: training audit failed: {name}")
            if name.startswith("checkpoint") and set(record.get("checkpoint_steps", [])) != set(queue.STEPS):
                raise ValueError(f"{variant}: missing training checkpoints")
            if name.startswith("window") and record.get("last_step") != 200:
                raise ValueError(f"{variant}: incomplete training ledger")
    for step in queue.STEPS:
        old_eval = root / "random3" / f"eval_step_{step}_n8"
        if (old_eval / "exit_code.txt").read_text().strip() != "0":
            raise ValueError("Random3 generation must already be complete")
        if (root / "sliding3" / f"eval_step_{step}_n8").exists():
            raise ValueError("Sliding3 eval already exists; review rather than overwrite or retry")
    recovery = root / "recoveries" / recovery_id
    recovery.mkdir(parents=True, exist_ok=False)
    for name in ("queue_state.json", "queue.pid", "queue.log"):
        if (root / name).is_file():
            shutil.copyfile(root / name, recovery / name)
    protected = {}
    for variant in queue.VARIANTS:
        run = root / variant
        if (run / "acceptance.json").is_file():
            shutil.copyfile(run / "acceptance.json", recovery / f"{variant}_prior_acceptance.json")
        files = [*run.glob("*.json"), *run.glob("*.sha256"), *run.glob("command.sh"),
                 *run.glob("diagnostics/*"), *run.glob("eval_step_*_n8/outputs/*"),
                 *run.glob("eval_step_*_n8/*.json"), *run.glob("eval_step_*_n8/*.sha256")]
        for path in files:
            if path.is_file() and path.name != "acceptance.json":
                protected[str(path)] = digest(path)
    record = {"training_commit": TRAINING_COMMIT, "analysis_commit": analysis_commit,
              "training_runtime": str(training), "analysis_runtime": str(analysis),
              "grader_sha256": HISTORICAL_GRADER_SHA256, "created_at": queue.now(),
              "recovery_id": recovery_id, "failed_job": str(failed_job),
              "reason": "JSONL splitlines incorrectly split Unicode separators inside valid strings",
              "protected_inputs": protected}
    queue.write_json(recovery / "manifest.json", record)
    return record


def verify_protected(record):
    for name, expected in record["protected_inputs"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed protected experiment evidence: {name}")


def execute_recovery(root, training, analysis, recovery, python, data, grader, env):
    state_path = root / "queue_state.json"
    for variant in queue.VARIANTS:
        run = root / variant
        card = json.loads((run / "run_card.json").read_text())
        jobs = recovery / variant
        queue.run_job(queue.audit_command(analysis, python, run, TRAINING_COMMIT, data, full=False),
                      jobs / "checkpoints", training, state_path, env)
        if variant == "sliding3":
            for step in queue.STEPS:
                actor = run / f"checkpoints/global_step_{step}/actor"
                model = actor / "huggingface"
                queue.run_job([python, training / "external/revisiting_opd/scripts/model_merger.py", "merge",
                               "--backend", "fsdp", "--local_dir", actor, "--target_dir", model],
                              jobs / f"merge_{step}", training, state_path, env)
                evaluation = run / f"eval_step_{step}_n8"
                evaluation.mkdir()
                eval_card = {"variant": variant, "step": step, "checkpoint_dir": str(actor.parent),
                             "actor_dir": str(actor), "model_dir": str(model), "eval_data_dir": str(data),
                             "n": 8, "temperature": 1.0, "top_p": 0.9, "max_tokens": 16384,
                             "eval_seed": 21, "grader": "verl", "enable_thinking": False,
                             "gpus": "0,1,2,3", "tasks": " ".join(queue.TASKS),
                             "source_commit": TRAINING_COMMIT}
                queue.write_json(evaluation / "eval_card.json", eval_card)
                (evaluation / "eval_data_hashes.sha256").write_text("".join(
                    f"{digest(data / (task + '.jsonl'))}  {data / (task + '.jsonl')}\n" for task in queue.TASKS))
                queue.run_job(queue.eval_command(training, python, model, data, evaluation / "outputs",
                                                Path(card["student_model"])),
                              evaluation, training, state_path, env, pid_name="eval.pid", log_name="eval.log", gpu=True)
        queue.run_job(queue.audit_command(analysis, python, run, TRAINING_COMMIT, data, full=True),
                      jobs / "full_audit", training, state_path, env)
        queue.run_job([python, analysis / "scripts/regrade_opd_eval_external.py", "--run-dir", run,
                       "--grader-source", grader, "--steps", "50,100,200"],
                      jobs / "regrade", training, state_path, env)
    queue.run_job([python, analysis / "scripts/compare_paired_opd_evals.py",
                   "--left-run", root / "random3", "--right-run", root / "sliding3",
                   "--left-label", "random3", "--right-label", "sliding3", "--pair-kind", "random-sliding",
                   "--bootstrap-seed", "20260911", "--bootstrap-replicates", "10000",
                   "--output-json", root / "paired_comparison.json"],
                  recovery / "comparison", training, state_path, env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-runtime", type=Path, required=True)
    parser.add_argument("--grader-source", type=Path, required=True)
    parser.add_argument("--recovery-id", required=True)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    training = args.training_runtime.resolve(strict=True)
    analysis = Path(__file__).resolve().parents[1]
    if root != ASSET_ROOT / "runs/20260911v2r1_sliding_window_seed21_ml2":
        raise ValueError("Only the approved ml2 window run is supported")
    if training != ASSET_ROOT / "deployments" / TRAINING_COMMIT:
        raise ValueError("Unexpected frozen runtime path")
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_grader_hash(args.grader_source, HISTORICAL_GRADER_SHA256)
        for release in (training, analysis):
            subprocess.run(["sha256sum", "--quiet", "-c", ".expected.sha256"], cwd=release, check=True)
        record = prepare_recovery(root, training, analysis, args.recovery_id)
        recovery = root / "recoveries" / args.recovery_id
        (root / "queue.pid").write_text(str(os.getpid()) + "\n")
        try:
            card = json.loads((root / "random3/run_card.json").read_text())
            python = str(Path(card["venv"]) / "bin/python")
            data = Path(card["val_data"]).parent / "eval_jsonl"
            env = dict(os.environ)
            env.update(PATH=f"{Path(python).parent}:{env.get('PATH', '')}",
                       PYTHONPATH=f"{training}:{training / 'external/revisiting_opd'}",
                       CUDA_VISIBLE_DEVICES="0,1,2,3", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
            cache = Path("/limx_embap/tos/swv/0911v2_eval")
            for variable, suffix in {"TMPDIR": "tmp", "VLLM_CACHE_ROOT": "vllm", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                                     "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda", "OUTLINES_CACHE_DIR": "outlines"}.items():
                (cache / suffix).mkdir(parents=True, exist_ok=True)
                env[variable] = str(cache / suffix)
            execute_recovery(root, training, analysis, recovery, python, data, args.grader_source, env)
            verify_protected(record)
            queue.write_json(recovery / "protected_inputs_verified.json", {"passed": True, "updated_at": queue.now()})
            queue.write_json(root / "queue_state.json", {"status": "complete", "variants": queue.VARIANTS,
                             "seed": 21, "recovery_id": args.recovery_id, "updated_at": queue.now(),
                             "analysis_commit": record["analysis_commit"], "training_commit": TRAINING_COMMIT})
        except BaseException as error:
            previous = json.loads((root / "queue_state.json").read_text())
            queue.write_json(root / "queue_state.json", {**previous, "status": "failed", "error": str(error),
                             "recovery_id": args.recovery_id, "updated_at": queue.now()})
            raise


if __name__ == "__main__":
    main()

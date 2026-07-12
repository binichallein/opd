#!/usr/bin/env python3
"""Regrade immutable OPD eval rollouts with the pinned historical grader."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable


TASK_COUNTS = {"math500": 500, "aime24": 30, "aime25": 30, "amc23": 83}
DEFAULT_STEPS = (50, 100, 200)
HISTORICAL_GRADER_SHA256 = (
    "04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--grader-source", type=Path, required=True)
    parser.add_argument(
        "--steps",
        default=",".join(str(step) for step in DEFAULT_STEPS),
    )
    return parser.parse_args()


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("steps must be positive")
    return steps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing raw output: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_grader_hash(path: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256.lower()
    ):
        raise ValueError("expected grader SHA must be a hexadecimal SHA-256")
    if not path.is_file():
        raise ValueError(f"historical grader is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(
            f"historical grader SHA mismatch: expected {expected_sha256}, got {actual}"
        )


def validate_eval_config(config: dict[str, Any], step: int) -> None:
    expected = {
        "tasks": list(TASK_COUNTS),
        "n": 8,
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 16384,
        "eval_seed": 21,
        "rollout_seeds": list(range(21, 29)),
        "grader": "verl",
        "enable_thinking": False,
    }
    for key, value in expected.items():
        observed = config.get(key)
        if key == "tasks":
            matches = set(observed or []) == set(value)
        else:
            matches = observed == value
        if not matches:
            raise ValueError(
                f"step {step} eval config {key} expected {value!r}, got {observed!r}"
            )


def raw_output_path(output_dir: Path, task: str, config: dict[str, Any]) -> Path:
    return output_dir / (
        f"{task}_t{float(config['temperature']):.1f}_"
        f"p{float(config['top_p']):.1f}_n{int(config['n'])}-"
        f"MNT{int(config['max_tokens'])}.jsonl"
    )


def validate_task_rows(
    rows: list[dict[str, Any]], task: str, expected_examples: int, step: int
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    keys: set[tuple[str, int, int]] = set()
    for row in rows:
        if str(row.get("task")) != task:
            raise ValueError(f"step {step} {task}: row task mismatch")
        example_id = str(row["example_id"])
        rollout_id = int(row["rollout_id"])
        seed = int(row["seed"])
        if seed != 21 + rollout_id:
            raise ValueError(f"step {step} {task}: rollout seed/id mismatch")
        key = (example_id, rollout_id, seed)
        if key in keys:
            raise ValueError(f"step {step} {task}: duplicate rollout key {key}")
        keys.add(key)
        groups[example_id].append(row)
    if len(groups) != expected_examples:
        raise ValueError(
            f"step {step} {task}: expected {expected_examples} examples, got {len(groups)}"
        )
    for example_id, group in groups.items():
        if (
            len(group) != 8
            or {int(row["rollout_id"]) for row in group} != set(range(8))
            or {int(row["seed"]) for row in group} != set(range(21, 29))
        ):
            raise ValueError(
                f"step {step} {task}/{example_id}: expected exactly 8 rollouts "
                "with rollout_id 0-7 and seeds 21-28"
            )
    return groups


def validate_step_inputs(
    run_dir: Path, step: int
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[Path, ...],
    dict[str, list[dict[str, Any]]],
]:
    output_dir = run_dir / f"eval_step_{step}_n8" / "outputs"
    config_path = output_dir / "eval_config.json"
    summary_path = output_dir / "summary.json"
    config = load_json(config_path)
    validate_eval_config(config, step)
    primary = load_json(summary_path)
    if set(primary.get("tasks", {})) != set(TASK_COUNTS):
        raise ValueError(f"step {step}: primary summary task set is incomplete")

    raw_paths: dict[str, Path] = {}
    primary_paths: dict[str, Path] = {}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for task, expected_examples in TASK_COUNTS.items():
        path = raw_output_path(output_dir, task, config)
        rows = load_jsonl(path)
        groups = validate_task_rows(rows, task, expected_examples, step)
        primary_task = primary["tasks"][task]
        if int(primary_task.get("num_examples", -1)) != len(groups):
            raise ValueError(f"step {step} {task}: primary example count mismatch")
        if int(primary_task.get("total_rollouts", -1)) != len(rows):
            raise ValueError(f"step {step} {task}: primary rollout count mismatch")
        primary_path = output_dir / f"{task}_graded.jsonl"
        primary_rows = load_jsonl(primary_path)
        primary_by_key = {}
        for primary_row in primary_rows:
            if type(primary_row.get("correct")) is not bool:
                raise ValueError(
                    f"step {step} {task}: primary correct must be a JSON boolean"
                )
            key = (
                str(primary_row.get("example_id")),
                int(primary_row.get("rollout_id", -1)),
                int(primary_row.get("seed", -1)),
            )
            if key in primary_by_key:
                raise ValueError(f"step {step} {task}: duplicate primary rollout key {key}")
            primary_by_key[key] = primary_row
        raw_by_key = {
            (
                str(row["example_id"]),
                int(row["rollout_id"]),
                int(row["seed"]),
            ): row
            for row in rows
        }
        if set(primary_by_key) != set(raw_by_key):
            raise ValueError(
                f"step {step} {task}: raw output differs from primary graded evidence"
            )
        for key, row in raw_by_key.items():
            primary_without_score = {
                name: value
                for name, value in primary_by_key[key].items()
                if name != "correct"
            }
            if primary_without_score != row:
                raise ValueError(
                    f"step {step} {task}: raw output differs from primary graded evidence"
                )
        raw_paths[task] = path.resolve(strict=True)
        primary_paths[task] = primary_path.resolve(strict=True)
        rows_by_task[task] = rows
    if sum(len(rows) for rows in rows_by_task.values()) != 5144:
        raise ValueError(f"step {step}: expected exactly 5,144 rollouts")
    evidence_paths = (
        *(raw_paths[task] for task in TASK_COUNTS),
        *(primary_paths[task] for task in TASK_COUNTS),
        config_path.resolve(strict=True),
        summary_path.resolve(strict=True),
        (run_dir / "acceptance.json").resolve(strict=True),
    )
    return config, primary, evidence_paths, rows_by_task


def load_grader(path: Path) -> Callable[[str, str], Any]:
    spec = importlib.util.spec_from_file_location(
        f"historical_opd_grader_{sha256_file(path)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import historical grader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grade_answer = getattr(module, "grade_answer_verl", None)
    if not callable(grade_answer):
        raise ImportError(f"historical grader has no grade_answer_verl: {path}")
    return grade_answer


def install_pinned_grader(
    run_dir: Path, source: Path, expected_sha256: str
) -> Path:
    target = run_dir / "grading" / "historical_utils_sha04f7.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_grader_hash(target, expected_sha256)
        return target.resolve(strict=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".historical_utils_", suffix=".py", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        validate_grader_hash(temporary, expected_sha256)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    validate_grader_hash(target, expected_sha256)
    return target.resolve(strict=True)


def _copy_raw_invariants(primary_task: dict[str, Any]) -> dict[str, Any]:
    invariants = {}
    for key in (
        "format_error_rollouts",
        "avg_response_length_tokens",
        "avg_response_length_chars",
    ):
        if key in primary_task:
            invariants[key] = primary_task[key]
    return invariants


def write_external_view(
    destination: Path,
    step: int,
    config: dict[str, Any],
    primary: dict[str, Any],
    evidence_paths: tuple[Path, ...],
    rows_by_task: dict[str, list[dict[str, Any]]],
    grader_path: Path,
    grade_answer: Callable[[str, str], Any],
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"external grader evidence already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".historical_external_grader_", dir=destination.parent)
    )
    try:
        summary: dict[str, Any] = {
            "n_expected": 8,
            "grader": "external",
            "eval_seed": 21,
            "rollout_seeds": list(range(21, 29)),
            "enable_thinking": False,
            "tasks": {},
        }
        for task in TASK_COUNTS:
            groups: dict[str, list[bool]] = defaultdict(list)
            graded_path = temporary / f"{task}_graded.jsonl"
            with graded_path.open("w", encoding="utf-8") as stream:
                for row in rows_by_task[task]:
                    correct = bool(
                        grade_answer(str(row["response"]), str(row["answer"]))
                    )
                    groups[str(row["example_id"])].append(correct)
                    graded = dict(row)
                    graded["correct"] = correct
                    stream.write(json.dumps(graded, ensure_ascii=False) + "\n")
            averages = [sum(scores) / 8 for scores in groups.values()]
            passes = [float(any(scores)) for scores in groups.values()]
            task_summary = {
                "num_examples": len(groups),
                "total_rollouts": len(rows_by_task[task]),
                "avg_at_n": sum(averages) / len(averages),
                "pass_at_n": sum(passes) / len(passes),
                "solve_none": sum(value == 0 for value in averages),
                "solve_all": sum(value == 1 for value in averages),
                **_copy_raw_invariants(primary["tasks"][task]),
                "graded_jsonl": str(destination / f"{task}_graded.jsonl"),
            }
            summary["tasks"][task] = task_summary
        summary["macro_avg_at_n"] = sum(
            summary["tasks"][task]["avg_at_n"] for task in TASK_COUNTS
        ) / len(TASK_COUNTS)
        summary["macro_pass_at_n"] = sum(
            summary["tasks"][task]["pass_at_n"] for task in TASK_COUNTS
        ) / len(TASK_COUNTS)
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_names = [f"{task}_graded.jsonl" for task in TASK_COUNTS]
        output_names.append("summary.json")
        (temporary / "output_hashes.sha256").write_text(
            "\n".join(
                f"{sha256_file(temporary / name)}  {destination / name}"
                for name in output_names
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_lines = [f"{sha256_file(path)}  {path}" for path in evidence_paths]
        manifest_lines.append(f"{sha256_file(grader_path)}  {grader_path}")
        (temporary / "input_hashes.sha256").write_text(
            "\n".join(manifest_lines) + "\n", encoding="utf-8"
        )
        temporary.rename(destination)
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def regrade_run(
    run_dir: Path,
    grader_source: Path,
    *,
    steps: tuple[int, ...] = DEFAULT_STEPS,
    expected_grader_sha256: str = HISTORICAL_GRADER_SHA256,
) -> dict[int, dict[str, Any]]:
    run_dir = run_dir.resolve(strict=True)
    grader_source = grader_source.resolve(strict=True)
    validate_grader_hash(grader_source, expected_grader_sha256)
    acceptance = load_json(run_dir / "acceptance.json")
    if acceptance.get("passed") is not True or acceptance.get("issues"):
        raise ValueError("final acceptance must pass before external regrading")
    accepted_checkpoints = {int(step) for step in acceptance.get("checkpoint_steps", [])}
    accepted_evals = {int(step) for step in acceptance.get("eval_steps", [])}
    if not set(steps).issubset(accepted_checkpoints & accepted_evals):
        raise ValueError("final acceptance does not cover every requested regrade step")

    validated = {}
    for step in steps:
        destination = run_dir / f"eval_step_{step}_n8" / "historical_external_grader"
        if destination.exists():
            raise FileExistsError(f"external grader evidence already exists: {destination}")
        validated[step] = validate_step_inputs(run_dir, step)

    grader_path = install_pinned_grader(
        run_dir, grader_source, expected_grader_sha256
    )
    grade_answer = load_grader(grader_path)
    results = {}
    for step in steps:
        config, primary, evidence_paths, rows_by_task = validated[step]
        destination = run_dir / f"eval_step_{step}_n8" / "historical_external_grader"
        results[step] = write_external_view(
            destination,
            step,
            config,
            primary,
            evidence_paths,
            rows_by_task,
            grader_path,
            grade_answer,
        )
    return results


def main() -> None:
    args = parse_args()
    results = regrade_run(
        args.run_dir,
        args.grader_source,
        steps=parse_steps(args.steps),
    )
    print(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the curated ML2 Block3 replication result, figures, and HTML report."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import sys
from typing import Any


STEPS = (50, 100, 200)
TASKS = ("math500", "aime24", "aime25", "amc23")
DIAGNOSTIC_STEPS = (1, *range(5, 201, 5))
HISTORICAL_GRADER_SHA256 = (
    "04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f"
)
HISTORICAL_TRAIN_RUNTIME = {
    "reference_logprob_micro_batch_per_gpu": 4,
    "vllm_gpu_memory_utilization": 0.7,
    "source": "scripts/run_revisiting_sampled_block_opd_math.sh defaults",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--historical-results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--label", default="ml2 A100 Block3 mean")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_sha256_manifest(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"invalid SHA-256 manifest line in {path}: {line!r}")
        digest, recorded_path = fields
        if any(character not in "0123456789abcdef" for character in digest.lower()):
            raise ValueError(f"invalid SHA-256 digest in {path}: {digest!r}")
        entries.append((digest.lower(), recorded_path.strip()))
    return entries


def validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"expected valid {label}")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"expected valid {label}")
    return normalized


def parse_hardware(env_path: Path) -> str:
    models = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3 and fields[0].isdigit() and fields[1].startswith("NVIDIA "):
            models.append(fields[1])
    if not models:
        raise ValueError("env.txt contains no GPU inventory")
    if len(set(models)) != 1:
        raise ValueError(f"mixed GPU inventory is unsupported: {models}")
    return f"{len(models)}x {models[0]}"


def validate_acceptance(
    run_dir: Path,
    run_card: dict[str, Any],
    acceptance: dict[str, Any],
) -> int:
    issues = acceptance.get("issues", [])
    if acceptance.get("passed") is not True or issues:
        raise ValueError(f"acceptance audit failed: {issues}")
    if acceptance.get("source_commit") != run_card.get("source_commit"):
        raise ValueError("acceptance source commit does not match run card")
    expected = {
        "checkpoint_steps": list(STEPS),
        "diagnostic_steps": list(DIAGNOSTIC_STEPS),
        "eval_steps": list(STEPS),
    }
    for key, expected_steps in expected.items():
        if acceptance.get(key) != expected_steps:
            raise ValueError(f"acceptance {key} is incomplete")
    exit_code = int((run_dir / "exit_code.txt").read_text(encoding="utf-8").strip())
    if exit_code != 0:
        raise ValueError(f"training exit code is {exit_code}, expected 0")
    return exit_code


def validate_run_contract(
    run_card: dict[str, Any], data_manifest: dict[str, Any]
) -> None:
    if run_card.get("variant") != "block3_mean":
        raise ValueError(
            f"expected block3_mean variant, got {run_card.get('variant')!r}"
        )
    if int(run_card.get("total_training_steps", -1)) != 200:
        raise ValueError("Block3 replication report requires exactly 200 training steps")
    train_sha = validate_sha256(
        data_manifest.get("sha256", {}).get("train_parquet"),
        "data-manifest training data SHA",
    )
    expected_train_sha = validate_sha256(
        run_card.get("expected_train_sha256"), "run-card training data SHA"
    )
    if train_sha != expected_train_sha:
        raise ValueError("training data SHA differs between run card and data manifest")
    if int(data_manifest.get("counts", {}).get("train", -1)) <= 0:
        raise ValueError("training row count is missing from data manifest")


def validate_eval_config(
    config: dict[str, Any], run_card: dict[str, Any], *, step: int
) -> None:
    n = int(config.get("n", -1))
    expected_seed = int(run_card["seed"])
    expected_seeds = list(range(expected_seed, expected_seed + n))
    checks = (
        (set(config.get("tasks", [])) == set(TASKS), "task set"),
        (n == 8, "n=8"),
        (math.isclose(float(config.get("temperature", -1)), 1.0), "temperature=1"),
        (math.isclose(float(config.get("top_p", -1)), 0.9), "top_p=0.9"),
        (
            int(config.get("max_tokens", -1))
            == int(run_card["max_response_length"]),
            "max_tokens",
        ),
        (int(config.get("eval_seed", -1)) == expected_seed, "eval seed"),
        (config.get("rollout_seeds") == expected_seeds, "rollout seeds"),
        (config.get("grader") == "verl", "primary grader"),
        (config.get("enable_thinking") is False, "thinking=false"),
    )
    failed = [label for passed, label in checks if not passed]
    if failed:
        raise ValueError(f"step {step} eval config mismatch: {', '.join(failed)}")


def validate_eval_summary(
    summary: dict[str, Any],
    config: dict[str, Any],
    *,
    step: int,
    label: str,
    expected_grader: str,
) -> None:
    n = int(config["n"])
    if int(summary.get("n_expected", -1)) != n:
        raise ValueError(f"{label} step {step} must be n=8")
    if summary.get("grader") != expected_grader:
        raise ValueError(f"{label} step {step} grader identity is invalid")
    for key in ("eval_seed", "rollout_seeds", "enable_thinking"):
        if summary.get(key) != config.get(key):
            raise ValueError(f"{label} step {step} {key} differs from eval config")
    tasks = summary.get("tasks", {})
    if set(tasks) != set(TASKS):
        raise ValueError(f"{label} step {step} task set is incomplete")
    if sum(int(task["total_rollouts"]) for task in tasks.values()) != 5144:
        raise ValueError(f"{label} step {step} does not contain 5,144 rollouts")
    for task, values in tasks.items():
        if int(values["total_rollouts"]) != int(values["num_examples"]) * n:
            raise ValueError(f"{label} step {step} {task} rollout count is invalid")
    for metric in ("macro_avg_at_n", "macro_pass_at_n"):
        task_metric = "avg_at_n" if "avg" in metric else "pass_at_n"
        recomputed = sum(float(tasks[task][task_metric]) for task in TASKS) / len(TASKS)
        if not math.isclose(float(summary[metric]), recomputed, abs_tol=1e-12):
            raise ValueError(f"{label} step {step} {metric} is inconsistent")


def validate_external_regrade(
    run_dir: Path,
    primary: dict[int, dict[str, Any]],
    external: dict[int, dict[str, Any]],
    configs: dict[int, dict[str, Any]],
) -> None:
    for step in STEPS:
        validate_eval_summary(
            external[step],
            configs[step],
            step=step,
            label="external grader",
            expected_grader="external",
        )
        hash_manifest = (
            run_dir
            / f"eval_step_{step}_n8"
            / "historical_external_grader"
            / "input_hashes.sha256"
        )
        entries = load_sha256_manifest(hash_manifest)
        expected_grader_suffix = "/grading/historical_utils_sha04f7.py"
        if not any(
            digest == HISTORICAL_GRADER_SHA256
            and recorded_path.endswith(expected_grader_suffix)
            for digest, recorded_path in entries
        ):
            recorded_hashes = {digest for digest, _ in entries}
            if HISTORICAL_GRADER_SHA256 not in recorded_hashes:
                raise ValueError(f"historical grader SHA is unverified at step {step}")
            raise ValueError(f"historical grader SHA/path is invalid at step {step}")
        config = configs[step]
        expected_output_suffixes = {
            (
                f"/eval_step_{step}_n8/outputs/{task}_"
                f"t{float(config['temperature']):.1f}_"
                f"p{float(config['top_p']):.1f}_n{int(config['n'])}-"
                f"MNT{int(config['max_tokens'])}.jsonl"
            )
            for task in TASKS
        }
        recorded_output_suffixes = {
            suffix
            for suffix in expected_output_suffixes
            if any(path.endswith(suffix) for _, path in entries)
        }
        if recorded_output_suffixes != expected_output_suffixes or len(entries) != 5:
            raise ValueError(f"external grader input hash manifest is invalid at step {step}")
        for task in TASKS:
            current = primary[step]["tasks"][task]
            regraded = external[step]["tasks"][task]
            for key in (
                "num_examples",
                "total_rollouts",
                "format_error_rollouts",
                "avg_response_length_tokens",
            ):
                if not math.isclose(float(current[key]), float(regraded[key]), abs_tol=1e-12):
                    raise ValueError(
                        f"external grader step {step} {task} changed raw-output field {key}"
                    )


def validate_historical_contract(
    run_card: dict[str, Any],
    data_manifest: dict[str, Any],
    configs: dict[int, dict[str, Any]],
    legacy: dict[str, Any],
) -> None:
    training = legacy["training"]
    current_training = {
        "student": Path(run_card["student_model"]).name,
        "teacher": Path(run_card["teacher_model"]).name,
        "train_sha256": data_manifest["sha256"]["train_parquet"],
        "seed": int(run_card["seed"]),
        "steps": int(run_card["total_training_steps"]),
        "train_batch_size": int(run_card["train_batch_size"]),
        "rollouts_per_prompt": int(run_card["rollout_group_size"]),
        "max_response_tokens": int(run_card["max_response_length"]),
    }
    for key, value in current_training.items():
        if training.get(key) != value:
            raise ValueError(f"historical training contract differs at {key}")
    if not math.isclose(
        float(training["learning_rate"]), float(run_card["learning_rate"]), abs_tol=0.0
    ):
        raise ValueError("historical training contract differs at learning_rate")
    historical_eval_sha = validate_sha256(
        training.get("eval_sha256"), "historical eval data SHA"
    )
    current_eval_sha = validate_sha256(
        data_manifest["sha256"].get("test_parquet"),
        "current test-parquet data SHA",
    )
    if historical_eval_sha != current_eval_sha:
        raise ValueError("historical eval data SHA differs from current test parquet")

    legacy_eval = legacy["evaluation"]
    reference_config = configs[STEPS[0]]
    comparisons = {
        "n": int(reference_config["n"]),
        "temperature": float(reference_config["temperature"]),
        "top_p": float(reference_config["top_p"]),
        "max_tokens": int(reference_config["max_tokens"]),
    }
    for key, value in comparisons.items():
        if not math.isclose(float(legacy_eval[key]), float(value), abs_tol=1e-12):
            raise ValueError(f"historical eval contract differs at {key}")
    if set(legacy_eval["tasks"]) != set(reference_config["tasks"]):
        raise ValueError("historical eval task set differs")


def metric_extreme(
    records: list[dict[str, Any]], key: str, *, maximum: bool
) -> dict[str, float | int]:
    record = (max if maximum else min)(records, key=lambda item: float(item[key]))
    return {"step": int(record["step"]), "value": float(record[key])}


def metric_series(summary_by_step: dict[int, dict[str, Any]], key: str) -> list[float]:
    return [float(summary_by_step[step][key]) for step in STEPS]


def build_result(
    run_dir: Path,
    historical_results: Path,
) -> dict[str, Any]:
    run_card = load_json(run_dir / "run_card.json")
    acceptance = load_json(run_dir / "acceptance.json")
    data_manifest = load_json(run_dir / "data_manifest.json")
    validate_run_contract(run_card, data_manifest)
    training_exit_code = validate_acceptance(run_dir, run_card, acceptance)
    records = load_jsonl(run_dir / "diagnostics" / "scalars.jsonl")
    if [int(record["step"]) for record in records] != list(DIAGNOSTIC_STEPS):
        raise ValueError("diagnostic schedule is incomplete")

    primary = {
        step: load_json(run_dir / f"eval_step_{step}_n8" / "outputs" / "summary.json")
        for step in STEPS
    }
    historical_regrade = {
        step: load_json(
            run_dir
            / f"eval_step_{step}_n8"
            / "historical_external_grader"
            / "summary.json"
        )
        for step in STEPS
    }
    eval_configs = {
        step: load_json(run_dir / f"eval_step_{step}_n8" / "outputs" / "eval_config.json")
        for step in STEPS
    }
    legacy = load_json(historical_results)
    legacy_token = legacy["results"]["token_opd"]
    legacy_block3 = legacy["results"]["block3_mean"]

    for step in STEPS:
        validate_eval_config(eval_configs[step], run_card, step=step)
        validate_eval_summary(
            primary[step],
            eval_configs[step],
            step=step,
            label="primary grader",
            expected_grader="verl",
        )
    validate_external_regrade(run_dir, primary, historical_regrade, eval_configs)
    validate_historical_contract(run_card, data_manifest, eval_configs, legacy)

    current_hardware = parse_hardware(run_dir / "env.txt")
    historical_hardware = legacy["comparability"]["hardware"][
        "token_block3_block5"
    ]
    total_eval_rollouts = sum(
        int(task["total_rollouts"])
        for step in STEPS
        for task in primary[step]["tasks"].values()
    )

    diagnostic_summary = {
        "record_count": len(records),
        "steps": [int(record["step"]) for record in records],
        "student_entropy_max": metric_extreme(
            records, "diagnostics/student_entropy", maximum=True
        ),
        "teacher_entropy_max": metric_extreme(
            records, "diagnostics/teacher_entropy", maximum=True
        ),
        "top16_overlap_min": metric_extreme(
            records, "diagnostics/topk_overlap_ratio", maximum=False
        ),
        "student_overlap_mass_min": metric_extreme(
            records, "diagnostics/student_overlap_mass", maximum=False
        ),
        "teacher_overlap_mass_min": metric_extreme(
            records, "diagnostics/teacher_overlap_mass", maximum=False
        ),
        "weighted_sign_flip_max": metric_extreme(
            records, "diagnostics/weighted_sign_flip_rate", maximum=True
        ),
        "normalized_leakage_max": metric_extreme(
            records, "diagnostics/normalized_leakage", maximum=True
        ),
        "grad_norm_max": metric_extreme(records, "actor/grad_norm", maximum=True),
        "block_ratio_outside_clip_max": metric_extreme(
            records,
            "diagnostics/post_update_block_ratio_outside_clip_fraction",
            maximum=True,
        ),
        "truncation_ratio_max": metric_extreme(
            records, "response_length/clip_ratio", maximum=True
        ),
        "high_entropy_steps_ge_2": [
            int(record["step"])
            for record in records
            if float(record["diagnostics/student_entropy"]) >= 2.0
        ],
        "all_recorded_nonfinite_counts_zero": all(
            float(record[key]) == 0.0
            for record in records
            for key in (
                "diagnostics/student_entropy_nonfinite_count",
                "diagnostics/teacher_entropy_nonfinite_count",
                "diagnostics/raw_token_advantage_nonfinite_count",
                "diagnostics/block_advantage_nonfinite_count",
                "diagnostics/post_update_block_ratio_overflow_count",
                "diagnostics/post_update_block_ratio_underflow_count",
                "diagnostics/post_update_block_ratio_nonfinite_count",
            )
        ),
    }

    legacy_delta = {
        "macro_avg_at_8": float(legacy_block3["macro_avg_at_8"])
        - float(legacy_token["macro_avg_at_8"]),
        "macro_pass_at_8": float(legacy_block3["macro_pass_at_8"])
        - float(legacy_token["macro_pass_at_8"]),
    }
    external_step_delta = {
        "step200_minus_step50_macro_avg_at_8": float(
            historical_regrade[200]["macro_avg_at_n"]
        )
        - float(historical_regrade[50]["macro_avg_at_n"]),
        "step200_minus_step50_macro_pass_at_8": float(
            historical_regrade[200]["macro_pass_at_n"]
        )
        - float(historical_regrade[50]["macro_pass_at_n"]),
    }
    cross_host_delta = {
        "ml2_step200_minus_train_block3_macro_avg_at_8": float(
            historical_regrade[200]["macro_avg_at_n"]
        )
        - float(legacy_block3["macro_avg_at_8"]),
        "ml2_step200_minus_train_block3_macro_pass_at_8": float(
            historical_regrade[200]["macro_pass_at_n"]
        )
        - float(legacy_block3["macro_pass_at_8"]),
    }
    step200_is_best = all(
        float(historical_regrade[200][metric])
        > max(float(historical_regrade[step][metric]) for step in (50, 100))
        for metric in ("macro_avg_at_n", "macro_pass_at_n")
    )
    current_runtime = {
        "reference_logprob_micro_batch_per_gpu": int(
            run_card["ref_log_prob_micro_batch_size_per_gpu"]
        ),
        "vllm_gpu_memory_utilization": float(
            run_card["rollout_gpu_memory_utilization"]
        ),
    }
    memory_engineering_differs = any(
        current_runtime[key] != HISTORICAL_TRAIN_RUNTIME[key]
        for key in current_runtime
    )

    return {
        "schema_version": 1,
        "experiment": "block3_mean_ml2_replication",
        "status": "complete",
        "source_commit": run_card["source_commit"],
        "run": {
            "run_dir": str(Path(run_card["diagnostic_output_dir"]).parent),
            "host": "ml2",
            "hardware": current_hardware,
            "training_exit_code": training_exit_code,
            "acceptance_passed": acceptance["passed"],
            "acceptance_issue_count": len(acceptance["issues"]),
            "acceptance_warning_count": len(acceptance.get("warnings", [])),
            "checkpoint_steps": acceptance["checkpoint_steps"],
            "diagnostic_steps": acceptance["diagnostic_steps"],
            "eval_steps": acceptance["eval_steps"],
            "total_eval_rollouts": total_eval_rollouts,
        },
        "contract": {
            "student": run_card["student_model"],
            "teacher": run_card["teacher_model"],
            "training_data": run_card["train_data"],
            "training_data_sha256": run_card["expected_train_sha256"],
            "training_rows": int(data_manifest["counts"]["train"]),
            "seed": int(run_card["seed"]),
            "steps": int(run_card["total_training_steps"]),
            "train_batch_size": int(run_card["train_batch_size"]),
            "rollouts_per_prompt": int(run_card["rollout_group_size"]),
            "learning_rate": float(run_card["learning_rate"]),
            "max_response_tokens": int(run_card["max_response_length"]),
            "block_size": 3,
            "block_advantage": "mean",
            "actor_micro_batch_per_gpu": int(
                run_card["actor_ppo_micro_batch_size_per_gpu"]
            ),
            "reference_logprob_micro_batch_per_gpu": int(
                run_card["ref_log_prob_micro_batch_size_per_gpu"]
            ),
            "rollout_logprob_micro_batch_per_gpu": int(
                run_card["rollout_log_prob_micro_batch_size_per_gpu"]
            ),
            "vllm_gpu_memory_utilization": float(
                run_card["rollout_gpu_memory_utilization"]
            ),
        },
        "diagnostics": diagnostic_summary,
        "evaluation": {
            "fixed_seed_builtin_verl": {str(step): primary[step] for step in STEPS},
            "fixed_seed_historical_external_grader": {
                str(step): historical_regrade[step] for step in STEPS
            },
            "eval_configs": {str(step): eval_configs[step] for step in STEPS},
            "historical_train_legacy": {
                "token_opd_step200": legacy_token,
                "block3_mean_step200": legacy_block3,
                "grader_sha256": HISTORICAL_GRADER_SHA256,
                "explicit_rollout_seeds": False,
            },
        },
        "deltas": {
            "historical_train_block3_minus_token": legacy_delta,
            "ml2_external_step_progression": external_step_delta,
            "cross_host_external_grader": cross_host_delta,
        },
        "comparability": {
            "same_models_data_objective_and_training_seed": True,
            "same_eval_prompts_decoding_n_and_max_tokens": True,
            "ml2_fixed_eval_rollout_seeds": eval_configs[50]["rollout_seeds"],
            "historical_train_eval_had_no_explicit_rollout_seed": (
                "rollout_seeds" not in legacy["evaluation"]
            ),
            "primary_grader_differs_from_historical": True,
            "historical_grader_regrade_available": True,
            "regrade_input_hash_manifest_paths_validated": True,
            "raw_outputs_rehashed_by_local_report_builder": False,
            "hardware": {
                "ml2": current_hardware,
                "historical_train": historical_hardware,
                "differs": current_hardware != historical_hardware,
            },
            "memory_engineering": {
                "ml2": current_runtime,
                "historical_train": HISTORICAL_TRAIN_RUNTIME,
                "differs": memory_engineering_differs,
            },
            "same_host_ml2_token_baseline_available": False,
            "replication_seed_count": 1,
        },
        "conclusion": {
            "status": "not_independently_validated",
            "training_stability": (
                "completed_without_sustained_numerical_or_fixed_eval_collapse_"
                "but_high_entropy_and_truncation_risk_remains"
            ),
            "within_ml2_checkpoint_progression": (
                "step200_outperforms_step50_and_step100"
                if step200_is_best
                else "step200_is_not_best_on_both_metrics"
            ),
            "block3_vs_token_on_ml2": "not_tested",
            "cross_host_absolute_repeatability": "inconclusive_due_to_eval_seed_and_runtime_differences",
            "historical_evidence": "one_seed_weak_positive",
            "research_claim": (
                "This run completed without non-finite values or a sustained fixed-eval "
                "collapse, but high-entropy and truncation risk remains. It does not "
                "independently replicate the causal Block3-over-token gain. A same-host "
                "token baseline and multi-seed paired runs remain required."
            ),
        },
    }


def plot_diagnostics(run_dir: Path, assets_dir: Path, label: str) -> None:
    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent
    for path in (scripts_dir, root_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from analyze_block10_collapse_diagnostics import (  # noqa: PLC0415
        SCALAR_GROUPS,
        load_snapshots,
        plot_entropy_segments,
        plot_heatmaps,
        plot_scalar_curves,
    )
    from opd_ext.analysis import load_scalar_records  # noqa: PLC0415

    records = load_scalar_records(run_dir / "diagnostics" / "scalars.jsonl")
    snapshots = load_snapshots(run_dir)
    for filename, metrics in SCALAR_GROUPS.items():
        plot_scalar_curves({label: records}, metrics, assets_dir / filename)
    plot_heatmaps(
        {label: snapshots},
        assets_dir / "position_heatmaps.png",
        bin_size=128,
        min_count=8,
    )
    plot_entropy_segments({label: snapshots}, assets_dir / "entropy_segments.png")


def plot_eval_curves(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    primary = result["evaluation"]["fixed_seed_builtin_verl"]
    external = result["evaluation"]["fixed_seed_historical_external_grader"]
    legacy = result["evaluation"]["historical_train_legacy"]
    figures, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "macro_avg_at_n", "Macro Avg@8"),
        (axes[1], "macro_pass_at_n", "Macro Pass@8"),
    ):
        axis.plot(STEPS, [primary[str(step)][key] for step in STEPS], marker="o", label="ml2 builtin VERL")
        axis.plot(STEPS, [external[str(step)][key] for step in STEPS], marker="s", label="ml2 historical grader")
        legacy_key = "macro_avg_at_8" if "avg" in key else "macro_pass_at_8"
        axis.axhline(
            legacy["token_opd_step200"][legacy_key],
            color="#7a7f87",
            linestyle="--",
            label="train token legacy",
        )
        axis.axhline(
            legacy["block3_mean_step200"][legacy_key],
            color="#b24a3b",
            linestyle=":",
            label="train Block3 legacy",
        )
        axis.set_title(title)
        axis.set_xlabel("Checkpoint step")
        axis.set_xticks(STEPS)
        axis.set_ylim(0, max(0.58, axis.get_ylim()[1]))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    handles, labels = axes[1].get_legend_handles_labels()
    figures.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figures.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figures)


def plot_task_step200(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    legacy = result["evaluation"]["historical_train_legacy"]
    ml2 = result["evaluation"]["fixed_seed_historical_external_grader"]["200"]
    sources = (
        ("train token legacy", legacy["token_opd_step200"], "avg_at_8", "pass_at_8"),
        ("train Block3 legacy", legacy["block3_mean_step200"], "avg_at_8", "pass_at_8"),
        ("ml2 Block3 fixed seeds", ml2, "avg_at_n", "pass_at_n"),
    )
    colors = ("#7a7f87", "#b24a3b", "#1f6f5f")
    x = np.arange(len(TASKS))
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for source_index, (label, summary, avg_key, pass_key) in enumerate(sources):
        tasks = summary["tasks"]
        axes[0].bar(
            x + (source_index - 1) * width,
            [tasks[task][avg_key] for task in TASKS],
            width,
            color=colors[source_index],
            label=label,
        )
        axes[1].bar(
            x + (source_index - 1) * width,
            [tasks[task][pass_key] for task in TASKS],
            width,
            color=colors[source_index],
            label=label,
        )
    for axis, title in zip(axes, ("Per-task Avg@8", "Per-task Pass@8")):
        axis.set_title(title)
        axis.set_xticks(x, [task.upper() for task in TASKS])
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_length_and_format(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    primary = result["evaluation"]["fixed_seed_builtin_verl"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = ("#2463a6", "#b24a3b", "#7c5e9e", "#1f6f5f")
    for task, color in zip(TASKS, colors):
        values = [primary[str(step)]["tasks"][task] for step in STEPS]
        axes[0].plot(
            STEPS,
            [value["avg_response_length_tokens"] for value in values],
            marker="o",
            color=color,
            label=task.upper(),
        )
        axes[1].plot(
            STEPS,
            [value["format_error_rollouts"] / value["total_rollouts"] for value in values],
            marker="o",
            color=color,
            label=task.upper(),
        )
    axes[0].set_title("Average response length")
    axes[0].set_ylabel("Tokens")
    axes[1].set_title("Format-error fraction")
    axes[1].set_ylabel("Fraction")
    for axis in axes:
        axis.set_xlabel("Checkpoint step")
        axis.set_xticks(STEPS)
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def task_rows(summary: dict[str, Any]) -> str:
    return "".join(
        "<tr>"
        f"<th>{html.escape(task.upper())}</th>"
        f"<td>{summary['tasks'][task]['avg_at_n']:.4f}</td>"
        f"<td>{summary['tasks'][task]['pass_at_n']:.4f}</td>"
        f"<td>{summary['tasks'][task]['avg_response_length_tokens']:.0f}</td>"
        f"<td>{summary['tasks'][task]['format_error_rollouts']}</td>"
        "</tr>"
        for task in TASKS
    )


def write_html(result: dict[str, Any], output_html: Path, assets_dir: Path) -> None:
    primary = result["evaluation"]["fixed_seed_builtin_verl"]
    external = result["evaluation"]["fixed_seed_historical_external_grader"]
    legacy = result["evaluation"]["historical_train_legacy"]
    diagnostics = result["diagnostics"]
    contract = result["contract"]
    run = result["run"]
    comparability = result["comparability"]
    asset_prefix = html.escape(assets_dir.name)

    checkpoint_rows = "".join(
        "<tr>"
        f"<th>Step {step}</th>"
        f"<td>{primary[str(step)]['macro_avg_at_n']:.4f}</td>"
        f"<td>{primary[str(step)]['macro_pass_at_n']:.4f}</td>"
        f"<td>{external[str(step)]['macro_avg_at_n']:.4f}</td>"
        f"<td>{external[str(step)]['macro_pass_at_n']:.4f}</td>"
        "</tr>"
        for step in STEPS
    )
    legacy_rows = "".join(
        "<tr>"
        f"<th>{label}</th>"
        f"<td>{summary['macro_avg_at_8']:.4f}</td>"
        f"<td>{summary['macro_pass_at_8']:.4f}</td>"
        "</tr>"
        for label, summary in (
            ("train token OPD, legacy eval", legacy["token_opd_step200"]),
            ("train Block3, legacy eval", legacy["block3_mean_step200"]),
            ("ml2 Block3, fixed seeds + historical grader", {
                "macro_avg_at_8": external["200"]["macro_avg_at_n"],
                "macro_pass_at_8": external["200"]["macro_pass_at_n"],
            }),
        )
    )
    standard_figures = (
        ("eval_macro_curves.png", "Checkpoint evaluation curves"),
        ("eval_task_step200.png", "Step 200 task comparison"),
        ("eval_length_format.png", "Length and format behavior"),
        ("scalar_alignment.png", "Alignment diagnostics"),
        ("scalar_credit.png", "Credit-assignment diagnostics"),
        ("scalar_optimization.png", "Optimization and policy drift"),
        ("entropy_segments.png", "Entropy by response segment"),
        ("position_heatmaps.png", "Step x output-position heatmaps (12 metrics)"),
    )
    figures = "".join(
        "<figure>"
        f"<h3>{html.escape(title)}</h3>"
        f"<img src=\"{asset_prefix}/{filename}\" alt=\"{html.escape(title)}\">"
        "</figure>"
        for filename, title in standard_figures
    )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Block3 mean ml2 复现实验</title>
<style>
:root {{ --ink:#172033; --muted:#596474; --line:#d8dee8; --soft:#f5f7fa; --green:#1f6f5f; --red:#a43d32; --blue:#2463a6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.58; letter-spacing:0; }}
main {{ width:min(1380px,calc(100% - 32px)); margin:28px auto 72px; }}
header {{ border-left:5px solid var(--green); padding:4px 0 4px 18px; margin-bottom:26px; }}
h1 {{ margin:0 0 7px; font-size:clamp(27px,3vw,42px); }}
h2 {{ margin:34px 0 12px; font-size:23px; }}
h3 {{ margin:0 0 10px; font-size:18px; }}
p {{ margin:8px 0; }}
.lead {{ color:var(--muted); font-size:17px; max-width:1000px; }}
.verdict {{ border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:20px 0; margin:24px 0; }}
.verdict strong {{ color:var(--red); font-size:21px; }}
.facts {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); }}
.fact {{ min-width:0; padding:13px 15px; border-right:1px solid var(--line); }}
.fact:last-child {{ border-right:0; }}
.fact span {{ display:block; color:var(--muted); font-size:12px; }}
.fact b {{ display:block; overflow-wrap:anywhere; font-size:18px; }}
.note {{ border-left:4px solid var(--blue); background:var(--soft); padding:12px 15px; margin:14px 0; }}
.warn {{ border-left-color:var(--red); }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 24px; }}
th,td {{ border:1px solid var(--line); padding:9px 11px; text-align:left; vertical-align:top; }}
thead th {{ background:var(--soft); }}
tbody th {{ background:#fafbfc; white-space:nowrap; }}
figure {{ margin:0 0 34px; }}
img {{ display:block; width:100%; height:auto; border:1px solid var(--line); background:#fff; }}
code {{ overflow-wrap:anywhere; }}
ul {{ padding-left:22px; }}
@media (max-width:780px) {{
  main {{ width:min(100% - 20px,1380px); margin-top:18px; }}
  .facts {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .fact:nth-child(2) {{ border-right:0; }}
  .fact:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }}
  table {{ display:block; overflow-x:auto; white-space:nowrap; }}
}}
</style>
</head>
<body><main>
<header>
<h1>Block3 mean ml2 复现实验</h1>
<p class="lead">DAPO-Math-17K、{html.escape(Path(contract['student']).name)} 学生、{html.escape(Path(contract['teacher']).name)} 教师；{contract['steps']}-step Block3 mean，完整诊断与 Step 50/100/200 n=8 评估。</p>
</header>

<section class="verdict">
<strong>结论：未观察到持续性数值或固定评测崩塌，但尾部高熵/截断风险仍存在；Block3 相对 token OPD 的收益尚未被第二台机器独立验证。</strong>
<p>ml2 没有同机 token baseline；Step 200 虽优于本次 Step 50/100，但历史 grader 重评分的 macro Avg@8/Pass@8 为 {external['200']['macro_avg_at_n']:.4f}/{external['200']['macro_pass_at_n']:.4f}，低于 train 旧 Block3 的 {legacy['block3_mean_step200']['macro_avg_at_8']:.4f}/{legacy['block3_mean_step200']['macro_pass_at_8']:.4f}。两次 eval 的 rollout seed 实现不同，因此也不能把差值完全归因于硬件或训练。</p>
</section>

<section class="facts">
<div class="fact"><span>Training</span><b>{contract['steps']} / {contract['steps']}</b></div>
<div class="fact"><span>Diagnostics</span><b>{diagnostics['record_count']} snapshots</b></div>
<div class="fact"><span>Full eval rollouts</span><b>{run['total_eval_rollouts']:,}</b></div>
<div class="fact"><span>Audit</span><b>{run['acceptance_issue_count']} issues / {run['acceptance_warning_count']} warnings</b></div>
</section>

<h2>实验契约</h2>
<table><tbody>
<tr><th>模型</th><td>Student: <code>{html.escape(Path(contract['student']).name)}</code>；Teacher: <code>{html.escape(Path(contract['teacher']).name)}</code>。</td></tr>
<tr><th>训练数据</th><td>DAPO-Math-17K raw {contract['training_rows']:,}-row pool；SHA-256 <code>{html.escape(contract['training_data_sha256'])}</code>。seed={contract['seed']} 的确定性索引排列，每步 {contract['train_batch_size']} prompt、每 prompt {contract['rollouts_per_prompt']} rollout。</td></tr>
<tr><th>目标</th><td>Block size 3；连续 token advantage 取 mean；block current/old log-prob 求和并做 block-level PPO ratio/clipping。</td></tr>
<tr><th>优化</th><td>{contract['steps']} steps；LR {contract['learning_rate']:.0e}；actor micro-batch/GPU={contract['actor_micro_batch_per_gpu']}；reference log-prob micro-batch/GPU={contract['reference_logprob_micro_batch_per_gpu']}；vLLM utilization={contract['vllm_gpu_memory_utilization']:.1f}；最大响应 {contract['max_response_tokens']:,}。</td></tr>
<tr><th>固定评估</th><td>Math500/AIME24/AIME25/AMC23；n=8；seed {comparability['ml2_fixed_eval_rollout_seeds'][0]}-{comparability['ml2_fixed_eval_rollout_seeds'][-1]}；temperature=1.0；top-p=0.9；thinking=false；每 checkpoint 5,144 rollout。</td></tr>
</tbody></table>

<h2>评估结果</h2>
<p>内置 VERL grader 是本次预注册主结果；historical grader 使用 train 旧实验实际采用的 <code>grade_answer_verl</code>（SHA <code>{legacy['grader_sha256']}</code>）离线重评分。生成器逐 checkpoint 校验 hash manifest 中 grader SHA/路径、四个 primary output 路径、n=8、seed 和 raw-output 元数据；仓库未复制大体积 raw rollout，因此本地报告生成器没有重新计算 raw 文件 SHA。</p>
<table>
<thead><tr><th>Checkpoint</th><th>内置 Avg@8</th><th>内置 Pass@8</th><th>历史 grader Avg@8</th><th>历史 grader Pass@8</th></tr></thead>
<tbody>{checkpoint_rows}</tbody>
</table>
<div class="note warn">内置 grader 无法正确覆盖当前 AMC23 answer 形式，三个 checkpoint 的 AMC23 均为 0；因此跨历史比较必须使用 historical grader 列。主结果仍保留，不能事后覆盖。</div>

<h2>Step 200 任务明细</h2>
<table>
<thead><tr><th>Task</th><th>Avg@8</th><th>Pass@8</th><th>平均 token</th><th>格式错误</th></tr></thead>
<tbody>{task_rows(external['200'])}</tbody>
</table>

<h2>与 train 旧结果的边界</h2>
<table>
<thead><tr><th>Run</th><th>Macro Avg@8</th><th>Macro Pass@8</th></tr></thead>
<tbody>{legacy_rows}</tbody>
</table>
<ul>
<li>train 旧实验中 Block3 相对 token 的单 seed 增量为 Avg@8 {result['deltas']['historical_train_block3_minus_token']['macro_avg_at_8']:+.4f}、Pass@8 {result['deltas']['historical_train_block3_minus_token']['macro_pass_at_8']:+.4f}，属于弱阳性证据。</li>
<li>本次 ml2 只跑 Block3，没有同机 token baseline，所以不能复现该因果差值。</li>
<li>train 旧 evaluator 未显式固定 rollout seed；ml2 固定 seed 21-28。historical grader 对齐只消除了评分器差异，没有消除采样差异。</li>
<li>硬件由 <code>{html.escape(comparability['hardware']['historical_train'])}</code> 变为 <code>{html.escape(comparability['hardware']['ml2'])}</code>；reference micro-batch 从 {comparability['memory_engineering']['historical_train']['reference_logprob_micro_batch_per_gpu']} 变为 {comparability['memory_engineering']['ml2']['reference_logprob_micro_batch_per_gpu']}，vLLM utilization 从 {comparability['memory_engineering']['historical_train']['vllm_gpu_memory_utilization']:.1f} 变为 {comparability['memory_engineering']['ml2']['vllm_gpu_memory_utilization']:.1f}。</li>
</ul>

<h2>训练诊断与风险</h2>
<p>{diagnostics['record_count']} 个快照全部通过有限性审计。student entropy 最大 {diagnostics['student_entropy_max']['value']:.3f}（Step {diagnostics['student_entropy_max']['step']}），最小共享 student Top-16 mass {diagnostics['student_overlap_mass_min']['value']:.3f}（Step {diagnostics['student_overlap_mass_min']['step']}），最大 weighted sign-flip {diagnostics['weighted_sign_flip_max']['value']:.3f}，最大 block ratio 越界率 {percent(diagnostics['block_ratio_outside_clip_max']['value'])}。</p>
<div class="note warn">最大训练 batch 截断率为 {percent(diagnostics['truncation_ratio_max']['value'])}（Step {diagnostics['truncation_ratio_max']['step']}），因此不能下“无条件稳定”的结论。Step 115-185 出现多次高熵 batch，Step 200 也再次处于高熵/全截断 batch；但 teacher entropy 同步上升、Top-16 overlap 保持较高，且 Step 200 固定 eval 最优、Math500 平均长度从 Step 50 的 3,098 降到 Step 200 的 1,792 token。现有证据只支持“没有持续性数值或固定 eval collapse”。</div>

{figures}

<h2>后续验证</h2>
<ol>
<li>在 ml2 用同一代码、同一资源设置补跑 token OPD baseline。</li>
<li>token 与 Block3 做至少 3 个 paired training seeds；每个 checkpoint 使用相同固定 eval seeds。</li>
<li>同时报告 historical grader 与预注册主 grader，或在实验前锁定一个覆盖 AMC23 的统一 grader。</li>
<li>以 prompt-batch hash 配对比较 entropy、overlap mass、sign-flip 和 leakage，区分数据难度与训练漂移。</li>
</ol>

<h2>可追溯性</h2>
<p>Source commit: <code>{html.escape(result['source_commit'])}</code>。训练退出码 {run['training_exit_code']}；三个 checkpoint、{diagnostics['record_count']} 个 NPZ 和三个 5,144-rollout eval 已通过自动审计（{run['acceptance_issue_count']} issues，{run['acceptance_warning_count']} warnings）。仓库仅保存 curated JSON、图表和 HTML；raw rollout、日志与 checkpoint 保留在实验存储。</p>
</main></body></html>"""
    output_html.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.assets_dir.mkdir(parents=True, exist_ok=True)

    result = build_result(args.run_dir, args.historical_results)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_diagnostics(args.run_dir, args.assets_dir, args.label)
    plot_eval_curves(result, args.assets_dir / "eval_macro_curves.png")
    plot_task_step200(result, args.assets_dir / "eval_task_step200.png")
    plot_length_and_format(result, args.assets_dir / "eval_length_format.png")
    write_html(result, args.output_html, args.assets_dir)


if __name__ == "__main__":
    main()

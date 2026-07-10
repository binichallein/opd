#!/usr/bin/env python3
"""Create paired Block10 collapse diagnostics from train/ml2 run artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from opd_ext.analysis import (
    aggregate_position_segment,
    bin_position_statistics,
    classify_leading_mechanism,
    classify_ordered_propagation,
    confirmed_chain_onset,
    detect_sustained_onset,
    first_nonfinite_step,
    load_scalar_records,
    normalize_hash_manifest_lines,
)


HEATMAP_METRICS = (
    ("student_entropy", "Student entropy", 0.0, 12.0),
    ("teacher_entropy", "Teacher entropy", 0.0, 12.0),
    ("entropy_gap_absolute", "Absolute entropy gap", 0.0, None),
    ("overlap_ratio", "Top-16 overlap ratio", 0.0, 1.0),
    ("sign_flip", "SignFlipRate", 0.0, 1.0),
    ("leakage", "LeakageMagnitude", 0.0, None),
)


SCALAR_GROUPS = {
    "scalar_alignment.png": (
        "diagnostics/student_entropy",
        "diagnostics/teacher_entropy",
        "diagnostics/entropy_gap_signed",
        "diagnostics/entropy_gap_absolute",
        "diagnostics/topk_overlap_ratio",
        "diagnostics/student_overlap_mass",
        "diagnostics/teacher_overlap_mass",
        "diagnostics/overlap_token_advantage",
    ),
    "scalar_credit.png": (
        "diagnostics/sign_flip_rate",
        "diagnostics/weighted_sign_flip_rate",
        "diagnostics/leakage_magnitude",
        "diagnostics/normalized_leakage",
        "diagnostics/raw_token_advantage_mean",
        "diagnostics/raw_token_advantage_std",
        "diagnostics/raw_token_advantage_p95",
        "diagnostics/raw_token_advantage_max",
        "diagnostics/block_advantage_mean",
        "diagnostics/block_advantage_std",
        "diagnostics/block_advantage_p95",
        "diagnostics/block_advantage_max",
    ),
    "scalar_optimization.png": (
        "diagnostics/post_update_block_log_ratio_abs_mean",
        "diagnostics/post_update_block_log_ratio_abs_p95",
        "diagnostics/post_update_block_log_ratio_abs_max",
        "diagnostics/post_update_block_ratio_p95",
        "diagnostics/post_update_block_ratio_max",
        "diagnostics/post_update_block_ratio_outside_clip_fraction",
        "actor/pg_loss",
        "actor/pg_clipfrac",
        "actor/grad_norm",
        "response_length/mean",
        "response_length/max",
        "response_length/clip_ratio",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, required=True)
    parser.add_argument("--ml2-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--position-bin", type=int, default=128)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_snapshots(run_dir: Path) -> list[dict[str, object]]:
    diagnostics_dir = run_dir / "diagnostics"
    snapshots = []
    for path in sorted(diagnostics_dir.glob("step_*.npz")):
        with np.load(path) as data:
            snapshot: dict[str, object] = {"step": int(data["step"].item())}
            metric_names = {key.split("__", 1)[0] for key in data.files if "__" in key}
            snapshot["metrics"] = {
                name: {
                    "sum": data[f"{name}__sum"].copy(),
                    "squared_sum": data[f"{name}__squared_sum"].copy(),
                    "valid_count": data[f"{name}__valid_count"].copy(),
                }
                for name in metric_names
            }
            snapshots.append(snapshot)
    if not snapshots:
        raise FileNotFoundError(f"no diagnostic snapshots under {diagnostics_dir}")
    return snapshots


def build_heatmap(
    snapshots: list[dict[str, object]],
    metric: str,
    bin_size: int,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    steps = []
    for snapshot in snapshots:
        metrics = snapshot["metrics"]
        if metric not in metrics:
            continue
        means, _ = bin_position_statistics(metrics[metric], bin_size, min_count)
        rows.append(means)
        steps.append(snapshot["step"])
    if not rows:
        raise KeyError(f"metric {metric} is absent from diagnostic snapshots")
    return np.stack(rows), np.asarray(steps, dtype=np.int64)


def plot_scalar_curves(
    records: dict[str, list[dict[str, object]]], metrics: tuple[str, ...], output_path: Path
) -> None:
    columns = 4
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(18, 3.8 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for axis, metric in zip(axes_flat, metrics):
        for host, host_records in records.items():
            points = [record for record in host_records if metric in record]
            if points:
                axis.plot(
                    [record["step"] for record in points],
                    [record[metric] for record in points],
                    label=host,
                    linewidth=1.8,
                )
        axis.set_title(metric.replace("diagnostics/", ""), fontsize=10)
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[0].legend(handles, labels)
    for axis in axes_flat[len(metrics) :]:
        axis.axis("off")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmaps(
    snapshots_by_host: dict[str, list[dict[str, object]]],
    output_path: Path,
    bin_size: int,
    min_count: int,
) -> None:
    hosts = tuple(snapshots_by_host)
    fig, axes = plt.subplots(len(HEATMAP_METRICS), len(hosts), figsize=(16, 18), constrained_layout=True)
    for row_index, (metric, title, fixed_min, fixed_max) in enumerate(HEATMAP_METRICS):
        matrices = {}
        steps_by_host = {}
        for host in hosts:
            matrices[host], steps_by_host[host] = build_heatmap(
                snapshots_by_host[host], metric, bin_size, min_count
            )
        finite_values = np.concatenate(
            [matrix[np.isfinite(matrix)] for matrix in matrices.values() if np.isfinite(matrix).any()]
        )
        dynamic_max = float(np.quantile(finite_values, 0.99)) if finite_values.size else 1.0
        vmin = fixed_min
        vmax = fixed_max if fixed_max is not None else max(dynamic_max, 1e-8)
        for column_index, host in enumerate(hosts):
            axis = axes[row_index, column_index]
            matrix = matrices[host]
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#d1d5db")
            image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(f"{host}: {title}")
            axis.set_ylabel("Training step")
            axis.set_xlabel("Output position (K tokens)")
            ticks = np.linspace(0, matrix.shape[1] - 1, 6, dtype=int)
            axis.set_xticks(ticks, [f"{tick * bin_size / 1000:.1f}" for tick in ticks])
            row_ticks = np.arange(len(steps_by_host[host]))
            axis.set_yticks(row_ticks, steps_by_host[host])
            fig.colorbar(image, ax=axis, fraction=0.025)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_entropy_segments(
    snapshots_by_host: dict[str, list[dict[str, object]]], output_path: Path
) -> None:
    segments = ((0, 4096, "0-4K"), (4096, 12288, "4-12K"), (12288, 16384, "12-16K"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for column, (host, snapshots) in enumerate(snapshots_by_host.items()):
        for row, (metric, model_name) in enumerate(
            (("student_entropy", "Student"), ("teacher_entropy", "Teacher"))
        ):
            axis = axes[row, column]
            for start, end, label in segments:
                steps, values = segment_series(snapshots, metric, start, end)
                axis.plot(steps, values, marker="o", label=label)
            axis.set_title(f"{host}: {model_name}")
            axis.set_xlabel("Training step")
            axis.set_ylabel(f"{model_name} entropy (nats)")
            axis.grid(alpha=0.25)
            axis.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def segment_series(
    snapshots: list[dict[str, object]],
    metric: str,
    start: int,
    end: int,
    min_coverage: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    steps = []
    values = []
    for snapshot in snapshots:
        statistics = snapshot["metrics"][metric]
        steps.append(int(snapshot["step"]))
        values.append(
            aggregate_position_segment(
                statistics,
                start=start,
                end=end,
                min_coverage=min_coverage,
            )
        )
    return np.asarray(steps, dtype=np.int64), np.asarray(values, dtype=np.float64)


def segment_onset(
    snapshots: list[dict[str, object]],
    metric: str,
    start: int,
    end: int,
    direction: str,
) -> int | None:
    steps, values = segment_series(snapshots, metric, start, end)
    return detect_sustained_onset(steps, values, direction=direction)


def compute_onsets(
    records: list[dict[str, object]], snapshots: list[dict[str, object]]
) -> dict[str, int | None]:
    def onset(metric: str, direction: str) -> int | None:
        points = [record for record in records if metric in record]
        if not points:
            return None
        return detect_sustained_onset(
            np.asarray([record["step"] for record in points]),
            np.asarray([record[metric] for record in points]),
            direction=direction,
        )

    def minimum(*values: int | None) -> int | None:
        available = [value for value in values if value is not None]
        return min(available) if available else None

    tail_teacher_entropy = segment_onset(
        snapshots, "teacher_entropy", 12288, 16384, "up"
    )
    tail_entropy_gap = segment_onset(
        snapshots, "entropy_gap_absolute", 12288, 16384, "up"
    )
    tail_overlap = segment_onset(snapshots, "overlap_ratio", 12288, 16384, "down")
    overlap = onset("diagnostics/topk_overlap_ratio", "down")
    student_mass = onset("diagnostics/student_overlap_mass", "down")
    support_drift = minimum(overlap, student_mass)
    teacher_ood_confirmation = minimum(tail_entropy_gap, tail_overlap)
    teacher_ood = confirmed_chain_onset(tail_teacher_entropy, teacher_ood_confirmation)
    if tail_teacher_entropy is not None and support_drift is not None:
        if tail_teacher_entropy <= support_drift:
            support_drift = None
    numerical_steps = [
        int(record["step"])
        for record in records
        if any(
            float(value) > 0
            for key, value in record.items()
            if key.startswith("diagnostics/")
            and key.endswith(("nonfinite_count", "overflow_count", "underflow_count"))
        )
    ]
    return {
        "sign_flip": onset("diagnostics/weighted_sign_flip_rate", "up"),
        "leakage": onset("diagnostics/normalized_leakage", "up"),
        "block_ratio": minimum(
            onset("diagnostics/post_update_block_log_ratio_abs_p95", "up"),
            onset("diagnostics/post_update_block_ratio_outside_clip_fraction", "up"),
        ),
        "teacher_ood": teacher_ood,
        "support_drift": support_drift,
        "numerical": minimum(
            min(numerical_steps) if numerical_steps else None,
            first_nonfinite_step(records),
        ),
        "student_entropy": onset("diagnostics/student_entropy", "up"),
        "tail_teacher_entropy": tail_teacher_entropy,
        "tail_entropy_gap": tail_entropy_gap,
        "tail_overlap": tail_overlap,
        "student_entropy_front": segment_onset(
            snapshots, "student_entropy", 0, 4096, "up"
        ),
        "student_entropy_middle": segment_onset(
            snapshots, "student_entropy", 4096, 12288, "up"
        ),
        "student_entropy_tail": segment_onset(
            snapshots, "student_entropy", 12288, 16384, "up"
        ),
    }


def classify_entropy_propagation(onsets: dict[str, int | None]) -> str:
    front = onsets.get("student_entropy_front")
    middle = onsets.get("student_entropy_middle")
    tail = onsets.get("student_entropy_tail")
    return classify_ordered_propagation(front=front, middle=middle, tail=tail)


def write_report(
    output_dir: Path,
    onsets_by_host: dict[str, dict[str, int | None]],
    classifications: dict[str, str],
    propagation: dict[str, str],
    acceptance: dict[str, object],
) -> None:
    stable = (
        bool(acceptance["passed"])
        and len(set(classifications.values())) == 1
        and "undetermined" not in classifications.values()
        and set(propagation.values()) == {"tail_to_front"}
    )
    rows = []
    for host, onsets in onsets_by_host.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(host)}</td>"
            f"<td>{html.escape(classifications[host])}</td>"
            f"<td>{html.escape(propagation[host])}</td>"
            + "".join(f"<td>{'-' if value is None else value}</td>" for value in onsets.values())
            + "</tr>"
        )
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in next(iter(onsets_by_host.values())))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Block10 Collapse 双机诊断</title>
<style>body{{font-family:system-ui,sans-serif;margin:32px;color:#172033}}main{{max-width:1280px;margin:auto}}img{{width:100%;border:1px solid #d7dde8;margin:12px 0 28px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #cbd3df;padding:8px;text-align:left}}th{{background:#eef2f7}}.status{{font-weight:700;color:{'#166534' if stable else '#9a3412'}}}</style></head>
<body><main><h1>Block10 Collapse 双机诊断</h1>
<p class="status">验收门禁：{'通过' if acceptance['passed'] else '未通过'}；{html.escape('; '.join(acceptance['issues']) or '所有必需产物完整且双机哈希一致')}</p>
<p class="status">跨机器稳定结论：{'是' if stable else '否；当前只能报告分机器领先机制'}</p>
<table><thead><tr><th>Host</th><th>Classification</th><th>Entropy propagation</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>师生分布动态</h2><img src="scalar_alignment.png" alt="alignment scalar curves">
<h2>Credit assignment 动态</h2><img src="scalar_credit.png" alt="credit scalar curves">
<h2>优化与 Policy drift 动态</h2><img src="scalar_optimization.png" alt="optimization scalar curves">
<h2>位置热图</h2><img src="position_heatmaps.png" alt="position heatmaps">
<h2>前中后段 Student / Teacher Entropy</h2><img src="entropy_segments.png" alt="entropy segments">
<p>异常起点以 Step 1-30 的 median 和 MAD 建立基线，连续两个诊断点越界才计为 sustained onset。时间并列窗口为 5 steps；并列时标记 mixed mechanism。</p>
</main></body></html>"""
    (output_dir / "diagnosis.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {"train-A800": args.train_run, "ml2-A100": args.ml2_run}
    acceptance_by_host = {}
    for host, run_dir in run_dirs.items():
        acceptance_path = run_dir / "acceptance.json"
        if not acceptance_path.is_file():
            if not args.allow_incomplete:
                raise FileNotFoundError(
                    f"missing {acceptance_path}; run scripts/audit_block10_run.py first"
                )
            acceptance_by_host[host] = {
                "passed": False,
                "issues": ["acceptance.json missing"],
            }
        else:
            acceptance_by_host[host] = json.loads(
                acceptance_path.read_text(encoding="utf-8")
            )
    cross_host_issues = []
    source_commits = {
        result.get("source_commit") for result in acceptance_by_host.values()
    }
    if len(source_commits) != 1 or None in source_commits:
        cross_host_issues.append("cross-host source_commit mismatch")
    for manifest_name in ("artifact_hashes.sha256", "script_hashes.sha256"):
        manifests = []
        for run_dir in run_dirs.values():
            path = run_dir / manifest_name
            if not path.is_file():
                manifests.append(None)
            else:
                manifests.append(
                    normalize_hash_manifest_lines(path.read_text(encoding="utf-8").splitlines())
                )
        if manifests[0] is None or manifests[1] is None or manifests[0] != manifests[1]:
            cross_host_issues.append(f"cross-host {manifest_name} mismatch")
    acceptance_issues = [
        f"{host}: {issue}"
        for host, result in acceptance_by_host.items()
        for issue in result.get("issues", [])
    ] + cross_host_issues
    acceptance = {
        "passed": all(result.get("passed", False) for result in acceptance_by_host.values())
        and not cross_host_issues,
        "issues": acceptance_issues,
        "by_host": acceptance_by_host,
    }
    snapshots = {host: load_snapshots(run_dir) for host, run_dir in run_dirs.items()}
    records = {
        host: load_scalar_records(run_dir / "diagnostics" / "scalars.jsonl")
        for host, run_dir in run_dirs.items()
    }
    prompt_hashes = {
        host: {
            int(record["step"]): record.get("prompt_batch_sha256")
            for record in host_records
            if "prompt_batch_sha256" in record
        }
        for host, host_records in records.items()
    }
    if len({tuple(sorted(values.items())) for values in prompt_hashes.values()}) != 1:
        acceptance["issues"].append("cross-host prompt batch hashes mismatch")
        acceptance["passed"] = False
    for filename, scalar_metrics in SCALAR_GROUPS.items():
        plot_scalar_curves(records, scalar_metrics, args.output_dir / filename)
    plot_heatmaps(
        snapshots,
        args.output_dir / "position_heatmaps.png",
        args.position_bin,
        args.min_count,
    )
    plot_entropy_segments(snapshots, args.output_dir / "entropy_segments.png")
    onsets = {
        host: compute_onsets(records[host], snapshots[host]) for host in run_dirs
    }
    classifications = {
        host: classify_leading_mechanism(host_onsets) for host, host_onsets in onsets.items()
    }
    propagation = {
        host: classify_entropy_propagation(host_onsets) for host, host_onsets in onsets.items()
    }
    summary = {
        "onsets": onsets,
        "classifications": classifications,
        "entropy_propagation": propagation,
        "acceptance": acceptance,
    }
    (args.output_dir / "onsets.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(args.output_dir, onsets, classifications, propagation, acceptance)


if __name__ == "__main__":
    main()

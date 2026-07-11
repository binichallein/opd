import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "scripts" / "analyze_block10_collapse_diagnostics.py"
SINGLE_ANALYZER_PATH = ROOT / "scripts" / "analyze_single_opd_diagnostics.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("opd_diagnostic_analyzer", ANALYZER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_single_analyzer():
    spec = importlib.util.spec_from_file_location(
        "single_opd_diagnostic_analyzer", SINGLE_ANALYZER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def position_statistics(length=16384, value=1.0, coverage=8):
    return {
        "sum": np.full(length, value * coverage, dtype=np.float64),
        "squared_sum": np.full(length, value * value * coverage, dtype=np.float64),
        "valid_count": np.full(length, coverage, dtype=np.int64),
    }


def test_position_and_entropy_plots_support_one_host(tmp_path):
    pytest.importorskip("matplotlib")
    analyzer = load_analyzer()
    metrics = {
        name: position_statistics(value=index + 1)
        for index, (name, *_rest) in enumerate(analyzer.HEATMAP_METRICS)
    }
    metrics["student_entropy"] = position_statistics(value=1.0)
    metrics["teacher_entropy"] = position_statistics(value=0.8)
    snapshots = {"ml2-A100": [{"step": 1, "metrics": metrics}]}

    heatmap_path = tmp_path / "position_heatmaps.png"
    entropy_path = tmp_path / "entropy_segments.png"
    analyzer.plot_heatmaps(snapshots, heatmap_path, bin_size=128, min_count=8)
    analyzer.plot_entropy_segments(snapshots, entropy_path)

    assert heatmap_path.stat().st_size > 0
    assert entropy_path.stat().st_size > 0


def test_position_heatmaps_cover_all_saved_alignment_and_policy_drift_metrics():
    pytest.importorskip("matplotlib")
    analyzer = load_analyzer()
    names = {name for name, *_rest in analyzer.HEATMAP_METRICS}

    assert {
        "student_entropy",
        "teacher_entropy",
        "entropy_gap_signed",
        "entropy_gap_absolute",
        "overlap_ratio",
        "student_overlap_mass",
        "teacher_overlap_mass",
        "overlap_token_advantage",
        "sign_flip",
        "leakage",
        "post_update_block_log_ratio_abs",
        "post_update_block_outside_clip",
    } <= names


def test_single_run_entrypoint_declares_all_required_artifacts():
    script = SINGLE_ANALYZER_PATH.read_text()

    for name in (
        "scalar_alignment.png",
        "scalar_credit.png",
        "scalar_optimization.png",
        "position_heatmaps.png",
        "entropy_segments.png",
        "diagnostics.html",
    ):
        assert name in script
    for label in (
        "Student entropy",
        "Teacher entropy",
        "Top-16 overlap ratio",
        "SignFlipRate",
        "LeakageMagnitude",
    ):
        assert label in script


def test_single_run_report_uses_relative_assets_and_lists_metric_families(tmp_path):
    analyzer = load_single_analyzer()

    analyzer.write_report(
        output_dir=tmp_path,
        label="ml2-A100 Block3",
        record_count=3,
        snapshot_count=2,
        first_step=1,
        last_step=5,
    )

    report = (tmp_path / "diagnostics.html").read_text(encoding="utf-8")
    assert str(tmp_path) not in report
    assert 'src="scalar_alignment.png"' in report
    assert 'src="scalar_credit.png"' in report
    assert 'src="scalar_optimization.png"' in report
    assert 'src="position_heatmaps.png"' in report
    assert 'src="entropy_segments.png"' in report
    for label in (
        "Student entropy",
        "Teacher entropy",
        "Signed entropy gap",
        "Absolute entropy gap",
        "Top-16 overlap ratio",
        "Student/teacher overlap mass",
        "Overlap-token advantage",
        "SignFlipRate / WeightedSignFlipRate",
        "LeakageMagnitude / NormalizedLeakage",
        "Raw-token and block advantage summaries",
        "Post-update block log-ratio",
        "Outside-clip fraction",
        "PG loss and pre-clip grad norm",
        "Response length and truncation ratio",
        "Rollout policy drift",
    ):
        assert label in report

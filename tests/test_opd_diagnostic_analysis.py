import numpy as np

from opd_ext.analysis import (
    bin_position_statistics,
    classify_leading_mechanism,
    classify_ordered_propagation,
    confirmed_chain_onset,
    detect_sustained_onset,
    first_nonfinite_step,
    normalize_hash_manifest_lines,
)


def test_position_binning_uses_weighted_mean_and_max_rollout_coverage():
    statistics = {
        "sum": np.array([2.0, 9.0, 4.0, 0.0]),
        "squared_sum": np.array([2.0, 27.0, 8.0, 0.0]),
        "valid_count": np.array([2, 3, 2, 0]),
    }

    means, counts = bin_position_statistics(statistics, bin_size=2, min_count=2)

    np.testing.assert_allclose(means[:2], np.array([11.0 / 5.0, 2.0]))
    np.testing.assert_array_equal(counts, np.array([3, 2]))


def test_position_binning_masks_bins_without_enough_coverage():
    statistics = {
        "sum": np.array([1.0, 0.0]),
        "squared_sum": np.array([1.0, 0.0]),
        "valid_count": np.array([1, 0]),
    }

    means, counts = bin_position_statistics(statistics, bin_size=1, min_count=2)

    assert np.isnan(means).all()
    np.testing.assert_array_equal(counts, np.array([1, 0]))


def test_one_rollout_cannot_satisfy_eight_rollout_heatmap_threshold():
    statistics = {
        "sum": np.ones(16),
        "squared_sum": np.ones(16),
        "valid_count": np.ones(16, dtype=np.int64),
    }

    means, coverage = bin_position_statistics(statistics, bin_size=16, min_count=8)

    assert np.isnan(means).all()
    np.testing.assert_array_equal(coverage, np.array([1]))


def test_sustained_onset_requires_two_consecutive_diagnostic_points():
    steps = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    values = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 4.0, 1.0, 4.0])

    assert detect_sustained_onset(steps, values, direction="up", baseline_max_step=30) is None

    values[-3:] = [1.0, 4.0, 4.2]
    assert detect_sustained_onset(steps, values, direction="up", baseline_max_step=30) == 40


def test_leading_mechanism_classification_obeys_temporal_rules():
    advantage_first = {
        "sign_flip": 35,
        "leakage": 35,
        "block_ratio": 55,
        "teacher_ood": 60,
        "support_drift": 60,
        "numerical": None,
        "student_entropy": 55,
    }
    assert classify_leading_mechanism(advantage_first) == "advantage_credit_leakage"

    tied = dict(advantage_first, block_ratio=35)
    assert classify_leading_mechanism(tied) == "mixed_mechanism"


def test_confirmed_chain_is_timestamped_at_first_signal_only_when_ordered():
    assert confirmed_chain_onset(35, 40) == 35
    assert confirmed_chain_onset(40, 35) is None
    assert confirmed_chain_onset(None, 40) is None


def test_tail_to_front_requires_strict_temporal_order():
    assert classify_ordered_propagation(front=45, middle=40, tail=35) == "tail_to_front"
    assert (
        classify_ordered_propagation(front=45, middle=35, tail=35)
        == "simultaneous_or_non_tail_to_front"
    )
    assert classify_ordered_propagation(front=None, middle=None, tail=None) == "no_sustained_onset"
    assert (
        classify_ordered_propagation(front=45, middle=None, tail=35)
        == "incomplete_propagation"
    )


def test_nonfinite_scan_includes_pg_loss_and_grad_norm():
    records = [
        {"step": 30, "actor/grad_norm": 1.0, "label": "ok"},
        {"step": 35, "actor/grad_norm": float("nan")},
        {"step": 40, "actor/pg_loss": float("inf")},
    ]

    assert first_nonfinite_step(records) == 35


def test_hash_manifest_normalization_ignores_host_specific_roots():
    train_lines = [
        "aaa  /mnt/data/cpfs/Yaleon/opd/data/math_opd_dapo17k_hf_full_eval4/train.parquet",
        "bbb  /mnt/data/models/Qwen3-1.7B-Base/config.json",
        "ccc  /mnt/data/cpfs/Yaleon/opd/external/revisiting_opd/verl/a.py",
    ]
    ml2_lines = [
        "aaa  /limx/a/opd/data/math_opd_dapo17k_hf_full_eval4/train.parquet",
        "bbb  /limx/models/Qwen3-1.7B-Base/config.json",
        "ccc  /limx/a/opd/external/revisiting_opd/verl/a.py",
    ]

    assert normalize_hash_manifest_lines(train_lines) == normalize_hash_manifest_lines(ml2_lines)

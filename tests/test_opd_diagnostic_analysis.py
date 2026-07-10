import numpy as np

from opd_ext.analysis import (
    bin_position_statistics,
    classify_leading_mechanism,
    detect_sustained_onset,
)


def test_position_binning_uses_sums_and_counts_not_mean_of_means():
    statistics = {
        "sum": np.array([2.0, 9.0, 4.0, 0.0]),
        "squared_sum": np.array([2.0, 27.0, 8.0, 0.0]),
        "valid_count": np.array([2, 3, 2, 0]),
    }

    means, counts = bin_position_statistics(statistics, bin_size=2, min_count=2)

    np.testing.assert_allclose(means[:2], np.array([11.0 / 5.0, 2.0]))
    np.testing.assert_array_equal(counts, np.array([5, 2]))


def test_position_binning_masks_bins_without_enough_coverage():
    statistics = {
        "sum": np.array([1.0, 0.0]),
        "squared_sum": np.array([1.0, 0.0]),
        "valid_count": np.array([1, 0]),
    }

    means, counts = bin_position_statistics(statistics, bin_size=1, min_count=2)

    assert np.isnan(means).all()
    np.testing.assert_array_equal(counts, np.array([1, 0]))


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

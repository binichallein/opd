import numpy as np
import torch

from opd_ext.diagnostics import (
    compute_block_credit_diagnostics,
    compute_block_ratio_diagnostics,
    compute_entropy_diagnostics,
    compute_masked_summary,
    compute_position_statistics,
    compute_topk_alignment_diagnostics,
    merge_position_stat_shards,
    parse_milestone_steps,
    save_diagnostic_snapshot,
    scatter_position_statistics,
    should_save_checkpoint,
    should_run_diagnostics,
)
from opd_ext.analysis import load_scalar_records


def test_block_size_one_has_zero_credit_leakage():
    advantages = torch.tensor([[0.5, -0.25, 1.0]])
    mask = torch.ones_like(advantages)

    result = compute_block_credit_diagnostics(
        advantages=advantages,
        response_mask=mask,
        block_size=1,
        sign_epsilon=1e-4,
    )

    torch.testing.assert_close(result["block_advantage_per_token"], advantages)
    assert result["sign_flip_rate"].item() == 0.0
    assert result["weighted_sign_flip_rate"].item() == 0.0
    assert result["leakage_magnitude"].item() == 0.0
    assert result["normalized_leakage"].item() == 0.0


def test_block_credit_diagnostics_detects_known_sign_flips():
    advantages = torch.tensor([[0.9, -0.3, -0.3]])
    mask = torch.ones_like(advantages)

    result = compute_block_credit_diagnostics(
        advantages=advantages,
        response_mask=mask,
        block_size=3,
        sign_epsilon=1e-4,
    )

    torch.testing.assert_close(
        result["block_advantage_per_token"],
        torch.tensor([[0.1, 0.1, 0.1]]),
    )
    torch.testing.assert_close(result["sign_flip_rate"], torch.tensor(2.0 / 3.0))
    torch.testing.assert_close(result["weighted_sign_flip_rate"], torch.tensor(0.4))
    torch.testing.assert_close(result["leakage_magnitude"], torch.tensor(0.53333336))
    torch.testing.assert_close(result["normalized_leakage"], torch.tensor(1.0666667))


def test_block_credit_diagnostics_respects_mask_and_partial_tail_block():
    advantages = torch.tensor([[1.0, 3.0, -2.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    result = compute_block_credit_diagnostics(
        advantages=advantages,
        response_mask=mask,
        block_size=2,
        sign_epsilon=1e-4,
    )

    torch.testing.assert_close(
        result["block_advantage_per_token"],
        torch.tensor([[2.0, 2.0, -2.0, 0.0]]),
    )
    assert result["eligible_count"].item() == 3
    assert result["sign_flip_count"].item() == 0
    torch.testing.assert_close(result["leakage_magnitude"], torch.tensor(2.0 / 3.0))


def test_topk_alignment_metrics_match_hand_calculation():
    student_ids = torch.tensor([[[1, 2, 3, 4]]])
    teacher_ids = torch.tensor([[[2, 3, 5, 6]]])
    student_log_probs = torch.log(torch.tensor([[[0.4, 0.3, 0.2, 0.1]]]))
    teacher_log_probs = torch.log(torch.tensor([[[0.5, 0.25, 0.15, 0.1]]]))
    mask = torch.ones((1, 1))

    result = compute_topk_alignment_diagnostics(
        student_topk_ids=student_ids,
        student_topk_log_probs=student_log_probs,
        teacher_topk_ids=teacher_ids,
        teacher_topk_log_probs=teacher_log_probs,
        response_mask=mask,
    )

    torch.testing.assert_close(result["overlap_ratio"], torch.tensor([[0.5]]))
    torch.testing.assert_close(result["student_overlap_mass"], torch.tensor([[0.5]]))
    torch.testing.assert_close(result["teacher_overlap_mass"], torch.tensor([[0.75]]))

    p_bar = torch.tensor([0.3, 0.2]) / 0.5
    q_bar = torch.tensor([0.5, 0.25]) / 0.75
    expected_advantage = torch.sum(p_bar * (torch.log(q_bar) - torch.log(p_bar))) / 2
    torch.testing.assert_close(
        result["overlap_token_advantage"],
        expected_advantage.reshape(1, 1),
    )
    assert result["overlap_valid_mask"].item()


def test_position_statistics_excludes_padding_and_preserves_positions():
    values = torch.tensor([[1.0, 2.0, 99.0], [3.0, 4.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    result = compute_position_statistics(values, mask)

    np.testing.assert_allclose(result["sum"], np.array([4.0, 6.0, 5.0]))
    np.testing.assert_allclose(result["squared_sum"], np.array([10.0, 20.0, 25.0]))
    np.testing.assert_array_equal(result["valid_count"], np.array([2, 2, 1]))


def test_checkpoint_milestones_and_final_step_are_saved():
    milestones = {40, 50, 60, 80, 100, 200}
    saved = [
        step
        for step in range(1, 201)
        if should_save_checkpoint(
            global_step=step,
            is_last_step=step == 200,
            milestone_steps=milestones,
        )
    ]

    assert saved == [40, 50, 60, 80, 100, 200]


def test_block_ratio_diagnostics_uses_summed_log_ratio_and_reports_clipping():
    old_log_prob = torch.zeros((1, 4))
    current_log_prob = torch.tensor([[0.1, 0.2, -0.1, -0.2]])
    mask = torch.ones_like(old_log_prob)

    result = compute_block_ratio_diagnostics(
        old_log_prob=old_log_prob,
        current_log_prob=current_log_prob,
        response_mask=mask,
        block_size=2,
        cliprange_low=0.2,
        cliprange_high=0.2,
    )

    torch.testing.assert_close(result["block_log_ratio"], torch.tensor([[0.3, -0.3]]))
    torch.testing.assert_close(result["block_ratio"], torch.exp(torch.tensor([[0.3, -0.3]])))
    assert result["clip_fraction"].item() == 1.0
    assert result["overflow_count"].item() == 0
    assert result["underflow_count"].item() == 0
    assert result["nonfinite_count"].item() == 0


def test_block_ratio_diagnostics_counts_both_exponential_extremes():
    result = compute_block_ratio_diagnostics(
        old_log_prob=torch.zeros((1, 2)),
        current_log_prob=torch.tensor([[90.0, -180.0]]),
        response_mask=torch.ones((1, 2)),
        block_size=1,
        cliprange_low=0.2,
        cliprange_high=0.2,
    )

    assert result["overflow_count"].item() == 1
    assert result["underflow_count"].item() == 1
    assert result["nonfinite_count"].item() == 0


def test_entropy_diagnostics_returns_signed_and_absolute_teacher_minus_student_gap():
    student = torch.tensor([[1.0, 2.0, 99.0]])
    teacher = torch.tensor([[1.5, 1.0, 88.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    result = compute_entropy_diagnostics(student, teacher, mask)

    torch.testing.assert_close(result["signed_gap"], torch.tensor([[0.5, -1.0, 0.0]]))
    torch.testing.assert_close(result["absolute_gap"], torch.tensor([[0.5, 1.0, 0.0]]))
    torch.testing.assert_close(result["student_mean"], torch.tensor(1.5))
    torch.testing.assert_close(result["teacher_mean"], torch.tensor(1.25))
    torch.testing.assert_close(result["signed_gap_mean"], torch.tensor(-0.25))
    torch.testing.assert_close(result["absolute_gap_mean"], torch.tensor(0.75))


def test_diagnostic_snapshot_round_trips_npz_and_jsonl(tmp_path):
    position_stats = {
        "student_entropy": {
            "sum": np.array([1.0, 2.0]),
            "squared_sum": np.array([1.0, 4.0]),
            "valid_count": np.array([2, 1]),
        }
    }

    path = save_diagnostic_snapshot(
        output_dir=tmp_path,
        step=5,
        scalars={"student_entropy": 1.25},
        position_stats=position_stats,
        metadata={"topk": 16, "position_stride": 1},
    )

    assert path.name == "step_00005.npz"
    assert not list(tmp_path.glob("*.tmp"))
    with np.load(path) as data:
        np.testing.assert_allclose(data["student_entropy__sum"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(data["student_entropy__valid_count"], np.array([2, 1]))
        assert data["step"].item() == 5
        assert data["topk"].item() == 16

    records = load_scalar_records(tmp_path / "scalars.jsonl")
    assert records == [
        {
            "position_stride": 1,
            "step": 5,
            "student_entropy": 1.25,
            "topk": 16,
        }
    ]


def test_diagnostic_schedule_includes_first_step_and_interval():
    selected = [
        step
        for step in range(1, 13)
        if should_run_diagnostics(global_step=step, enabled=True, interval=5)
    ]
    assert selected == [1, 5, 10]
    assert not should_run_diagnostics(global_step=5, enabled=False, interval=5)


def test_milestone_parser_rejects_invalid_or_nonpositive_steps():
    assert parse_milestone_steps("40, 50,60,80,100,200") == {40, 50, 60, 80, 100, 200}
    assert parse_milestone_steps("") == set()

    for invalid in ("40,nope", "0,40", "-1,40"):
        try:
            parse_milestone_steps(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid milestones: {invalid}")


def test_position_stat_shards_are_summed_across_workers():
    shards = {
        "sum": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "squared_sum": torch.tensor([[1.0, 4.0], [9.0, 16.0]]),
        "valid_count": torch.tensor([[1, 1], [1, 2]]),
    }

    merged = merge_position_stat_shards(shards)

    np.testing.assert_allclose(merged["sum"], np.array([4.0, 6.0]))
    np.testing.assert_allclose(merged["squared_sum"], np.array([10.0, 20.0]))
    np.testing.assert_array_equal(merged["valid_count"], np.array([2, 3]))


def test_masked_summary_reports_distribution_and_nonfinite_values():
    values = torch.tensor([[1.0, 2.0, float("inf"), 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

    summary = compute_masked_summary(values, mask)

    assert summary["count"].item() == 3
    assert summary["finite_count"].item() == 2
    assert summary["nonfinite_count"].item() == 1
    torch.testing.assert_close(summary["mean"], torch.tensor(1.5))
    torch.testing.assert_close(summary["std"], torch.tensor(0.5))
    torch.testing.assert_close(summary["p95"], torch.tensor(1.95))
    torch.testing.assert_close(summary["max"], torch.tensor(2.0))


def test_scatter_position_statistics_restores_sampled_positions():
    sampled = {
        "sum": np.array([3.0, 7.0]),
        "squared_sum": np.array([5.0, 25.0]),
        "valid_count": np.array([2, 2]),
    }

    full = scatter_position_statistics(
        sampled,
        positions=np.array([0, 3]),
        output_length=5,
    )

    np.testing.assert_allclose(full["sum"], np.array([3.0, 0.0, 0.0, 7.0, 0.0]))
    np.testing.assert_allclose(full["squared_sum"], np.array([5.0, 0.0, 0.0, 25.0, 0.0]))
    np.testing.assert_array_equal(full["valid_count"], np.array([2, 0, 0, 2, 0]))

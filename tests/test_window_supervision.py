"""Independent window oracles; no imports or mutations of the upstream trainer."""

import copy
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def window():
    name = "opd_ext.window_supervision"
    return importlib.import_module(name) if importlib.util.find_spec(name) else SimpleNamespace()


def partitions(mask, block_size, offset):
    """Explicit local-coordinate partition, deliberately independent of scatter IDs."""
    for row, flags in enumerate(mask.tolist()):
        cursor = 0
        while cursor < len(flags):
            if not flags[cursor]:
                cursor += 1
                continue
            end = cursor
            while end < len(flags) and flags[end]:
                end += 1
            if offset:
                stop = min(cursor + offset, end)
                yield row, slice(cursor, stop)
                cursor = stop
            while cursor < end:
                stop = min(cursor + block_size, end)
                yield row, slice(cursor, stop)
                cursor = stop


def ppo_terms(log_ratio, advantage, low=0.2, high=0.2, clipc=3.0):
    ratio = log_ratio.exp()
    raw = -advantage * ratio
    clipped = -advantage * ratio.clamp(1 - low, 1 + high)
    upper = torch.maximum(raw, clipped)
    dual = -advantage * clipc
    return (
        torch.where(advantage < 0, torch.minimum(upper, dual), upper),
        (clipped > raw).to(log_ratio.dtype),
        -log_ratio,
        ((upper > dual) & (advantage < 0)).to(log_ratio.dtype),
    )


def loop_loss(old, current, adv, mask, block_size, offset, low=0.2, high=0.2, clipc=3.0):
    zero = torch.where(mask.bool(), current, 0).sum() * 0
    totals = [zero] * 4
    for row, span in partitions(mask, block_size, offset):
        delta = (current[row, span] - old[row, span].detach()).sum()
        mean_adv = adv[row, span].detach().mean()
        terms = ppo_terms(delta, mean_adv, low, high, clipc)
        totals = [total + term * (span.stop - span.start) for total, term in zip(totals, terms)]
    return tuple(total / mask.sum().clamp(min=1) for total in totals)


def assert_tuple_close(actual, expected, rtol=1e-10, atol=1e-12):
    assert isinstance(actual, tuple) and len(actual) == 4
    for left, right in zip(actual, expected):
        assert left.ndim == 0
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)


def inputs(dtype=torch.float64, active=False):
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    old = torch.full(mask.shape, -4.0, dtype=dtype)
    delta = torch.tensor(
        [
            [0.7, 0.8, 0.9, -0.9, -0.6, -0.7, 0.05, 0.12],
            [0, 1.3, 1.4, 0, -0.5, 0.5, 0.6, 0],
            [0.5, 0.4, 0, 0, 0, 0, 0, 0],
        ],
        dtype=dtype,
    )
    adv = torch.tensor(
        [[-1, -3, -2, 1, 3, 2, -1, 4], [0, 3, 2, 0, -3, 2, -5, 0], [2, 4, 0, 0, 0, 0, 0, 0]],
        dtype=dtype,
    )
    current = (old + delta * (1 if active else 0.02)).requires_grad_()
    return old, current, adv, mask


@pytest.mark.parametrize("block_size", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("active", [False, True])
def test_random_loss_and_gradient_match_loop_on_unequal_regions(window, block_size, active):
    old, current, adv, mask = inputs(active=active)
    for offset in range(block_size):
        actual = window.compute_window_policy_loss(
            old, current, adv, mask, block_size, offset=offset
        )
        expected = loop_loss(old, current, adv, mask, block_size, offset)
        assert_tuple_close(actual, expected)
        actual_grad = torch.autograd.grad(actual[0], current)[0]
        expected_grad = torch.autograd.grad(expected[0], current)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sliding_is_mean_of_complete_phase_losses_and_gradients(window, active, dtype):
    old, current, adv, mask = inputs(dtype, active)
    phases = [
        window.compute_window_policy_loss(old, current, adv, mask, offset=r) for r in range(3)
    ]
    actual = window.compute_window_policy_loss(
        old, current, adv, mask, window_mode="sliding", offset=2
    )
    expected = tuple(torch.stack([phase[i] for phase in phases]).mean() for i in range(4))
    rtol, atol = (1e-10, 1e-12) if dtype == torch.float64 else (1e-5, 1e-6)
    assert_tuple_close(actual, expected, rtol, atol)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0], current)[0],
        torch.autograd.grad(expected[0], current)[0],
        rtol=rtol,
        atol=atol,
    )
    if active:
        assert expected[1] > 0 and expected[3] > 0
    else:
        assert expected[1] == 0 and expected[3] == 0


@pytest.mark.parametrize("block_size", [1, 3])
def test_phase_zero_matches_existing_core_mean_formula(window, block_size):
    old, current, adv, _ = inputs(active=True)
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3, [1] * 2 + [0] * 6], dtype=torch.bool)
    pad = (-mask.shape[1]) % block_size

    def blocks(value):
        return (
            torch.nn.functional.pad(torch.where(mask, value, 0), (0, pad))
            .reshape(mask.shape[0], -1, block_size)
            .sum(-1)
        )

    counts = blocks(mask.to(old.dtype))
    weight = counts / block_size
    terms = ppo_terms(blocks(current) - blocks(old), blocks(adv) / counts.clamp(min=1))
    expected = tuple((term * weight).sum() / weight.sum() for term in terms)
    actual = window.compute_window_policy_loss(old, current, adv, mask, block_size)
    assert_tuple_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0], current)[0],
        torch.autograd.grad(expected[0], current)[0],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ["random", "sliding"])
def test_teacher_old_and_advantage_are_detached_and_padding_nan_has_zero_gradient(window, mode):
    old = torch.tensor([[-2.0, -3.0, float("nan"), -4.0]], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor(
        [[-1.0, -4.0, float("nan"), -2.0]], dtype=torch.float64, requires_grad=True
    )
    current = torch.tensor(
        [[-1.98, -3.01, float("nan"), -4.02]], dtype=torch.float64, requires_grad=True
    )
    adv = teacher - old
    adv.retain_grad()
    mask = torch.tensor([[1, 1, 0, 1]])
    result = window.compute_window_policy_loss(old, current, adv, mask, window_mode=mode)
    assert all(torch.isfinite(value) for value in result)
    result[0].backward()
    assert teacher.grad is None and old.grad is None and adv.grad is None
    assert torch.isfinite(current.grad).all() and current.grad[0, 2] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_arithmetic_dtype_and_gradient_are_preserved_or_promoted(window, dtype):
    old, current, adv, mask = inputs(dtype)
    result = window.compute_window_policy_loss(old, current, adv, mask, window_mode="sliding")
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert all(value.dtype == expected_dtype for value in result)
    expected = window.compute_window_policy_loss(
        old.to(expected_dtype),
        current.detach().to(expected_dtype),
        adv.to(expected_dtype),
        mask,
        window_mode="sliding",
    )
    assert_tuple_close(result, expected)
    result[0].backward()
    assert current.grad.dtype == dtype and torch.isfinite(current.grad).all()


def test_log_ratio_subtraction_precedes_summation(window):
    old = torch.full((1, 3), -1e8, dtype=torch.float32)
    current = (old + 8).requires_grad_()
    adv, mask = torch.ones_like(old), torch.ones_like(old)
    stable = (current - old).sum()
    historical = current.sum() - old.sum()
    assert stable.item() == 24 and historical.item() == 32
    actual = window.compute_window_policy_loss(old, current, adv, mask)
    assert actual[2].item() == -24
    assert_tuple_close(actual, loop_loss(old, current, adv, mask, 3, 0))


@pytest.mark.parametrize("shape", [(2, 4), (2, 0), (0, 3)])
def test_empty_masks_return_differentiable_zero(window, shape):
    current = torch.full(shape, float("nan"), dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(shape, dtype=torch.bool)
    result = window.compute_window_policy_loss(
        current, current, current, mask, window_mode="sliding"
    )
    assert all(value.item() == 0 for value in result)
    result[0].backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0},
        {"block_size": -1},
        {"block_size": 1.5},
        {"block_size": True},
        {"window_mode": "fixed"},
        {"offset": -1},
        {"offset": 3},
        {"offset": 1.0},
        {"offset": [1]},
        {"offset": torch.tensor([1])},
        {"offset": True},
        {"window_mode": "sliding", "offset": 3},
        {"loss_agg_mode": "seq-mean-token-mean"},
        {"cliprange": -0.1},
        {"cliprange": float("nan")},
        {"cliprange_low": 1.1},
        {"cliprange_high": -0.1},
        {"cliprange_high": float("inf")},
        {"clip_ratio_c": 1.0},
        {"clip_ratio_c": float("nan")},
        {"clip_ratio_c": float("inf")},
    ],
)
def test_loss_rejects_invalid_configuration(window, kwargs):
    with pytest.raises(ValueError):
        window.compute_window_policy_loss(*inputs(), **kwargs)


@pytest.mark.parametrize("which", range(4))
@pytest.mark.parametrize("mutation", ["rank", "shape", "nonfinite"])
def test_loss_rejects_invalid_tensors(window, which, mutation):
    args = list(inputs())
    if mutation == "rank":
        args[which] = args[which].flatten()
    elif mutation == "shape":
        args[which] = args[which][:, :-1]
    else:
        args[which] = args[which].detach().to(torch.float64).clone()
        args[which][0, 0] = float("nan")
    with pytest.raises(ValueError):
        window.compute_window_policy_loss(*args)


def test_loss_rejects_nonbinary_mask(window):
    old, current, adv, mask = inputs()
    mask = mask.to(old.dtype)
    mask[0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        window.compute_window_policy_loss(old, current, adv, mask)


@pytest.mark.parametrize("kind", ["ratio", "advantage_sum", "log_ratio", "loss"])
def test_loss_rejects_nonfinite_intermediates(window, kind):
    old = torch.zeros((1, 3), dtype=torch.float32)
    current, adv = old.clone(), torch.ones_like(old)
    if kind == "ratio":
        current[:] = 40
    elif kind == "advantage_sum":
        adv[:] = torch.finfo(old.dtype).max
    elif kind == "log_ratio":
        current[:] = torch.finfo(old.dtype).max
        old[:] = -torch.finfo(old.dtype).max
    else:
        current[:] = 1
        adv[:] = -2e37
    with pytest.raises(ValueError, match="finite"):
        window.compute_window_policy_loss(old, current, adv, torch.ones_like(old))


def test_asymmetric_clipping_and_explicit_clip_fallbacks(window):
    old, current, adv, mask = inputs(active=True)
    expected = loop_loss(old, current, adv, mask, 3, 1, 0.1, 0.4, 2)
    actual = window.compute_window_policy_loss(
        old,
        current,
        adv,
        mask,
        offset=1,
        cliprange=None,
        cliprange_low=0.1,
        cliprange_high=0.4,
        clip_ratio_c=2,
    )
    assert_tuple_close(actual, expected)


def loop_credit(adv, mask, block_size, mode, offset):
    means, sums = torch.zeros_like(adv), torch.zeros_like(adv)
    coverage = torch.zeros_like(mask, dtype=torch.int64)
    phases = range(block_size) if mode == "sliding" else (offset,)
    for phase in phases:
        for row, span in partitions(mask, block_size, phase):
            means[row, span] += adv[row, span].mean()
            sums[row, span] += adv[row, span].sum()
            coverage[row, span] += 1
    return means / len(phases), sums / len(phases), coverage


@pytest.mark.parametrize("length", range(1, 9))
@pytest.mark.parametrize("block_size", [1, 2, 3, 5, 10])
def test_all_lengths_offsets_and_gaps_have_exact_credit_and_coverage(window, length, block_size):
    # Every length occurs twice, with leading padding and a gap to force local resets.
    mask = torch.tensor([[0] + [1] * length + [0, 0] + [1] * length + [0]])
    adv = torch.arange(mask.numel(), dtype=torch.float64).reshape_as(mask).square() - 7
    adv[mask == 0] = float("nan")
    for mode, offset in [("random", r) for r in range(block_size)] + [("sliding", 0)]:
        result = window.window_credit_diagnostics(adv, mask, block_size, mode, offset)
        means, sums, coverage = loop_credit(adv, mask, block_size, mode, offset)
        torch.testing.assert_close(
            result["block_advantage_per_token"], means, rtol=1e-10, atol=1e-12
        )
        torch.testing.assert_close(
            result["effective_token_coefficient"], sums, rtol=1e-10, atol=1e-12
        )
        torch.testing.assert_close(result["coverage_per_token"], coverage)


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0, [2, 2, 2, 5, 5, 5]),
        (1, [1, 3, 3, 3, 5.5, 5.5]),
        (2, [1.5, 1.5, 4, 4, 4, 6]),
    ],
)
def test_agreed_six_token_phases(window, offset, expected):
    adv = torch.arange(1, 7, dtype=torch.float64).reshape(1, -1)
    result = window.window_credit_diagnostics(adv, torch.ones_like(adv), offset=offset)
    torch.testing.assert_close(result["block_advantage_per_token"], adv.new_tensor([expected]))


def test_credit_preserves_all_historical_keys_and_phase_zero_values(window):
    from opd_ext.diagnostics import compute_block_credit_diagnostics

    adv = torch.tensor([[0.9, -0.3, -0.3, 1, 3, float("nan")]], dtype=torch.float64)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    historical = compute_block_credit_diagnostics(adv, mask, 3)
    result = window.window_credit_diagnostics(adv, mask)
    assert set(result) == set(historical) | {"effective_token_coefficient", "coverage_per_token"}
    for name, value in historical.items():
        torch.testing.assert_close(result[name], value, rtol=1e-10, atol=1e-12)


def test_sliding_credit_statistics_use_mean_view_not_true_coefficient(window):
    adv = torch.tensor([[1.0, -4.0, 2.0, 6.0, -8.0]], dtype=torch.float64)
    mask = torch.ones_like(adv, dtype=torch.bool)
    means, sums, _ = loop_credit(adv, mask, 3, "sliding", 0)
    assert not torch.allclose(means, sums)
    result = window.window_credit_diagnostics(adv, mask, window_mode="sliding")
    eligible = (adv.abs() > 1e-4) & (means.abs() > 1e-4)
    flips = eligible & (adv.sign() != means.sign())
    leakage = (adv - means).abs()
    expected = {
        "eligible_mask": eligible,
        "sign_flip_mask": flips,
        "eligible_count": eligible.sum(),
        "sign_flip_count": flips.sum(),
        "sign_flip_rate": flips.sum().to(adv.dtype) / eligible.sum(),
        "weighted_sign_flip_rate": adv[flips].abs().sum() / adv[eligible].abs().sum(),
        "leakage_per_token": leakage,
        "leakage_magnitude": leakage.mean(),
        "normalized_leakage": leakage.sum() / adv.abs().sum(),
    }
    for key, value in expected.items():
        torch.testing.assert_close(result[key], value, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("mode", ["random", "sliding"])
def test_effective_coefficient_is_unclipped_loss_derivative_at_unit_ratio(window, mode):
    old, _, adv, mask = inputs()
    current = old.clone().requires_grad_()
    result = window.window_credit_diagnostics(adv, mask, window_mode=mode, offset=1)
    loss = window.compute_window_policy_loss(old, current, adv, mask, window_mode=mode, offset=1)[0]
    gradient = torch.autograd.grad(loss, current)[0]
    torch.testing.assert_close(
        result["effective_token_coefficient"], -gradient * mask.sum(), rtol=1e-10, atol=1e-12
    )


RATIO_SCALARS = {
    "clip_fraction",
    "overflow_count",
    "underflow_count",
    "nonfinite_count",
    "log_ratio_abs_mean",
    "log_ratio_abs_p95",
    "log_ratio_abs_max",
    "ratio_p95",
    "ratio_max",
}
RATIO_TOKENS = {
    "token_log_ratio_abs",
    "token_finite_mask",
    "token_outside_clip",
    "coverage_per_token",
}


@pytest.mark.parametrize("mode", ["random", "sliding"])
def test_ratio_scalars_pool_windows_and_token_views_match_oracle(window, mode):
    old, current, _, mask = inputs(active=True)
    phases = range(3) if mode == "sliding" else (1,)
    logs = []
    per_token = torch.zeros_like(old)
    outside = torch.zeros_like(mask)
    for phase in phases:
        for row, span in partitions(mask, 3, phase):
            delta = (current[row, span] - old[row, span]).sum().detach()
            logs.append(delta)
            per_token[row, span] += delta.abs() / len(phases)
            outside[row, span] |= (delta.exp() < 0.8) | (delta.exp() > 1.2)
    logs = torch.stack(logs)
    ratios = logs.exp()
    result = window.window_ratio_diagnostics(old, current, mask, window_mode=mode, offset=1)
    assert set(result) == RATIO_SCALARS | RATIO_TOKENS
    expected = {
        "log_ratio_abs_mean": logs.abs().mean(),
        "log_ratio_abs_p95": logs.abs().quantile(0.95),
        "log_ratio_abs_max": logs.abs().max(),
        "ratio_p95": ratios.quantile(0.95),
        "ratio_max": ratios.max(),
        "clip_fraction": ((ratios < 0.8) | (ratios > 1.2)).to(old.dtype).mean(),
        "token_log_ratio_abs": per_token,
        "token_finite_mask": mask,
        "token_outside_clip": outside,
        "coverage_per_token": mask.to(torch.int64) * len(phases),
    }
    for name, value in expected.items():
        torch.testing.assert_close(result[name], value, rtol=1e-10, atol=1e-12)
    for name in ("overflow_count", "underflow_count", "nonfinite_count"):
        assert result[name].item() == 0


def test_ratio_phase_zero_scalar_compatibility(window):
    from opd_ext.diagnostics import compute_block_ratio_diagnostics

    old, current, _, _ = inputs(torch.float32, active=True)
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3, [1] * 2 + [0] * 6])
    expected = compute_block_ratio_diagnostics(old, current, mask, 3, 0.2, 0.2)
    result = window.window_ratio_diagnostics(old, current, mask)
    assert {key for key, value in expected.items() if value.ndim == 0} == RATIO_SCALARS
    for name in RATIO_SCALARS:
        torch.testing.assert_close(result[name], expected[name])


def test_ratio_reports_nonfinite_windows_extremes_and_masks_padding(window):
    old = torch.zeros((1, 6), dtype=torch.float64)
    current = torch.tensor([[float("nan"), 0.5, 0.5, 90, -180, float("nan")]], dtype=old.dtype)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    single = window.window_ratio_diagnostics(old, current, mask, block_size=1)
    assert single["nonfinite_count"] == 1
    assert single["overflow_count"] == 1
    assert single["underflow_count"] == 1
    assert single["ratio_max"] == old.new_tensor(80.0).exp()
    result = window.window_ratio_diagnostics(old, current, mask, window_mode="sliding")
    assert result["nonfinite_count"] == 3
    torch.testing.assert_close(
        result["token_finite_mask"], torch.tensor([[0, 0, 0, 1, 1, 0]]).bool()
    )
    torch.testing.assert_close(
        result["token_outside_clip"], torch.tensor([[0, 1, 1, 1, 1, 0]]).bool()
    )
    assert torch.isfinite(result["token_log_ratio_abs"]).all()
    assert (result["token_log_ratio_abs"][~result["token_finite_mask"]] == 0).all()
    torch.testing.assert_close(result["coverage_per_token"], mask * 3)


@pytest.mark.parametrize("shape", [(2, 4), (2, 0), (0, 3)])
def test_empty_diagnostics_are_finite_zeros(window, shape):
    value = torch.full(shape, float("nan"), dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(shape)
    for result in (
        window.window_credit_diagnostics(value, mask, window_mode="sliding"),
        window.window_ratio_diagnostics(value, value, mask, window_mode="sliding"),
    ):
        for tensor in result.values():
            assert torch.isfinite(tensor).all() and (tensor == 0).all()
            assert not tensor.requires_grad


@pytest.mark.parametrize("name", ["window_credit_diagnostics", "window_ratio_diagnostics"])
@pytest.mark.parametrize("invalid", ["rank", "nonbinary", "shape", "mode", "offset", "block_size"])
def test_diagnostics_validate_structure(window, name, invalid):
    old, current, adv, mask = inputs()
    kwargs = {}
    if invalid == "rank":
        mask = mask.flatten()
    elif invalid == "nonbinary":
        mask = mask.to(old.dtype) * 0.5
    elif invalid == "shape":
        mask = mask[:, :-1]
    else:
        kwargs[{"mode": "window_mode", "offset": "offset", "block_size": "block_size"}[invalid]] = {
            "mode": "fixed",
            "offset": 3,
            "block_size": 0,
        }[invalid]
    args = (adv, mask) if name == "window_credit_diagnostics" else (old, current, mask)
    with pytest.raises(ValueError):
        getattr(window, name)(*args, **kwargs)


@pytest.mark.parametrize("epsilon", [0, -1, float("nan"), float("inf")])
def test_credit_rejects_invalid_epsilon(window, epsilon):
    _, _, adv, mask = inputs()
    with pytest.raises(ValueError):
        window.window_credit_diagnostics(adv, mask, sign_epsilon=epsilon)


def test_credit_rejects_valid_nonfinite_advantages(window):
    _, _, adv, mask = inputs()
    adv[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        window.window_credit_diagnostics(adv, mask)


def assert_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_schedule_draws_exact_private_pcg64_sequence_without_global_rng_mutation(window):
    numpy_before, torch_before = np.random.get_state(), torch.random.get_rng_state().clone()
    schedule = window.OffsetSchedule(3, 910021)
    expected = np.random.Generator(np.random.PCG64(910021))
    assert [schedule.next(step) for step in range(1, 101)] == [
        int(expected.integers(3)) for _ in range(100)
    ]
    assert_rng_equal(numpy_before, np.random.get_state())
    assert torch.equal(torch_before, torch.random.get_rng_state())


def test_schedule_json_state_is_flat_independent_and_resumes_exactly(window):
    schedule = window.OffsetSchedule(block_size=3, seed=910021)
    initial = schedule.state_dict()
    assert set(initial) == {"block_size", "seed", "last_step", "current_offset", "rng_state"}
    assert initial["block_size"] == 3 and initial["seed"] == 910021
    assert initial["last_step"] == 0 and initial["current_offset"] is None
    for step in range(1, 8):
        offset = schedule.next(step)
    saved = json.loads(json.dumps(schedule.state_dict()))
    assert saved["last_step"] == 7 and saved["current_offset"] == offset
    resumed = window.OffsetSchedule(3, 910021)
    resumed.load_state_dict(saved)
    assert resumed.state_dict() == saved
    saved["rng_state"]["state"]["state"] = 0
    assert resumed.state_dict() == schedule.state_dict()
    snapshot = resumed.state_dict()
    snapshot["rng_state"]["state"]["state"] = 0
    assert resumed.state_dict() == schedule.state_dict()
    assert [resumed.next(step) for step in range(8, 51)] == [
        schedule.next(step) for step in range(8, 51)
    ]
    fresh = window.OffsetSchedule(3, 910021)
    fresh.load_state_dict(initial)
    assert fresh.next(1) == window.OffsetSchedule(3, 910021).next(1)


@pytest.mark.parametrize("step", [0, 2, -1, 1.0, True, [1]])
def test_schedule_rejects_nonsequential_initial_steps_without_advancing(window, step):
    schedule = window.OffsetSchedule(3, 21)
    before = schedule.state_dict()
    with pytest.raises(ValueError):
        schedule.next(step)
    assert schedule.state_dict() == before


def test_schedule_rejects_repeated_skipped_and_resumed_steps(window):
    schedule = window.OffsetSchedule(3, 21)
    schedule.next(1)
    resumed = window.OffsetSchedule(3, 21)
    resumed.load_state_dict(schedule.state_dict())
    for step in (1, 3, 0):
        before = resumed.state_dict()
        with pytest.raises(ValueError):
            resumed.next(step)
        assert resumed.state_dict() == before
    assert resumed.next(2) == schedule.next(2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0, "seed": 21},
        {"block_size": 3.0, "seed": 21},
        {"block_size": True, "seed": 21},
        {"block_size": 3, "seed": -1},
        {"block_size": 3, "seed": 1.0},
        {"block_size": 3, "seed": True},
    ],
)
def test_schedule_rejects_invalid_constructor(window, kwargs):
    with pytest.raises(ValueError):
        window.OffsetSchedule(**kwargs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("seed", 22),
        ("block_size", 4),
        ("last_step", -1),
        ("last_step", 1.5),
        ("last_step", 0),
        ("current_offset", None),
        ("current_offset", 3),
        ("current_offset", True),
        ("rng_state", None),
        ("rng_state", {}),
        ("rng_state", {"bit_generator": "MT19937"}),
    ],
)
def test_schedule_load_rejects_invalid_state_atomically(window, key, value):
    schedule = window.OffsetSchedule(3, 21)
    schedule.next(1)
    before = schedule.state_dict()
    invalid = copy.deepcopy(before)
    invalid[key] = value
    with pytest.raises(ValueError):
        schedule.load_state_dict(invalid)
    assert schedule.state_dict() == before


def test_schedule_rejects_missing_fields_and_invalid_initial_offset(window):
    schedule = window.OffsetSchedule(3, 21)
    for key in schedule.state_dict():
        state = schedule.state_dict()
        del state[key]
        with pytest.raises(ValueError):
            schedule.load_state_dict(state)
    state = schedule.state_dict()
    state["current_offset"] = 0
    with pytest.raises(ValueError):
        schedule.load_state_dict(state)


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("block_size", [1, 3])
def test_phase_diagnostics_are_scalar_loss_input_gradient_statistics(window, active, block_size):
    old, current, adv, mask = inputs(active=active)
    if not active:
        current = old.clone().requires_grad_()
    losses = [loop_loss(old, current, adv, mask, block_size, r)[0] for r in range(block_size)]
    gradients = torch.stack([torch.autograd.grad(loss, current)[0] for loss in losses])
    mean = gradients.mean(0)
    norms = gradients.flatten(1).norm(dim=1)
    expected = {
        "phase_disagreement_squared_norm": (gradients - mean).square().flatten(1).sum(1).mean(),
        "mean_phase_gradient_norm": norms.mean(),
        "mean_gradient_norm": mean.norm(),
        "loss_identity_error": old.new_zeros(()),
        "gradient_identity_error": old.new_zeros(()),
    }
    for phase in range(block_size):
        expected[f"phase_{phase}_loss"] = losses[phase].detach()
        expected[f"phase_{phase}_gradient_norm"] = norms[phase]
        for other in range(phase + 1, block_size):
            product = (gradients[phase] * gradients[other]).sum()
            expected[f"phase_{phase}_{other}_cosine"] = product / (norms[phase] * norms[other])
            expected[f"phase_{phase}_{other}_cosine_valid"] = torch.tensor(True)
    with torch.no_grad():
        result = window.window_phase_diagnostics(old, current, adv, mask, block_size=block_size)
    assert set(result) == set(expected)
    for key, value in expected.items():
        assert isinstance(result[key], torch.Tensor) and result[key].ndim == 0
        assert not result[key].requires_grad
        torch.testing.assert_close(result[key], value, rtol=1e-10, atol=1e-12)
    assert current.grad is None


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode, torch.enable_grad])
def test_diagnostics_are_pure_do_not_backpropagate_to_model_or_change_rng(window, context):
    old, _, adv, mask = inputs(active=True)
    parameter = torch.nn.Parameter(old.clone())
    parameter.grad = torch.full_like(parameter, 7)
    current = parameter * 0.99
    old.requires_grad_()
    adv.requires_grad_()
    original = [value.detach().clone() for value in (old, current, adv, mask)]
    numpy_before, torch_before = np.random.get_state(), torch.random.get_rng_state().clone()
    baseline = window.compute_window_policy_loss(old, current, adv, mask, window_mode="sliding")[0]
    baseline_grad = torch.autograd.grad(baseline, parameter, retain_graph=True)[0]
    with context():
        results = [
            window.window_credit_diagnostics(adv, mask, window_mode="sliding"),
            window.window_ratio_diagnostics(old, current, mask, window_mode="sliding"),
            window.window_phase_diagnostics(old, current, adv, mask),
        ]
    for result in results:
        assert all(not value.requires_grad for value in result.values())
    after = window.compute_window_policy_loss(old, current, adv, mask, window_mode="sliding")[0]
    after_grad = torch.autograd.grad(after, parameter)[0]
    torch.testing.assert_close(after, baseline)
    torch.testing.assert_close(after_grad, baseline_grad)
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7))
    assert old.grad is None and adv.grad is None
    for value, before in zip((old, current, adv, mask), original):
        torch.testing.assert_close(value, before)
    assert_rng_equal(numpy_before, np.random.get_state())
    assert torch.equal(torch_before, torch.random.get_rng_state())


def test_phase_diagnostics_empty_mask_zero_norms_and_custom_clips(window):
    old, current, adv, mask = inputs(active=True)
    custom = window.window_phase_diagnostics(
        old, current, adv, mask, clip_low=0.1, clip_high=0.4, clipc=2
    )
    for phase in range(3):
        expected = loop_loss(old, current, adv, mask, 3, phase, 0.1, 0.4, 2)[0]
        torch.testing.assert_close(custom[f"phase_{phase}_loss"], expected.detach())
    empty = window.window_phase_diagnostics(old, current, adv, torch.zeros_like(mask))
    assert all(value.item() == 0 for value in empty.values())


def test_phase_cosine_valid_distinguishes_undefined_zero_from_defined_cosine(window):
    old = torch.zeros((1, 3), dtype=torch.float64)
    adv = torch.tensor([[1.0, -2.0, 1.0]], dtype=old.dtype)
    mask = torch.ones_like(old)
    result = window.window_phase_diagnostics(old, old, adv, mask)
    for pair in ("0_1", "0_2"):
        assert result[f"phase_{pair}_cosine_valid"].item() is False
        assert result[f"phase_{pair}_cosine"].item() == 0
    assert result["phase_1_2_cosine_valid"].item() is True
    assert all(value.ndim == 0 and not value.requires_grad for value in result.values())
    zero = window.window_phase_diagnostics(old, old, torch.zeros_like(adv), mask)
    for pair in ("0_1", "0_2", "1_2"):
        assert zero[f"phase_{pair}_cosine_valid"].item() is False
        assert zero[f"phase_{pair}_cosine"].item() == 0


def test_sliding_loss_rejects_nonfinite_final_phase_reduction(window):
    old = torch.zeros((1, 1), dtype=torch.float32)
    adv = torch.full_like(old, -2e38)
    mask = torch.ones_like(old)
    phase = window.compute_window_policy_loss(old, old, adv, mask, clip_ratio_c=1.1)
    assert torch.isfinite(phase[0])
    with pytest.raises(ValueError, match="finite"):
        window.compute_window_policy_loss(
            old, old, adv, mask, window_mode="sliding", clip_ratio_c=1.1
        )


@pytest.mark.parametrize("mode", ["random", "sliding"])
def test_mixed_dtypes_preserve_fp64_and_noncontiguous_inputs(window, mode):
    old, current, adv, mask = inputs()
    old = old[:, ::2].to(torch.float32)
    current = current.detach()[:, ::2].requires_grad_()
    adv, mask = adv[:, ::2], mask[:, ::2]
    assert not current.is_contiguous()
    actual = window.compute_window_policy_loss(old, current, adv, mask, window_mode=mode)
    phases = range(3) if mode == "sliding" else (0,)
    expected_phases = [loop_loss(old, current, adv, mask, 3, r) for r in phases]
    expected = tuple(torch.stack([phase[i] for phase in expected_phases]).mean() for i in range(4))
    assert_tuple_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual[0], current)[0],
        torch.autograd.grad(expected[0], current)[0],
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("which", [0, 1, 2])
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_loss_rejects_valid_infinities_but_ignores_padding_infinities(window, which, value):
    args = list(inputs())
    args[which] = args[which].detach().clone()
    args[which][0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        window.compute_window_policy_loss(*args)
    args[3][0, 0] = False
    result = window.compute_window_policy_loss(*args)
    assert all(torch.isfinite(tensor) for tensor in result)


def test_schedule_load_wrapped_driver_json_does_not_consume_global_rng(window):
    numpy_before, torch_before = np.random.get_state(), torch.random.get_rng_state().clone()
    schedule = window.OffsetSchedule(3, 910021)
    schedule.next(1)
    wrapped = json.loads(json.dumps({"mode": "random", **schedule.state_dict()}))
    resumed = window.OffsetSchedule(3, 910021)
    resumed.load_state_dict(wrapped)
    assert schedule.next(2) == resumed.next(2)
    assert_rng_equal(numpy_before, np.random.get_state())
    assert torch.equal(torch_before, torch.random.get_rng_state())

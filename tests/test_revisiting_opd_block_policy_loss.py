import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CORE_ALGOS = ROOT / "external" / "revisiting_opd" / "verl" / "trainer" / "ppo" / "core_algos.py"

torch_functional = types.SimpleNamespace(masked_mean=lambda x, mask: (x * mask).sum() / mask.sum())
sys.modules.setdefault("verl", types.ModuleType("verl"))
sys.modules.setdefault("verl.utils", types.ModuleType("verl.utils"))
sys.modules["verl.utils.torch_functional"] = torch_functional

spec = importlib.util.spec_from_file_location("revisiting_opd_core_algos", CORE_ALGOS)
core_algos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_algos)
aggregate_blockwise_policy_inputs = core_algos.aggregate_blockwise_policy_inputs
compute_policy_loss_gspo = core_algos.compute_policy_loss_gspo
compute_policy_loss = core_algos.compute_policy_loss
validate_block_opd_compatibility = core_algos.validate_block_opd_compatibility


def test_blockwise_policy_inputs_sum_logprob_and_mean_advantage():
    old_log_prob = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -99.0]])
    log_prob = torch.tensor([[-0.9, -2.1, -2.8, -4.2, -99.0]])
    advantages = torch.tensor([[1.0, 3.0, 5.0, 7.0, 99.0]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0]])

    block_old, block_log, block_adv, block_mask = aggregate_blockwise_policy_inputs(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        block_size=2,
        block_advantage_mode="mean",
    )

    torch.testing.assert_close(block_old, torch.tensor([[-3.0, -7.0, 0.0]]))
    torch.testing.assert_close(block_log, torch.tensor([[-3.0, -7.0, 0.0]]))
    torch.testing.assert_close(block_adv, torch.tensor([[2.0, 6.0, 0.0]]))
    torch.testing.assert_close(block_mask, torch.tensor([[1.0, 1.0, 0.0]]))


def test_blockwise_policy_inputs_mixed_interpolates_sum_and_mean():
    old_log_prob = torch.zeros((1, 3))
    log_prob = torch.zeros((1, 3))
    advantages = torch.tensor([[2.0, 4.0, 8.0]])
    response_mask = torch.ones((1, 3))

    _, _, block_adv, block_mask = aggregate_blockwise_policy_inputs(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        block_size=3,
        block_advantage_mode="mixed",
        block_mix_lambda=0.5,
    )

    # sum=14, mean=14/3, mixed=0.5*sum + 0.5*mean
    torch.testing.assert_close(block_adv, torch.tensor([[(14.0 + 14.0 / 3.0) / 2.0]]))
    torch.testing.assert_close(block_mask, torch.tensor([[1.0]]))


def test_partial_block_uses_fractional_mask_weight():
    old_log_prob = torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0]])
    log_prob = torch.tensor([[-1.1, -1.9, -3.2, -3.8, -5.1]])
    advantages = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
    response_mask = torch.ones((1, 5))

    block_old, block_log, block_adv, block_mask = aggregate_blockwise_policy_inputs(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        block_size=2,
        block_advantage_mode="mean",
    )

    torch.testing.assert_close(block_old, torch.tensor([[-3.0, -7.0, -5.0]]))
    torch.testing.assert_close(block_log, torch.tensor([[-3.0, -7.0, -5.1]]))
    torch.testing.assert_close(block_adv, torch.tensor([[2.0, 6.0, 9.0]]))
    torch.testing.assert_close(block_mask, torch.tensor([[1.0, 1.0, 0.5]]))


def test_masked_invalid_logprobs_do_not_create_nan_blocks():
    old_log_prob = torch.tensor([[-1.0, float("-inf")]])
    log_prob = torch.tensor([[-1.1, float("-inf")]])
    advantages = torch.tensor([[2.0, 99.0]])
    response_mask = torch.tensor([[1.0, 0.0]])

    block_old, block_log, block_adv, block_mask = aggregate_blockwise_policy_inputs(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        block_size=2,
        block_advantage_mode="mean",
    )

    assert torch.isfinite(block_old).all()
    assert torch.isfinite(block_log).all()
    assert torch.isfinite(block_adv).all()
    torch.testing.assert_close(block_old, torch.tensor([[-1.0]]))
    torch.testing.assert_close(block_log, torch.tensor([[-1.1]]))
    torch.testing.assert_close(block_adv, torch.tensor([[2.0]]))
    torch.testing.assert_close(block_mask, torch.tensor([[0.5]]))


def test_gspo_rejects_block_opd_with_clear_error():
    values = torch.zeros((1, 2))
    mask = torch.ones((1, 2))

    try:
        compute_policy_loss_gspo(
            old_log_prob=values,
            log_prob=values,
            advantages=values,
            response_mask=mask,
            cliprange=0.2,
            opd_block_size=2,
        )
    except ValueError as exc:
        assert "Block OPD" in str(exc)
    else:
        raise AssertionError("GSPO accepted block OPD settings")


def test_full_kl_rejects_block_opd_with_clear_error():
    try:
        validate_block_opd_compatibility(
            block_size=2,
            loss_mode="vanilla",
            use_full_kl=True,
        )
    except ValueError as exc:
        assert "full/top-k KL" in str(exc)
    else:
        raise AssertionError("full/top-k KL accepted block OPD settings")


def test_diagnostic_reads_do_not_change_block10_loss_gradient_or_rng():
    from opd_ext.diagnostics import (
        compute_block_credit_diagnostics,
        compute_block_ratio_diagnostics,
    )

    old_log_prob = torch.tensor([[-1.0, -1.1, -1.2, -1.3, -1.4]])
    advantages = torch.tensor([[0.5, -0.2, 0.7, -0.1, 0.3]])
    response_mask = torch.ones_like(old_log_prob)

    def loss_and_gradient(with_diagnostics: bool):
        current_log_prob = torch.tensor(
            [[-0.98, -1.12, -1.18, -1.31, -1.39]], requires_grad=True
        )
        rng_before = torch.random.get_rng_state().clone()
        if with_diagnostics:
            compute_block_credit_diagnostics(
                advantages=advantages,
                response_mask=response_mask,
                block_size=10,
            )
            compute_block_ratio_diagnostics(
                old_log_prob=old_log_prob,
                current_log_prob=current_log_prob,
                response_mask=response_mask,
                block_size=10,
                cliprange_low=0.2,
                cliprange_high=0.2,
            )
        rng_after = torch.random.get_rng_state().clone()
        loss, *_ = compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=current_log_prob,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.2,
            opd_block_size=10,
            opd_block_advantage_mode="mean",
        )
        loss.backward()
        return loss.detach(), current_log_prob.grad.detach(), rng_before, rng_after

    baseline_loss, baseline_grad, _, _ = loss_and_gradient(False)
    diagnostic_loss, diagnostic_grad, rng_before, rng_after = loss_and_gradient(True)

    torch.testing.assert_close(diagnostic_loss, baseline_loss)
    torch.testing.assert_close(diagnostic_grad, baseline_grad)
    assert torch.equal(rng_before, rng_after)

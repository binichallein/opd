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

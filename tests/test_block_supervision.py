import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clean_opd_train import build_supervision_units


def test_block_size_one_matches_dense_token_units():
    raw_advantage = torch.tensor([0.5, -0.25, 1.0])
    lambdas = torch.tensor([1.0, 1.25, 0.75])
    weights = torch.tensor([1.0, 0.5, 2.0])
    current_logp = torch.tensor([-2.0, -3.0, -4.0])
    signals = {"overlap": torch.tensor([0.1, 0.2, 0.3])}

    units = build_supervision_units(
        weights=weights,
        raw_advantage=raw_advantage,
        lambdas=lambdas,
        signals=signals,
        current_logp=current_logp,
        token_supervision_stride=1,
        token_supervision_offset=0,
        block_size=1,
    )

    torch.testing.assert_close(units["weights"], weights)
    torch.testing.assert_close(units["raw_advantage"], raw_advantage)
    torch.testing.assert_close(units["advantage"], raw_advantage * lambdas)
    torch.testing.assert_close(units["current_logp"], current_logp)
    torch.testing.assert_close(units["signals"]["overlap"], signals["overlap"])
    assert units["supervised_tokens"] == 3
    assert units["supervision_units"] == 3


def test_block_size_two_uses_non_overlapping_logprob_sums_and_drops_tail():
    raw_advantage = torch.tensor([0.5, 0.25, -0.5, 1.0, 99.0])
    lambdas = torch.tensor([1.0, 2.0, 3.0, 4.0, 99.0])
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 99.0])
    current_logp = torch.tensor([-1.0, -2.0, -3.0, -4.0, -99.0])
    reference_logp = torch.tensor([-0.5, -1.5, -2.5, -3.5, -99.0])
    signals = {"overlap": torch.tensor([0.2, 0.4, 0.6, 0.8, 99.0])}

    units = build_supervision_units(
        weights=weights,
        raw_advantage=raw_advantage,
        lambdas=lambdas,
        signals=signals,
        current_logp=current_logp,
        reference_logp=reference_logp,
        token_supervision_stride=1,
        token_supervision_offset=0,
        block_size=2,
    )

    torch.testing.assert_close(units["weights"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(units["raw_advantage"], torch.tensor([0.75, 0.5]))
    torch.testing.assert_close(units["advantage"], torch.tensor([1.0, 2.5]))
    torch.testing.assert_close(units["current_logp"], torch.tensor([-3.0, -7.0]))
    torch.testing.assert_close(units["reference_logp"], torch.tensor([-2.0, -6.0]))
    torch.testing.assert_close(units["signals"]["overlap"], torch.tensor([0.3, 0.7]))
    assert units["supervised_tokens"] == 4
    assert units["supervision_units"] == 2


def test_block_mean_advantage_uses_block_mean_signal_and_block_logprob_sum():
    raw_advantage = torch.tensor([0.5, 0.25, -0.5, 1.0, 99.0])
    lambdas = torch.tensor([1.0, 2.0, 3.0, 4.0, 99.0])
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 99.0])
    current_logp = torch.tensor([-1.0, -2.0, -3.0, -4.0, -99.0])
    signals = {"overlap": torch.tensor([0.2, 0.4, 0.6, 0.8, 99.0])}

    units = build_supervision_units(
        weights=weights,
        raw_advantage=raw_advantage,
        lambdas=lambdas,
        signals=signals,
        current_logp=current_logp,
        token_supervision_stride=1,
        token_supervision_offset=0,
        block_size=2,
        block_advantage_mode="mean",
    )

    torch.testing.assert_close(units["weights"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(units["raw_advantage"], torch.tensor([0.375, 0.25]))
    torch.testing.assert_close(units["advantage"], torch.tensor([0.5, 1.25]))
    torch.testing.assert_close(units["current_logp"], torch.tensor([-3.0, -7.0]))
    assert units["supervised_tokens"] == 4
    assert units["supervision_units"] == 2


def test_block_mixed_advantage_preserves_token_units_with_block_mean_context():
    raw_advantage = torch.tensor([0.5, 0.25, -0.5, 1.0, 99.0])
    lambdas = torch.ones_like(raw_advantage)
    weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 99.0])
    current_logp = torch.tensor([-1.0, -2.0, -3.0, -4.0, -99.0])
    signals = {"overlap": torch.tensor([0.2, 0.4, 0.6, 0.8, 99.0])}

    units = build_supervision_units(
        weights=weights,
        raw_advantage=raw_advantage,
        lambdas=lambdas,
        signals=signals,
        current_logp=current_logp,
        token_supervision_stride=1,
        token_supervision_offset=0,
        block_size=2,
        block_advantage_mode="mixed",
        block_mix_lambda=0.5,
    )

    torch.testing.assert_close(units["weights"], torch.tensor([1.0, 3.0, 2.0, 4.0]))
    torch.testing.assert_close(units["raw_advantage"], torch.tensor([0.4375, 0.3125, -0.125, 0.625]))
    torch.testing.assert_close(units["advantage"], torch.tensor([0.4375, 0.3125, -0.125, 0.625]))
    torch.testing.assert_close(units["current_logp"], torch.tensor([-1.0, -2.0, -3.0, -4.0]))
    torch.testing.assert_close(units["signals"]["overlap"], torch.tensor([0.2, 0.4, 0.6, 0.8]))
    assert units["supervised_tokens"] == 4
    assert units["supervision_units"] == 4

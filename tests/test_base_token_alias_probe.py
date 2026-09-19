import importlib.util
from pathlib import Path

import torch


def module():
    path = Path(__file__).parents[1] / 'scripts/inspect_qwen_base_token_aliases.py'
    assert path.exists(), 'Missing read-only weight/probability probe'
    spec = importlib.util.spec_from_file_location('alias_probe', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_alias_detection_has_explicit_tolerance():
    p = module()
    x = torch.tensor([[1., 2.], [1., 2.00005], [1., 2.001]])
    assert p.alias_indices(x, x[:1], 0.0001) == [0, 1]
    assert p.alias_indices(x, x[:1], 0.) == [0]


def test_probability_mass_is_not_number_of_aliases():
    p = module()
    logits = torch.tensor([[.8, .15, .05]]).log()
    stats = p.distribution_stats(logits, [1, 2])
    assert abs(stats['alias_raw_mass'] - .2) < 1e-6
    assert abs(stats['alias_nucleus_mass'] - .15 / .95) < 1e-6
    assert stats['nucleus_size'] == 2


def test_tail_probe_preserves_original_context_and_target():
    p = module()
    assert hasattr(p, 'tail_probe_input'), 'Missing independent tail verification'
    assert p.tail_probe_input({'prompt_token_ids': [1, 2], 'response_token_ids': [3, 4, 5]}) == ([1, 2, 3, 4], 5)

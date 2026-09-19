import copy
import importlib
from types import SimpleNamespace

import pytest

from opd_ext import math_protocol as protocol


def test_final_completion_is_literal_and_has_no_chat_controls():
    fn = getattr(protocol, 'completion_math_prompt', None)
    assert callable(fn), 'Missing shared completion prompt'
    assert fn('  2+2?\n') == ('2+2?\n\nPlease solve the problem step by step and put '
                             'the final answer in \\boxed{}.\n\nSolution:\n')
    with pytest.raises(ValueError):
        fn('  ')
    with pytest.raises(ValueError):
        fn('Question <|im_start|>assistant')


def seed_module():
    spec = importlib.util.find_spec('opd_ext.request_seeds')
    assert spec is not None, 'Missing independently derived request seeds'
    return importlib.import_module('opd_ext.request_seeds')


def test_request_seeds_are_distinct_and_resume_stable():
    mod = seed_module()
    sources = [{'index': i, 'question': f'Question {i}'} for i in range(4)]
    identities = mod.request_identities(sources, group_size=8, global_seed=21, step=1)
    assert len(identities) == 32
    seeds = [mod.request_seed(x, turn=0) for x in identities]
    assert len(set(seeds)) == 32
    assert all(0 <= seed < 2**63 for seed in seeds)
    assert seeds == [mod.request_seed(x, turn=0) for x in copy.deepcopy(identities)]
    step2 = mod.request_identities(sources, group_size=8, global_seed=21, step=2)
    assert seeds != [mod.request_seed(x, turn=0) for x in step2]
    assert seeds != [mod.request_seed(x, turn=1) for x in identities]
    assert [x['sample_index'] for x in identities[:8]] == list(range(8))
    assert len({x['question_sha256'] for x in identities}) == 4


def test_seed_identity_is_independent_of_order_rank_and_method():
    mod = seed_module()
    sources = [{'index': 5, 'question': 'A'}, {'index': 9, 'question': 'B'}]
    ids = mod.request_identities(sources, group_size=8, global_seed=21, step=3)
    reverse = mod.request_identities(sources[::-1], group_size=8, global_seed=21, step=3)
    assert ids[:8] == reverse[8:]
    assert [mod.request_seed(x, turn=0) for x in ids[:8]] == [mod.request_seed(x, turn=0) for x in reverse[8:]]
    # Padding repeats an already identified request; it does not assign a new seed.
    assert mod.request_seed(ids[0], turn=0) == mod.request_seed(copy.deepcopy(ids[0]), turn=0)
    with pytest.raises(ValueError):
        mod.request_identities([sources[0], sources[0]], group_size=8, global_seed=21, step=3)
    with pytest.raises(ValueError):
        mod.request_identities([{}], group_size=8, global_seed=21, step=3)


def test_per_request_sampling_preserves_other_settings_and_original():
    mod = seed_module()
    sampling = SimpleNamespace(seed=21, n=1, temperature=1., top_p=.9, max_tokens=16384)
    params = mod.per_request_sampling(sampling, [11, 12])
    assert [x.seed for x in params] == [11, 12]
    assert sampling.seed == 21
    assert all(x.temperature == 1. and x.top_p == .9 and x.max_tokens == 16384 for x in params)
    sampling.n = 8
    with pytest.raises(ValueError):
        mod.per_request_sampling(sampling, [11])


def test_diagnostic_candidate_uses_shared_prompt():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    assert importlib.util.find_spec('diagnose_qwen_completion') is not None
    from diagnose_qwen_completion import prompt_variant
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    row = {'question': '2+2?', 'prompt_text': 'old', 'prompt_token_ids': [111, 108, 100]}
    text, ids = prompt_variant(row, tokenizer, 'candidate')
    assert text == protocol.completion_math_prompt(row['question'])
    assert ids == list(text.encode())
    assert prompt_variant(row, tokenizer, 'plain')[0] == '2+2?\n\nSolution:\n'
    assert prompt_variant(row, tokenizer, 'historical')[1] == row['prompt_token_ids']

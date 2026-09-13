import gzip
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def module():
    return importlib.import_module('opd_ext.math_protocol')


def test_prompt_matches_historical_eval_and_is_idempotent():
    m = module()
    expected = 'Find x.\n\nPlease reason step by step, and put your final answer within \\boxed{}.'
    assert m.math_prompt(' Find x. ') == expected
    assert m.math_prompt(expected) == expected
    with pytest.raises(ValueError):
        m.math_prompt('')


def test_render_explicitly_disables_without_silent_fallback():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
            return messages[0]['content'] + '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    m = module()
    assert m.render_nonthinking(Tokenizer(), 'question').endswith(m.DISABLED_SUFFIX)
    class Unsupported:
        def apply_chat_template(self, *args, **kwargs):
            raise TypeError('unsupported')
    with pytest.raises(TypeError):
        m.render_nonthinking(Unsupported(), 'question')


def test_generation_metadata_preserves_engine_tokens_and_stop():
    m = module()
    params = SimpleNamespace(temperature=1., top_p=.9, top_k=-1, seed=21,
                             max_tokens=4, n=1, ignore_eos=False, stop_token_ids=[151643,151645])
    sample = SimpleNamespace(token_ids=[7,151645], finish_reason='stop', stop_reason=151645,
                             cumulative_logprob=-2.)
    result = m.generation_record([3,4], sample, params, 151643)
    assert result['response_token_ids'] == [7,151645]
    assert result['finish_reason'] == 'stop'
    assert result['sampling']['stop_token_ids'] == [151643,151645]
    assert m.length_mask(torch.tensor([[7,151645,151643,151643]]), [2]).tolist() == [[1,1,0,0]]
    with pytest.raises(ValueError):
        m.length_mask(torch.zeros((1,4)), [5])


def test_rollout_archive_is_lossless_and_never_overwrites(tmp_path):
    m = module()
    metadata = {'prompt_token_ids':[3,4], 'response_token_ids':[7,151645],
                'finish_reason':'stop','stop_reason':151645,'sampling':{'max_tokens':4}}
    records = np.empty(1,dtype=object)
    records[0] = metadata
    batch = SimpleNamespace(batch={'prompts':torch.tensor([[0,3,4]]),
                                  'responses':torch.tensor([[7,151645,151643,151643]]),
                                  'attention_mask':torch.tensor([[0,1,1,1,1,0,0]]),
                                  'rollout_log_probs':torch.tensor([[-1.,-2.,-1.,-1.]])},
                            non_tensor_batch={'generation_record':records, 'uid':np.array(['g']),
                                              'traj_uid':np.array(['t']),
                                              'source_extra_info':np.array([{'index':12,'question':'q'}],dtype=object)})
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: '<|im_end|>' if 151645 in ids else 'prompt')
    metrics = m.save_rollouts(batch, tokenizer, tmp_path, step=1, run_id='r', attempt_id='a', source_commit='sha')
    assert metrics['rollout_archive/count'] == 1
    p = tmp_path/'a/step_000001/raw.jsonl.gz'
    with gzip.open(p,'rt') as f:
        row=json.loads(f.readline())
    assert row['response_text'] == '<|im_end|>'
    assert row['response_token_ids'] == [7,151645]
    assert row['response_mask'] == [1,1]
    assert row['source_extra_info']['index'] == 12
    with pytest.raises(FileExistsError):
        m.save_rollouts(batch, tokenizer, tmp_path, step=1, run_id='r', attempt_id='a', source_commit='sha')


def test_generated_think_detection_excludes_prompt_control_block():
    m=module()
    assert m.generated_think_tags([151667,7,151668])
    assert not m.generated_think_tags([7,151645])

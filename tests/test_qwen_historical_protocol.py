import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opd_ext import math_protocol as m

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'qwen3_historical17_v1'


class Tokenizer:
    eos_token_id = 151645

    def get_vocab(self):
        return {'<|im_start|>': 151644, '<|im_end|>': 151645}

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n<|im_start|>assistant\n'

    def encode(self, text, **kwargs):
        assert kwargs == {'add_special_tokens': False}
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return str(ids)


def test_qwen_historical_prompt_is_exact_without_thinking_override():
    assert getattr(m, 'QWEN_HISTORICAL_PROTOCOL', None) == PROTOCOL
    tok = Tokenizer()
    question = '  Find {x}.\n'
    text = m.render_qwen_historical(tok, m.historical_math_prompt(question))
    assert tok.kwargs == {'tokenize': False, 'add_generation_prompt': True}
    evidence = m.validate_qwen_historical_prompt(tok, question, list(text.encode()))
    assert evidence['enable_thinking'] is None
    assert evidence['protocol'] == PROTOCOL
    assert evidence['prompt_token_ids'] == list(text.encode())
    assert evidence['stop_token_ids'] == []
    short = list(text.encode())[:16] + list(text.encode())[-16:]
    assert m.validate_qwen_historical_prompt(tok, question, short, max_prompt_length=32)['prompt_token_ids'] == short
    with pytest.raises(ValueError):
        m.validate_qwen_historical_prompt(tok, question, short)


def test_metadata_retention_does_not_replace_historical_eos_mask():
    response = torch.tensor([[7, 151643, 151643, 151643]])
    legacy = torch.ones_like(response)
    selected = m.select_response_mask(response, [2], legacy, preserve_legacy=True)
    assert selected is legacy
    assert m.select_response_mask(response, [2], legacy).tolist() == [[1, 1, 0, 0]]


def test_qwen_archive_separates_generated_tokens_from_full_training_mask(tmp_path):
    metadata = {'prompt_token_ids': [3, 4], 'response_token_ids': [7, 151643],
                'finish_reason': 'stop', 'stop_reason': 151643, 'sampling': {'max_tokens': 4}}
    records = np.empty(1, dtype=object)
    records[0] = metadata
    batch = SimpleNamespace(batch={
        'prompts': torch.tensor([[0, 3, 4]]), 'responses': torch.tensor([[7, 151643, 151643, 151643]]),
        'attention_mask': torch.tensor([[0, 1, 1, 1, 1, 1, 1]]),
        'rollout_log_probs': torch.tensor([[-1., -2., -1., -1.]])}, non_tensor_batch={
            'generation_record': records, 'uid': np.array(['g']), 'traj_uid': np.array(['t']),
            'source_extra_info': np.array([{'question': 'q', 'index': 1}], dtype=object)})
    m.save_rollouts(batch, Tokenizer(), tmp_path, step=1, run_id='r', attempt_id='formal',
                    source_commit='commit', protocol=PROTOCOL)
    with gzip.open(tmp_path / 'formal/step_000001/raw.jsonl.gz', 'rt') as stream:
        row = json.loads(stream.readline())
    assert row['enable_thinking'] is None
    assert row['response_token_ids'] == [7, 151643]
    assert row['response_length'] == 2
    assert row['training_response_mask'] == [1, 1, 1, 1]
    assert row['training_response_token_ids'] == [7, 151643, 151643, 151643]
    assert row['training_rollout_log_probs'] == [-1., -2., -1., -1.]
    assert row['mask_policy'] == 'historical_eos_mask'


def test_qwen_launcher_keeps_legacy_stop_and_mask_policy():
    script = (ROOT / 'scripts/run_revisiting_sampled_block_opd_math.sh').read_text()
    assert PROTOCOL in script
    assert '+actor_rollout_ref.rollout.preserve_legacy_response_mask=true' in script
    source = (ROOT / 'external/revisiting_opd/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py').read_text()
    assert 'select_response_mask' in source and 'preserve_legacy_response_mask' in source

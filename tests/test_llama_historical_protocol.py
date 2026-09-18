import ast
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opd_ext import math_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'external/revisiting_opd'
HISTORICAL = 'llama32_historical17_v1'
QUESTION = ' Find {x}. \n'
LEGACY_TEXT = (
    'Math problem: ' + QUESTION + '\n\n'
    'Please carefully reason through the math problem step by step and derive the correct answer. '
    'You must conduct reasoning inside <think> and </think> and give the final answer within \\boxed{}.\n'
)


class NativeTokenizer:
    vocab = {'<|begin_of_text|>': 128000, '<|end_of_text|>': 128001,
             '<|start_header_id|>': 128006, '<|end_header_id|>': 128007,
             '<|eom_id|>': 128008, '<|eot_id|>': 128009}
    pad_token_id = 128001
    eos_token_id = 128009

    def get_vocab(self):
        return self.vocab

    def apply_chat_template(self, messages, **kwargs):
        self.last_kwargs = kwargs
        self.last_messages = messages
        return ('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'
                'Today Date: ' + kwargs.get('date_string', 'UNFIXED') + '<|eot_id|>'
                '<|start_header_id|>user<|end_header_id|>\n\n' + messages[0]['content']
                + '<|eot_id|>' + protocol.LLAMA_SUFFIX)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        parts = re.split('(' + '|'.join(map(re.escape, self.vocab)) + ')', text)
        return [token for part in parts for token in
                ([self.vocab[part]] if part in self.vocab else [1000 + ord(c) for c in part])]

    def decode(self, ids, **kwargs):
        assert kwargs == {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False}
        inverse = {value: key for key, value in self.vocab.items()}
        return ''.join(inverse.get(i, chr(i - 1000)) for i in ids)


class QwenTokenizer(NativeTokenizer):
    vocab = {'<|im_start|>': 151644, '<|im_end|>': 151645, '<|endoftext|>': 151643,
             '<think>': 151667, '</think>': 151668}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n' + protocol.DISABLED_SUFFIX


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method(path, name, namespace):
    tree = ast.parse(path.read_text())
    fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


def test_legacy_instruction_is_exact_and_preserves_question_whitespace():
    assert getattr(protocol, 'LLAMA_HISTORICAL_PROTOCOL', None) == HISTORICAL
    assert protocol.historical_math_prompt(QUESTION) == LEGACY_TEXT
    namespace = {}
    exec((EXTERNAL / 'agent_system/environments/prompts/math.py').read_text(), namespace)
    assert protocol.historical_math_prompt(QUESTION) == namespace['MATH_TEMPLATE'].format(task_description=QUESTION)


def test_historical_render_omits_thinking_kwarg_and_uses_fixed_native_template():
    tokenizer = NativeTokenizer()
    rendered = protocol.render_historical_training(tokenizer, LEGACY_TEXT)
    assert tokenizer.last_kwargs == {'tokenize': False, 'add_generation_prompt': True, 'date_string': '18 Sep 2026'}
    assert tokenizer.last_messages == [{'role': 'user', 'content': LEGACY_TEXT}]
    assert rendered.startswith('<|begin_of_text|>') and rendered.endswith(protocol.LLAMA_SUFFIX)
    assert '<|im_start|>' not in rendered
    assert protocol.valid_control_prefix(rendered, tokenizer, HISTORICAL)
    assert not protocol.valid_control_prefix(rendered + protocol.DISABLED_SUFFIX, tokenizer, HISTORICAL)
    assert protocol.tokenizer_protocol(tokenizer) == protocol.LLAMA_PROTOCOL
    with pytest.raises(ValueError, match='Llama'):
        protocol.render_historical_training(SimpleNamespace(), LEGACY_TEXT)


def test_actual_historical_prompt_gate_checks_native_ids_and_eval_difference():
    tokenizer = NativeTokenizer()
    text = protocol.render_historical_training(tokenizer, LEGACY_TEXT)
    ids = tokenizer.encode(text, add_special_tokens=False)
    evidence = protocol.validate_historical_training_prompt(tokenizer, QUESTION, ids)
    assert evidence['protocol'] == HISTORICAL and evidence['enable_thinking'] is None
    assert evidence['prompt_token_ids'] == ids and evidence['prompt_text'] == text
    assert evidence['stop_token_ids'] == [128001, 128008, 128009]
    assert ids[0] == 128000 and ids.count(128000) == 1 and 128009 in ids
    evaluation = protocol.render_nonthinking(tokenizer, protocol.math_prompt(QUESTION))
    assert tokenizer.last_kwargs['enable_thinking'] is False
    assert '<think>' not in evaluation and evaluation != text
    for bad_ids in ([128000] + ids, ids[:-1], tokenizer.encode(evaluation, add_special_tokens=False)):
        with pytest.raises(ValueError, match='prompt|BOS'):
            protocol.validate_historical_training_prompt(tokenizer, QUESTION, bad_ids)
    truncated = ids[:128] + ids[-128:]
    assert protocol.validate_historical_training_prompt(
        tokenizer, QUESTION, truncated, max_prompt_length=256, truncation='middle')['prompt_token_ids'] == truncated
    with pytest.raises(ValueError):
        protocol.validate_historical_training_prompt(tokenizer, QUESTION, truncated, max_prompt_length=256)


def make_batch(tokenizer, text, response_text, count=1):
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    params = SimpleNamespace(temperature=1., top_p=.9, top_k=-1, seed=21, max_tokens=len(response_ids)+2,
                             n=1, ignore_eos=False, stop_token_ids=[128001, 128008, 128009])
    sample = SimpleNamespace(token_ids=response_ids, finish_reason='stop', stop_reason=128009, cumulative_logprob=-2.)
    records = np.empty(count, dtype=object)
    records[:] = [protocol.generation_record(prompt_ids, sample, params, tokenizer.eos_token_id) for _ in range(count)]
    batch = SimpleNamespace(
        batch={'prompts': torch.tensor([[128001] + prompt_ids] * count),
               'responses': torch.tensor([response_ids + [128001, 128001]] * count),
               'attention_mask': torch.tensor([[0] + [1]*len(prompt_ids) + [1]*len(response_ids) + [0, 0]] * count),
               'rollout_log_probs': torch.full((count, len(response_ids)+2), -1.)},
        non_tensor_batch={'generation_record': records, 'uid': np.array([f'g{i//8}' for i in range(count)]),
                          'traj_uid': np.array([f't{i}' for i in range(count)]),
                          'source_extra_info': np.array([{'index': 12, 'question': QUESTION}] * count, dtype=object)})
    return batch


@pytest.mark.parametrize('tag', ['', '<think>reason</think>', '</think>'])
@pytest.mark.parametrize('selected', [HISTORICAL, protocol.LLAMA_PROTOCOL, None])
def test_archives_are_lossless_and_truthfully_label_thinking(tmp_path, tag, selected):
    tokenizer = NativeTokenizer()
    content = LEGACY_TEXT if selected == HISTORICAL else protocol.math_prompt(QUESTION)
    text = tokenizer.apply_chat_template([{'content': content}], date_string=protocol.LLAMA_DATE)
    batch = make_batch(tokenizer, text, tag + 'answer<|eot_id|>')
    kwargs = {} if selected is None else {'protocol': selected}
    metrics = protocol.save_rollouts(batch, tokenizer, tmp_path, step=1, run_id='r', attempt_id='a', source_commit='sha', **kwargs)
    path = tmp_path / 'a/step_000001/raw.jsonl.gz'
    with gzip.open(path, 'rt') as handle:
        row = json.loads(handle.readline())
    assert row['protocol'] == (selected or protocol.LLAMA_PROTOCOL)
    assert row['enable_thinking'] is (None if selected == HISTORICAL else False)
    assert row['generated_think_tags'] is bool(tag)
    assert metrics['rollout_archive/generated_think_tag_rate'] == int(bool(tag))
    for key, value in batch.non_tensor_batch['generation_record'][0].items():
        assert row[key] == value
    assert row['prompt_text'] == text
    assert row['response_text'] == tag + 'answer<|eot_id|>'
    assert row['response_mask'] == [1] * row['response_length'] and row['padding_length'] == 2
    assert row['source_extra_info'] == {'index': 12, 'question': QUESTION}
    assert row['rollout_log_probs'] == [-1.] * row['response_length']
    assert path.with_suffix('.gz.sha256').read_text().split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        protocol.save_rollouts(batch, tokenizer, tmp_path, step=1, run_id='r', attempt_id='a', source_commit='sha', **kwargs)


def test_llama_tag_detection_does_not_use_qwen_numeric_ids():
    tokenizer = NativeTokenizer()
    assert protocol.generated_think_tags([151667])
    assert not protocol.generated_think_tags([151667], tokenizer=tokenizer, text='ordinary text')
    assert protocol.generated_think_tags([3, 4], tokenizer=tokenizer, text='<think>reason')
    assert protocol.generated_think_tags(tokenizer.encode('</think>', add_special_tokens=False), tokenizer=tokenizer)


@pytest.mark.parametrize('stop', protocol.LLAMA_STOP_IDS)
def test_native_stop_tokens_remain_in_actual_length_masks(stop):
    assert protocol.length_mask(torch.tensor([[7, stop, 128001, 128001]]), [2]).tolist() == [[1, 1, 0, 0]]


@pytest.mark.parametrize('fault', ['family', 'mask', 'ids'])
def test_archive_rejects_protocol_or_training_tensor_mismatch(tmp_path, fault):
    tokenizer = NativeTokenizer()
    text = tokenizer.apply_chat_template([{'content': LEGACY_TEXT}], date_string=protocol.LLAMA_DATE)
    batch = make_batch(tokenizer, text, 'answer<|eot_id|>')
    if fault == 'mask':
        batch.batch['attention_mask'][0, -3] = 0
    elif fault == 'ids':
        batch.batch['responses'][0, 0] += 1
    with pytest.raises(ValueError):
        protocol.save_rollouts(batch, tokenizer, tmp_path, step=1, run_id='r', attempt_id='a', source_commit='sha',
                               protocol=protocol.PROTOCOL if fault == 'family' else HISTORICAL)


@pytest.mark.parametrize('selected', [HISTORICAL, protocol.LLAMA_PROTOCOL, protocol.PROTOCOL, 'legacy'])
def test_launcher_protocol_arguments_without_launching_jobs(selected):
    source = (ROOT / 'scripts/run_revisiting_sampled_block_opd_math.sh').read_text()
    block = source[source.index('protocol_args=()'):source.index('\ncase "${VARIANT}"')]
    env = dict(os.environ, OPD_PROMPT_PROTOCOL=selected, LOSSLESS_ROLLOUT_DIR='/unused/rollouts',
               ROLLOUT_ATTEMPT_ID='test', SOURCE_COMMIT='sha')
    result = subprocess.run(['bash', '-c', 'set -eu\n' + block + '\nprintf "%s\\n" "${protocol_args[@]}"'],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    if selected == 'legacy':
        assert not any(args)
        return
    assert f'+data.opd_prompt_protocol={selected}' in args
    assert '+actor_rollout_ref.rollout.retain_generation_metadata=true' in args
    stop_ids = '[151643,151645]' if selected == protocol.PROTOCOL else '[128001,128008,128009]'
    assert f'+actor_rollout_ref.rollout.stop_token_ids={stop_ids}' in args
    assert ('+data.apply_chat_template_kwargs.enable_thinking=false' in args) is (selected != HISTORICAL)


@pytest.mark.parametrize('selected', [HISTORICAL, protocol.LLAMA_PROTOCOL])
def test_actual_environment_and_collector_path(selected):
    tokenizer = NativeTokenizer()
    class Config(dict):
        __getattr__ = dict.__getitem__
    config = SimpleNamespace(data=Config(opd_prompt_protocol=selected, max_prompt_length=2048, truncation='middle',
                                        return_raw_chat=True, apply_chat_template_kwargs={} if selected == HISTORICAL else {'enable_thinking': False}))
    tree = ast.parse((EXTERNAL / 'agent_system/environments/env_manager.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MathEnvironmentManager')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'build_text_obs')
    namespace = {'List': list}
    exec((EXTERNAL / 'agent_system/environments/prompts/math.py').read_text(), namespace)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<math environment>', 'exec'), namespace)
    manager = SimpleNamespace(config=config, tasks=[QUESTION])
    observations = namespace['build_text_obs'](manager, [QUESTION])
    expected = LEGACY_TEXT if selected == HISTORICAL else protocol.math_prompt(QUESTION)
    assert observations == [expected]
    def tokenize(prompt, tokenizer, **kwargs):
        ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)])
        return ids, torch.ones_like(ids)
    preprocess = method(EXTERNAL / 'agent_system/multi_turn_rollout/rollout_loop.py', 'preprocess_single_sample',
                        {'np': np, 'torch': torch, 'verl_F': SimpleNamespace(tokenize_and_postprocess_data=tokenize),
                         'compute_position_id_with_mask': lambda mask: mask.cumsum(-1)-1})
    collector = SimpleNamespace(config=config, tokenizer=tokenizer, processor=None)
    batch = SimpleNamespace(non_tensor_batch={'raw_prompt': [[{'role': 'user', 'content': QUESTION}]],
                                             'data_source': ['math']}, batch={'input_ids': torch.zeros((1, 1))})
    processed = preprocess(collector, 0, batch, {'text': observations})
    assert processed['raw_prompt_ids'] == processed['input_ids'].tolist()
    assert tokenizer.last_kwargs.get('date_string') == '18 Sep 2026'
    assert ('enable_thinking' not in tokenizer.last_kwargs) if selected == HISTORICAL else tokenizer.last_kwargs['enable_thinking'] is False
    config.data['apply_chat_template_kwargs'] = {'enable_thinking': False} if selected == HISTORICAL else {}
    with pytest.raises(ValueError, match='thinking'):
        preprocess(collector, 0, batch, {'text': observations})


@pytest.mark.parametrize('selected', [HISTORICAL, protocol.LLAMA_PROTOCOL])
def test_shared_audit_uses_selected_training_protocol(tmp_path, monkeypatch, selected):
    tokenizer = NativeTokenizer()
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    module = load_script('run_qwen06_nonthinking')
    text = (tokenizer.apply_chat_template([{'content': LEGACY_TEXT}], date_string=protocol.LLAMA_DATE)
            if selected == HISTORICAL else protocol.render_nonthinking(tokenizer, protocol.math_prompt(QUESTION)))
    batch = make_batch(tokenizer, text, '<think>reason</think>answer<|eot_id|>', count=32)
    protocol.save_rollouts(batch, tokenizer, tmp_path / 'rollouts', step=1, run_id='r', attempt_id='a', source_commit='sha', protocol=selected)
    evidence = module.audit_rollouts(tmp_path, [1], student='unused', protocol=selected, stop_ids=protocol.LLAMA_STOP_IDS)
    assert evidence[0]['n'] == 32 and evidence[0]['generated_think_tags'] == 32
    if selected == HISTORICAL:
        assert module.audit_rollouts(tmp_path, [1], student='unused', protocol=selected) == evidence
    path = tmp_path / 'rollouts/a/step_000001/raw.jsonl.gz'
    with gzip.open(path, 'rt') as handle:
        rows = [json.loads(line) for line in handle]
    for fault in ('thinking', 'protocol', 'stops'):
        modified = json.loads(json.dumps(rows))
        if fault == 'thinking':
            modified[0]['enable_thinking'] = False if selected == HISTORICAL else None
        elif fault == 'protocol':
            modified[0]['protocol'] = protocol.LLAMA_PROTOCOL if selected == HISTORICAL else HISTORICAL
        else:
            modified[0]['sampling']['stop_token_ids'] = [151643, 151645]
        with gzip.open(path, 'wt') as handle:
            handle.write(''.join(json.dumps(row) + '\n' for row in modified))
        path.with_suffix('.gz.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
        with pytest.raises(ValueError):
            module.audit_rollouts(tmp_path, [1], student='unused', protocol=selected, stop_ids=protocol.LLAMA_STOP_IDS)


@pytest.mark.parametrize('selected', [HISTORICAL, protocol.LLAMA_PROTOCOL])
def test_gpu_gate_prompt_contract_and_tag_policy_without_gpu(selected):
    gate = load_script('verify_nonthinking_gpu')
    tokenizer = NativeTokenizer()
    observation = LEGACY_TEXT if selected == HISTORICAL else protocol.math_prompt(QUESTION)
    text = tokenizer.apply_chat_template([{'content': observation}], date_string=protocol.LLAMA_DATE)
    ids = tokenizer.encode(text, add_special_tokens=False)
    processed = {'raw_prompt_ids': ids, 'input_ids': torch.tensor([128001] + ids),
                 'attention_mask': torch.tensor([0] + [1] * len(ids))}
    evaluation = protocol.render_nonthinking(tokenizer, protocol.math_prompt(QUESTION))
    evidence = gate.validate_prompt_contract(tokenizer, QUESTION, observation, processed, selected, evaluation)
    assert evidence['prompt_token_ids'] == ids and evidence['prompt_text'] == text
    assert evidence['train_eval_prompt_ids_identical'] is (selected != HISTORICAL)
    assert evidence['eval_prompt_text'] == evaluation
    processed['raw_prompt_ids'] = [128000] + ids
    with pytest.raises((ValueError, AssertionError)):
        gate.validate_prompt_contract(tokenizer, QUESTION, observation, processed, selected, evaluation)
    rows = [{'generated_think_tags': True, 'finish_reason': 'stop', 'response_token_ids': [7, 128009]}] * 16
    summary = gate.generation_summary(rows, selected)
    assert summary['passed'] is (selected == HISTORICAL)
    assert summary['generated_think_tag_count'] == 16
    assert summary['stop_token_ids'] == [128001, 128008, 128009]
    assert not gate.generation_summary(rows[:15], selected)['passed']
    for row in rows:
        row['generated_think_tags'] = False
    assert gate.generation_summary(rows, selected)['passed']


def test_qwen_gpu_gate_keeps_disabled_prefix_and_zero_tag_requirement():
    gate = load_script('verify_nonthinking_gpu')
    tokenizer = QwenTokenizer()
    observation = protocol.math_prompt(QUESTION)
    evaluation = protocol.render_nonthinking(tokenizer, observation)
    ids = tokenizer.encode(evaluation, add_special_tokens=False)
    processed = {'raw_prompt_ids': ids, 'input_ids': torch.tensor(ids), 'attention_mask': torch.ones(len(ids))}
    evidence = gate.validate_prompt_contract(tokenizer, QUESTION, observation, processed, protocol.PROTOCOL, evaluation)
    assert evidence['prompt_token_ids'] == ids and evidence['train_eval_prompt_ids_identical'] is True
    assert evidence['enable_thinking'] is False and evidence['eval_enable_thinking'] is False
    rows = [{'generated_think_tags': False, 'finish_reason': 'length', 'response_token_ids': [7]} for _ in range(16)]
    assert gate.generation_summary(rows, protocol.PROTOCOL)['passed']
    rows[0]['generated_think_tags'] = True
    assert not gate.generation_summary(rows, protocol.PROTOCOL)['passed']


@pytest.mark.parametrize('selected', [HISTORICAL, protocol.LLAMA_PROTOCOL, protocol.PROTOCOL])
def test_gpu_gate_sampling_parameters_remain_unchanged(selected):
    tree = ast.parse((ROOT / 'scripts/verify_nonthinking_gpu.py').read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == 'SamplingParams')
    namespace = {'args': SimpleNamespace(protocol=selected), 'SamplingParams': lambda **kwargs: kwargs,
                 'LLAMA_PROTOCOL': protocol.LLAMA_PROTOCOL, 'LLAMA_HISTORICAL_PROTOCOL': HISTORICAL,
                 'LLAMA_STOP_IDS': protocol.LLAMA_STOP_IDS, 'STOP_TOKEN_IDS': protocol.STOP_TOKEN_IDS}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<sampling parameters>', 'exec'), namespace)
    assert namespace['params'] == {'n': 1, 'temperature': 1., 'top_p': .9, 'top_k': -1, 'seed': 21,
                                   'max_tokens': 16384, 'ignore_eos': False, 'detokenize': False, 'logprobs': 0,
                                   'stop_token_ids': protocol.STOP_TOKEN_IDS if selected == protocol.PROTOCOL else protocol.LLAMA_STOP_IDS}


def test_external_trainer_forwards_selected_protocol_without_touching_generation_masking():
    tree = ast.parse((EXTERNAL / 'verl/trainer/ppo/ray_trainer_multitask.py').read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == 'save_rollouts')
    forwarded = next(k.value for k in call.keywords if k.arg == 'protocol')
    for selected in (HISTORICAL, protocol.LLAMA_PROTOCOL, protocol.PROTOCOL, None):
        config = SimpleNamespace(data={'opd_prompt_protocol': selected})
        assert eval(compile(ast.Expression(forwarded), '<archive hook>', 'eval'), {'self': SimpleNamespace(config=config)}) == selected
    launcher = (ROOT / 'scripts/run_revisiting_sampled_block_opd_math.sh').read_text()
    assert '+actor_rollout_ref.actor.opd_mask_special_tokens=False' in launcher

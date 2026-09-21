import ast
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opd_ext import math_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'external/revisiting_opd'
INSTRUCT = 'qwen3_native_chat_no_thinking_boxed_v1'
INSTRUCTION = '\n\nPlease solve the problem step by step and put the final answer in \\boxed{}.'
SUFFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'


class Tokenizer:
    eos_token_id = 151645
    bos_token_id = None
    pad_token_id = 151643
    name_or_path = 'fake-native-qwen'

    def encode(self, text, **kwargs):
        if text in ('<|im_end|>', '<|endoftext|>'):
            return [151645 if text == '<|im_end|>' else 151643]
        return list(text.encode())

    def decode(self, ids, **kwargs):
        special = {151645: b'<|im_end|>', 151643: b'<|endoftext|>'}
        return b''.join(bytes([i]) if i < 256 else special[i] for i in ids).decode()

    def get_vocab(self):
        return {**{str(i): i for i in range(256)}, '<|im_end|>': 151645, '<|endoftext|>': 151643}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt is True and enable_thinking is False
        text = '<|im_start|>user\n' + messages[0]['content'] + '<|im_end|>\n' + SUFFIX
        return list(text.encode()) if tokenize else text


def qualification():
    spec = importlib.util.spec_from_file_location('qualification_reference', ROOT / 'scripts/qualify_qwen17_pairs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    assert getattr(protocol, 'QWEN_INSTRUCT_PROTOCOL', None) == INSTRUCT
    names = ('qwen_instruct_math_prompt', 'qwen_instruct_render', 'qwen_instruct_input_ids')
    assert all(callable(getattr(protocol, name, None)) for name in names)
    assert protocol.qwen_instruct_user_content is protocol.qwen_instruct_math_prompt
    return tuple(getattr(protocol, name) for name in names)


@pytest.mark.parametrize('question', ['  2+2?\n', 'Find x.\nSecond line.', 'Already' + INSTRUCTION, ''])
def test_exact_qualification_prompt_and_native_ids(question):
    content, render, input_ids = helpers()
    tokenizer = Tokenizer()
    reference = qualification().prompt_input(tokenizer, question, 'instruct')
    assert content(question) == question.strip() + INSTRUCTION
    assert reference['messages'] == [{'role': 'user', 'content': content(question)}]
    assert render(tokenizer, question) == reference['completion_prompt_text']
    assert input_ids(tokenizer, question) == reference['base_prompt_token_ids']
    assert render(tokenizer, question).endswith(SUFFIX)


def local_tokenizer():
    path = Path(os.environ.get('QWEN_INSTRUCT_TOKENIZER', str(
        Path.home() / '.cache/modelscope/hub/models/Qwen/Qwen3-1___7B')))
    if not (path / 'tokenizer_config.json').is_file():
        pytest.skip('No local Qwen Instruct tokenizer; set QWEN_INSTRUCT_TOKENIZER')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def test_real_cached_tokenizer_matches_qualification():
    _, render, input_ids = helpers()
    tokenizer = local_tokenizer()
    assert tokenizer.eos_token_id == 151645
    assert protocol.evaluation_stop_ids(tokenizer) == [151645, 151643]
    for question in ('  Compute $2^3$.\n', 'Find x.\nSecond line.', 'Already' + INSTRUCTION):
        reference = qualification().prompt_input(tokenizer, question, 'instruct')
        assert render(tokenizer, question) == reference['completion_prompt_text']
        assert input_ids(tokenizer, question) == reference['base_prompt_token_ids']


@pytest.mark.parametrize('length', [2047, 2048, 2049])
def test_prompt_budget_boundary(length):
    _, render, input_ids = helpers()
    tokenizer = Tokenizer()
    question = 'x' * (length - len(render(tokenizer, '')))
    full = list(render(tokenizer, question).encode())
    assert len(full) == length
    actual = input_ids(tokenizer, question)
    if length <= 2048:
        assert actual == full == qualification().prompt_input(tokenizer, question, 'instruct')['base_prompt_token_ids']
    else:
        assert actual == full[:1024] + full[-1024:]


def test_long_input_middle_truncation_only_and_validation():
    _, render, input_ids = helpers()
    tokenizer = Tokenizer()
    question = 'long question ' * 400
    full = tokenizer.apply_chat_template([{'role': 'user', 'content': question.strip() + INSTRUCTION}],
                                        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert input_ids(tokenizer, question) == full[:1024] + full[-1024:]
    assert input_ids(tokenizer, question, max_prompt_length=101) == full[:50] + full[-51:]
    assert tokenizer.decode(input_ids(tokenizer, question)).endswith(SUFFIX)
    with pytest.raises(ValueError):
        qualification().prompt_input(tokenizer, question, 'instruct')
    for maximum in (0, 1, 2049):
        with pytest.raises(ValueError):
            input_ids(tokenizer, question, max_prompt_length=maximum)
    tokenizer.eos_token_id = 151643
    with pytest.raises(ValueError, match='EOS'):
        render(tokenizer, 'q')
    tokenizer.eos_token_id = 151645
    tokenizer.apply_chat_template = lambda *a, **kw: 'missing empty closed think'
    with pytest.raises(ValueError):
        render(tokenizer, 'q')


def test_native_ids_are_not_reencoded_and_mismatch_is_rejected():
    _, _, input_ids = helpers()
    tokenizer = Tokenizer()
    tokenizer.encode = lambda *a, **kw: pytest.fail('Must use native tokenize=True IDs')
    assert input_ids(tokenizer, 'q') == qualification().prompt_input(tokenizer, 'q', 'instruct')['base_prompt_token_ids']
    tokenizer.decode = lambda *a, **kw: 'incorrect decoded text'
    with pytest.raises(ValueError):
        input_ids(tokenizer, 'q')


def source_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = ast.walk(tree)
    if class_name:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    fn = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


class Config(dict):
    __getattr__ = dict.__getitem__


@pytest.mark.parametrize('real', [False, True], ids=['fake-tokenizer', 'real-tokenizer'])
@pytest.mark.parametrize('question', ['  2+2?\n', 'long ' * 2400], ids=['short', 'long'])
def test_actual_environment_and_collector_use_canonical_native_ids(question, real):
    helpers()
    config = SimpleNamespace(data=Config(opd_prompt_protocol=INSTRUCT, max_prompt_length=2048,
        truncation='middle', return_raw_chat=True, apply_chat_template_kwargs={'enable_thinking': False}))
    build = source_function(VENDOR / 'agent_system/environments/env_manager.py', 'build_text_obs', {},
                            class_name='MathEnvironmentManager')
    observation = build(SimpleNamespace(config=config, tasks=[question]), [question])
    assert observation == [question]
    # Execute vendor padding, but forbid its string-tokenization path for this protocol.
    ns = {'torch': torch, 'F': torch.nn.functional}
    functional = VENDOR / 'verl/utils/torch_functional.py'
    source_function(functional, 'pad_sequence_to_length', ns)
    postprocess = source_function(functional, 'postprocess_data', ns)
    preprocess = source_function(VENDOR / 'agent_system/multi_turn_rollout/rollout_loop.py',
        'preprocess_single_sample', {'torch': torch, 'np': np,
            'verl_F': SimpleNamespace(postprocess_data=postprocess,
                tokenize_and_postprocess_data=lambda **kw: pytest.fail('Collector reencoded native IDs')),
            'compute_position_id_with_mask': lambda mask: (mask.cumsum(-1) - 1).clamp(min=0)})
    tokenizer = local_tokenizer() if real else Tokenizer()
    collector = SimpleNamespace(config=config, tokenizer=tokenizer, processor=None)
    batch = SimpleNamespace(batch={'input_ids': [0]}, non_tensor_batch={
        'raw_prompt': [[{'role': 'user', 'content': 'old'}]], 'data_source': ['math']})
    actual = preprocess(collector, 0, batch, {'text': observation})
    expected = protocol.qwen_instruct_input_ids(tokenizer, question)
    if question.strip() == '2+2?':
        assert expected == qualification().prompt_input(tokenizer, question, 'instruct')['base_prompt_token_ids']
    assert actual['raw_prompt_ids'] == expected
    assert actual['input_ids'][actual['attention_mask'].bool()].tolist() == expected
    assert actual['input_ids'].shape == (2048,)
    assert actual['raw_prompt'] == [{'role': 'user', 'content': question.strip() + INSTRUCTION}]
    config.data['apply_chat_template_kwargs'] = {}
    with pytest.raises(ValueError):
        preprocess(collector, 0, batch, {'text': observation})


@pytest.mark.parametrize('terminal,finish,stop', [(151645, 'stop', None), (151643, 'stop', 151643), (9, 'length', None)])
def test_archive_native_mapping_eos_stops_length_mask_and_seed(tmp_path, terminal, finish, stop):
    helpers()
    from opd_ext.request_seeds import request_identities, request_seed
    identity = request_identities([{'index': 4, 'question': 'q'}], group_size=1, global_seed=21, step=2)[0]
    seed = request_seed(identity, turn=0)
    params = SimpleNamespace(temperature=1., top_p=.9, top_k=-1, seed=seed,
                            max_tokens=2 if finish == 'length' else 4, n=1,
                            ignore_eos=False, stop_token_ids=[151645, 151643])
    sample = SimpleNamespace(token_ids=[7, terminal], finish_reason=finish, stop_reason=stop, cumulative_logprob=-2.)
    record = protocol.generation_record([3, 4], sample, params, 151645)
    record.update(request_identity=identity, request_turn=0)
    response = torch.tensor([[7, terminal, 151643, 151643]])
    mask = protocol.select_response_mask(response, [2], torch.ones_like(response))
    assert mask.tolist() == [[1, 1, 0, 0]]
    batch = SimpleNamespace(batch={'prompts': torch.tensor([[0, 3, 4]]), 'responses': response,
        'attention_mask': torch.cat([torch.tensor([[0, 1, 1]]), mask], dim=1),
        'rollout_log_probs': torch.tensor([[-1., -2., 0., 0.]])}, non_tensor_batch={
            'generation_record': [record], 'uid': ['u'], 'traj_uid': ['t'],
            'source_extra_info': [{'index': 4, 'question': 'q'}]})
    kwargs = dict(step=2, run_id='r', attempt_id='a', source_commit='sha', protocol=INSTRUCT)
    protocol.save_rollouts(batch, Tokenizer(), tmp_path, **kwargs)
    with gzip.open(tmp_path / 'a/step_000002/raw.jsonl.gz', 'rt') as stream:
        saved = json.loads(stream.readline())
    assert saved['protocol'] == INSTRUCT and saved['enable_thinking'] is False
    assert saved['eos_token_id'] == 151645
    assert saved['sampling']['stop_token_ids'] == [151645, 151643]
    assert saved['response_mask'] == [1, 1] and saved['padding_length'] == 2
    assert saved['request_identity'] == identity and saved['sampling']['seed'] == seed
    assert saved['finish_reason'] == finish and saved['stop_reason'] == stop
    assert 'training_response_mask' not in saved
    batch.batch['attention_mask'][0, -1] = 1
    with pytest.raises(ValueError, match='mask'):
        protocol.save_rollouts(batch, Tokenizer(), tmp_path / 'bad', **kwargs)


@pytest.mark.parametrize('real', [False, True], ids=['fake-tokenizer', 'real-tokenizer'])
@pytest.mark.parametrize('archive', [False, True])
def test_eval_worker_dispatch_retains_native_ids_and_finish(monkeypatch, tmp_path, archive, real):
    from test_eval_rollout_archive import load_evaluator, engine, worker_args
    tokenizer = local_tokenizer() if real else Tokenizer()
    module = load_evaluator(monkeypatch)
    _, calls = engine(module)
    module.LLM.get_tokenizer = lambda self: tokenizer
    rows = module.worker_generate(worker_args(), prompt_protocol=INSTRUCT,
                                  rollout_archive_dir=tmp_path if archive else None)
    for row in rows:
        expected = qualification().prompt_input(tokenizer, row['problem'], 'instruct')
        assert row['prompt_token_ids'] == expected['base_prompt_token_ids']
        assert row['response_token_ids'] == [7, 8, 151645]
        assert row['finish_reason'] == 'stop' and row['stop_reason'] == 151645
        assert row['prompt_protocol'] == INSTRUCT
        if archive:
            assert row['rendered_prompt'] == expected['completion_prompt_text']
            assert row['sampling']['stop_token_ids'] == [151645, 151643]
            assert row['eos_token_id'] == 151645
    assert [p['seed'] for p in calls['sampling']] == [21, 25]
    assert all(p['stop_token_ids'] == [151645, 151643] for p in calls['sampling'])
    args = list(worker_args())
    args[8] = True
    with pytest.raises(ValueError):
        module.worker_generate(tuple(args), prompt_protocol=INSTRUCT)


def test_eval_cli_accepts_and_forwards_protocol(monkeypatch, tmp_path):
    from test_eval_rollout_archive import load_evaluator
    module = load_evaluator(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['eval', '--model-path', 'model', '--eval-jsonl-dir', str(tmp_path),
        '--output-dir', str(tmp_path / 'output'), '--tasks', 'math500', '--prompt-protocol', INSTRUCT])
    monkeypatch.setattr(module, 'load_jsonl', lambda path: [])
    (tmp_path / 'math500.jsonl').touch()
    submitted = []
    class Executor:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, fn, item, **kwargs):
            submitted.append(kwargs)
            future = module.concurrent.futures.Future()
            future.set_result([])
            return future
    monkeypatch.setattr(module.concurrent.futures, 'ProcessPoolExecutor', Executor)
    monkeypatch.setattr(module, 'grade_outputs', lambda *a: {})
    module.main()
    assert submitted and all(kw['prompt_protocol'] == INSTRUCT for kw in submitted)
    assert json.loads((tmp_path / 'output/eval_config.json').read_text())['prompt_protocol'] == INSTRUCT


def shell_protocol_args(selected, seed='sha256_step_question_sample_v1'):
    source = (ROOT / 'scripts/run_revisiting_sampled_block_opd_math.sh').read_text()
    block = source[source.index('protocol_args=()'):source.index('case "${VARIANT}" in')]
    return subprocess.run(['bash', '-c', 'set -eu\n' + block + '\nprintf "%s\\n" "${protocol_args[@]}"'],
        env={**os.environ, 'OPD_PROMPT_PROTOCOL': selected, 'OPD_REQUEST_SEED_RULE': seed,
             'LOSSLESS_ROLLOUT_DIR': '/raw', 'ROLLOUT_ATTEMPT_ID': 'a', 'SOURCE_COMMIT': 'sha'},
        text=True, capture_output=True)


def test_training_shell_accepts_instruct_stops_masks_and_existing_seed_rule():
    result = shell_protocol_args(INSTRUCT)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    for expected in (f'+data.opd_prompt_protocol={INSTRUCT}', '+data.apply_chat_template_kwargs.enable_thinking=false',
        '+actor_rollout_ref.rollout.stop_token_ids=[151645,151643]',
        '+actor_rollout_ref.rollout.retain_generation_metadata=true',
        f'+actor_rollout_ref.rollout.opd_prompt_protocol={INSTRUCT}',
        '+actor_rollout_ref.rollout.preserve_legacy_response_mask=false',
        '+actor_rollout_ref.rollout.request_seed_rule=sha256_step_question_sample_v1',
        '+trainer.lossless_rollout_dir=/raw'):
        assert expected in args
    assert shell_protocol_args(INSTRUCT, 'unknown').returncode != 0
    worker = (VENDOR / 'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py').read_text()
    assert 'select_response_mask(' in worker and 'per_request_sampling(' in worker
    assert "prompts.meta_info[\"eos_token_id\"]" in worker


@pytest.mark.parametrize('selected', [INSTRUCT, protocol.QWEN_COMPLETION_PROTOCOL, 'legacy'])
def test_actual_worker_uses_native_eos_not_generation_config_stop_list(selected):
    path = VENDOR / 'verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'generate_sequences')
    # Execute the real worker's metadata prefix without importing vLLM or allocating a model.
    cutoff = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'batch_size')
    fn.body = fn.body[:cutoff] + [ast.Return(value=ast.Name(id='eos_token_id', ctx=ast.Load()))]
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn],
                        type_ignores=[])
    namespace = {'vllm_version': '0.11.0'}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    tokenizer = Tokenizer()
    worker = SimpleNamespace(config=Config(opd_prompt_protocol=selected),
                             inference_engine=SimpleNamespace(get_tokenizer=lambda: tokenizer))
    prompts = SimpleNamespace(batch={key: torch.tensor([[1]]) for key in
                                    ('input_ids', 'attention_mask', 'position_ids')},
                              meta_info={'eos_token_id': [151645, 151643]})
    actual = namespace['generate_sequences'](worker, prompts)
    assert actual == (151645 if selected == INSTRUCT else [151645, 151643])
    assert prompts.meta_info['eos_token_id'] == [151645, 151643]
    if selected == INSTRUCT:
        tokenizer.eos_token_id = 151643
        with pytest.raises(ValueError):
            namespace['generate_sequences'](worker, prompts)


@pytest.mark.parametrize('model_var,revision_var', [('STUDENT_MODEL', 'STUDENT_MODEL_REVISION'),
                                                  ('MATH_TEACHER', 'TEACHER_MODEL_REVISION')])
def test_launcher_checks_real_modelscope_revision(tmp_path, model_var, revision_var):
    source = (ROOT / 'scripts/launch_revisiting_block_opd_formal_train.sh').read_text()
    start = source.index('if [[ -n \'${' + revision_var + '}\' ]]')
    block = source[start:source.index('\nfi', start) + 3]
    # Reproduce the outer double-quoted launcher expansion, without run_on_target/SSH.
    block = block.replace('\\"', '"').replace('\\$', '$')
    for name, value in {model_var: str(tmp_path), revision_var: 'modelscope-real', 'OPD_PROMPT_PROTOCOL': INSTRUCT}.items():
        block = block.replace('${' + name + '}', value)
    (tmp_path / 'SOURCE_REVISION').write_text('modelscope-real\n')
    result = subprocess.run(['bash', '-ec', block], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    (tmp_path / 'SOURCE_REVISION').write_text('wrong\n')
    assert subprocess.run(['bash', '-ec', block], capture_output=True).returncode != 0
    assert not (tmp_path / 'HF_REVISION').exists()


@pytest.mark.parametrize('selected,stops,mask', [
    ('legacy', None, None), (protocol.PROTOCOL, '[151643,151645]', None),
    (protocol.QWEN_COMPLETION_PROTOCOL, '[]', 'true'), (protocol.QWEN_HISTORICAL_PROTOCOL, '[]', 'true'),
    (protocol.LLAMA_PROTOCOL, '[128001,128008,128009]', None),
    (protocol.LLAMA_HISTORICAL_PROTOCOL, '[128001,128008,128009]', None)])
def test_old_protocol_shell_contracts_unchanged(selected, stops, mask):
    result = shell_protocol_args(selected, 'legacy')
    assert result.returncode == 0, result.stderr
    if stops:
        assert '+actor_rollout_ref.rollout.stop_token_ids=' + stops in result.stdout
    if mask:
        assert '+actor_rollout_ref.rollout.preserve_legacy_response_mask=' + mask in result.stdout
    else:
        assert 'preserve_legacy_response_mask' not in result.stdout


def test_canonical_vendor_patch_and_runtime_manifest_match():
    patch = ROOT / 'patches/revisiting_opd/blockwise_sampled_opd.patch'
    result = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)],
                            cwd=VENDOR, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for line in (ROOT / 'manifests/revisiting_opd_runtime.sha256').read_text().splitlines():
        digest, path = line.split('  ', 1)
        assert hashlib.sha256((VENDOR / path).read_bytes()).hexdigest() == digest, path

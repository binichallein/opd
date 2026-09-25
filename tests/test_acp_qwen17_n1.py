import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    path = ROOT / 'scripts/run_acp_qwen17_n1.py'
    assert path.exists(), 'ACP 100-step adapter is missing'
    spec = importlib.util.spec_from_file_location('acp_n1_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_approved_order_and_fail_closed(module):
    calls = []
    module.execute_ordered(lambda v: calls.append(('train', v)) or {'passed': True},
                           lambda v: calls.append(('eval', v)) or {'passed': True})
    assert calls == [(kind, name) for v in ('block3_mean', 'token_opd')
                     for kind, name in [('train', v), *[('eval', f'{v}_step{s}') for s in (100, 75, 50, 25)]]]
    trained = []
    with pytest.raises(ValueError):
        module.execute_ordered(lambda v: trained.append(v) or {'passed': True}, lambda _: {'passed': False})
    assert trained == ['block3_mean']


@pytest.mark.parametrize('variant', ['block3_mean', 'token_opd'])
@pytest.mark.parametrize('probe', [None, 1, 2])
def test_exact_training_contract(module, variant, probe):
    root = module.ROOT / 'runs/test'
    env = module.training_env(Path('/runtime'), 'a' * 40, root, variant, probe)
    assert env['REMOTE'] == 'acp'
    assert env['TOTAL_TRAINING_STEPS'] == '100'
    assert env['TRAIN_BATCH_SIZE'] == '32' and env['ROLLOUT_GROUP_SIZE'] == '1'
    assert env['PPO_MINI_BATCH_SIZE'] == '32'
    assert env['OPD_REQUEST_SEED_RULE'] == 'legacy'
    assert env['OPD_PROMPT_PROTOCOL'] == 'qwen3_native_chat_no_thinking_boxed_v1'
    assert env['STUDENT_MODEL'] == str(module.ROOT / 'models/Qwen3-1.7B')
    assert env['MATH_TEACHER'] == str(module.ROOT / 'models/Qwen3-8B')
    assert env['VENV'].startswith(str(module.ROOT / 'envs'))
    assert env['RAY_NUM_CPUS'] == '32'
    assert env['STOP_AFTER_STEP'] == str(probe or -1)
    assert env['DIAGNOSTIC_SAVE_STEPS'] == {None: '25,50,75,100', 1: '1', 2: '1,2'}[probe]
    assert env['RESUME_MODE'] == ('resume_path' if probe == 2 else 'disable')
    assert env['RESUME_FROM_PATH'] == (str(root / variant / 'checkpoints/global_step_1') if probe == 2 else '')


def test_card_rejects_wrong_seed_prompt_and_horizon(module):
    source = dict(seed=21, request_seed_rule='legacy', opd_prompt_protocol=module.PROTOCOL,
                  total_training_steps=100, diagnostic_save_steps='25,50,75,100')
    module.require_equal_card(source, source)
    for k in source:
        with pytest.raises(ValueError):
            module.require_equal_card(source, dict(source, **{k: 'changed'}))


def test_paths_and_host_fail_closed(module):
    module.validate_location(module.HOST, module.ROOT, 'fuse.quarkfs_client')
    for args in [('ml2', module.ROOT, 'fuse.quarkfs_client'),
                 (module.HOST, Path('/workspace/opd'), 'overlay'),
                 (module.HOST, module.ROOT, 'overlay')]:
        with pytest.raises(ValueError):
            module.validate_location(*args)


def test_rollout_requires_native_eos_real_lengths_and_legacy_seed(module):
    row = dict(protocol=module.PROTOCOL, enable_thinking=False, step=1,
               prompt_token_ids=[1, 2], eos_token_id=151645, response_length=2,
               response_token_ids=[3, 151645], response_tensor_width=16384,
               padding_length=16382, response_mask=[1, 1], rollout_log_probs=[-.2, -.1],
               finish_reason='stop', stop_reason=None,
               sampling=dict(temperature=1., top_p=.9, top_k=-1, max_tokens=16384, n=1,
                             ignore_eos=False, stop_token_ids=[151645, 151643], seed=21))
    module.validate_rollout(row, [1, 2], 1)
    for change in [dict(enable_thinking=True), dict(eos_token_id=151643),
                   dict(response_mask=[1, 0]), dict(rollout_log_probs=[float('nan'), -.1]),
                   dict(sampling=dict(row['sampling'], seed=22)), dict(stop_reason=151643)]:
        with pytest.raises(ValueError):
            module.validate_rollout(dict(row, **change), [1, 2], 1)


def test_launcher_accepts_only_fixed_afs_deployment_prefix():
    launcher = (ROOT / 'scripts/launch_revisiting_block_opd_formal_train.sh').read_text()
    assert '/mnt/afs/202609/tyf-qwen-opd/deployments/' in launcher
    assert '"${REMOTE}" == acp' in launcher
    subprocess.run(['bash', '-n', str(ROOT / 'scripts/launch_revisiting_block_opd_formal_train.sh')], check=True)


def test_historical_grading_gate_requires_every_saved_result(module, monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(module.q, 'QUALIFICATION', tmp_path)
    monkeypatch.setattr(module.q.qualify.old, 'load_historical_grader', lambda _: (None, None))
    monkeypatch.setattr(module.q.qualify.old, 'score_records', lambda rows, *_: rows)
    for cell in ('smoke', 'direct', 'continuation'):
        for role in ('student', 'teacher'):
            folder = tmp_path / cell / role
            folder.mkdir(parents=True)
            record = dict(correct=True, extracted_answer='2')
            for name in ('raw.jsonl', 'results.jsonl'):
                (folder / name).write_text(json.dumps(record)+'\n')
    assert module.grading_gate()['num_records'] == 6
    (tmp_path / 'direct/teacher/results.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Historical grading differs'):
        module.grading_gate()

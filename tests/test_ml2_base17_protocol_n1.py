import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    path = ROOT / 'scripts/run_ml2_base17_protocol_n1.py'
    assert path.exists(), 'Missing authorized ml2 protocol comparison queue'
    spec = importlib.util.spec_from_file_location('ml2_protocol_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('variant', ['block3_mean', 'token_opd'])
@pytest.mark.parametrize('probe', [None, 1, 2])
def test_training_contract(module, variant, probe):
    run = module.configured_runner()
    run.configure()
    env = run.training_env(Path('/runtime'), 'a' * 40, module.RUN_ROOT, variant, probe)
    card = run.expected_card(Path('/runtime'), 'a' * 40, module.RUN_ROOT, variant, probe)
    assert env['REMOTE'] == 'ml2' and env['RAY_NUM_CPUS'] == '64'
    assert env['STUDENT_MODEL'] == str(module.STUDENT)
    assert env['MATH_TEACHER'] == str(module.TEACHER)
    assert env['TOTAL_TRAINING_STEPS'] == '100'
    assert env['TRAIN_BATCH_SIZE'] == '32' and env['ROLLOUT_GROUP_SIZE'] == '1'
    assert env['PPO_MINI_BATCH_SIZE'] == '32' and env['LEARNING_RATE'] == '2e-6'
    assert env['OPD_REQUEST_SEED_RULE'] == 'legacy'
    assert env['OPD_PROMPT_PROTOCOL'] == 'qwen3_completion_boxed_v1'
    assert card['opd_block_ablation'] == 'legacy'
    assert card['opd_block_size'] == (3 if variant == 'block3_mean' else 1)
    assert card['opd_block_advantage_mode'] == ('mean' if variant == 'block3_mean' else 'sum')
    assert card['total_training_steps'] == 100
    assert card['diagnostic_save_steps'] == {None:'25,50,75,100',1:'1',2:'1,2'}[probe]
    assert card['resume_mode'] == ('resume_path' if probe == 2 else 'disable')
    assert str(module.BASELINE_ROOT) not in env['RUN_ROOT']
    assert run.HARDWARE == '4xA100-80GB; 64 Ray CPUs'
    assert 'ACP' not in run.LIFETIME


def test_host_guard_and_order(module):
    run = module.configured_runner()
    run.validate_location('di-20260407234928-vrvxk', module.ROOT, 'nfs4')
    with pytest.raises(ValueError):
        run.validate_location('acp', module.ROOT, 'nfs4')
    events = []
    run.execute_ordered(lambda v: events.append(('train',v)) or {'passed':True},
                        lambda v: events.append(('eval',v)) or {'passed':True})
    assert events == [(kind,name) for v in ('block3_mean','token_opd') for kind,name in
                     [('train',v), *[('eval',f'{v}_step{s}') for s in (100,75,50,25)]]]


def test_evaluation_is_matched_completion_full_n8(module):
    run = module.configured_runner()
    run.configure()
    for variant in run.VARIANTS:
        for step in (25,50,75,100):
            checkpoint, model = run.q.model_paths(module.RUN_ROOT, f'{variant}_step{step}')
            assert checkpoint == module.RUN_ROOT / variant / f'checkpoints/global_step_{step}/actor'
            cmd = run.q.evaluation_command(Path('/runtime'), model, Path('/out'))
            assert cmd[cmd.index('--prompt-protocol')+1] == run.PROTOCOL
            assert cmd[cmd.index('--grader')+1] == 'external'
            assert cmd[cmd.index('--n')+1] == '8'
            assert cmd[cmd.index('--eval-seed')+1] == '21'
            assert '--retain-rollouts' in cmd


def test_old_new_card_comparison_rejects_nonprotocol_drift(module):
    original = dict(seed=21,request_seed_rule='legacy',learning_rate=2e-6,
                    opd_prompt_protocol='qwen3_historical17_v1',opd_block_size=3,
                    opd_block_ablation='legacy',student_model=str(module.STUDENT),total_training_steps=100)
    new = dict(original,opd_prompt_protocol='qwen3_completion_boxed_v1',source_commit='new')
    module.validate_baseline_card(new, original)
    for key in ('seed','request_seed_rule','learning_rate','opd_block_size',
                'opd_block_ablation','student_model','total_training_steps'):
        with pytest.raises(ValueError):
            module.validate_baseline_card(dict(new,**{key:'changed'}),original)


def valid_row():
    ids = [12,13,151643] + [151643]*16381
    return dict(protocol='qwen3_completion_boxed_v1',enable_thinking=False,step=1,
        prompt_token_ids=[11],eos_token_id=151643,mask_policy='historical_eos_mask',
        sampling=dict(temperature=1.,top_p=.9,top_k=-1,max_tokens=16384,n=1,
                      ignore_eos=False,stop_token_ids=[],seed=21),
        training_response_token_ids=ids,response_token_ids=ids[:3],response_length=3,
        response_tensor_width=16384,padding_length=16381,response_mask=[1]*3,
        training_response_mask=[1]*3+[0]*16381,rollout_log_probs=[-.1]*3,
        training_rollout_log_probs=[-.1]*16384,finish_reason='stop',stop_reason=None)


def test_legacy_seed_completion_mask_and_eos(module):
    row = valid_row()
    module.validate_rollout(row,[11],1)
    changed = copy.deepcopy(row)
    changed['response_token_ids'][-1] = 151645
    changed['training_response_token_ids'][2] = 151645
    with pytest.raises(ValueError):
        module.validate_rollout(changed,[11],1)
    for change in ('seed','stop','mask','length','prompt','think','nonfinite'):
        changed = copy.deepcopy(row)
        if change == 'seed': changed['sampling']['seed'] = 22
        if change == 'stop': changed['sampling']['stop_token_ids'] = [151645]
        if change == 'mask': changed['training_response_mask'][3] = 1
        if change == 'length': changed['response_length'] = 4
        if change == 'prompt': changed['prompt_token_ids'] = [99]
        if change == 'think': changed['enable_thinking'] = True
        if change == 'nonfinite': changed['training_rollout_log_probs'][0] = float('nan')
        with pytest.raises(ValueError):
            module.validate_rollout(changed,[11],1)


def test_literal_prompt_has_no_chat_or_think(module):
    from opd_ext.math_protocol import completion_math_prompt
    prompt = completion_math_prompt('What is 1+1?')
    assert 'What is 1+1?' in prompt and 'boxed' in prompt and prompt.endswith('Solution:\n')
    assert '<|im_start|>' not in prompt and '<think>' not in prompt

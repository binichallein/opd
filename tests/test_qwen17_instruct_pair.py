import copy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@pytest.fixture
def subject():
    return importlib.import_module('run_qwen17_instruct_pair')


def test_order_block_eval_base_token_eval(subject):
    events = []
    def train(v):
        events.append(('train', v))
        return {'passed': True}
    def evaluate(v):
        events.append(('eval', v))
        return {'passed': True}
    results = subject.execute_ordered(train, evaluate)
    assert events == [('train', 'block3_mean')] + [
        ('eval', f'block3_mean_step{s}') for s in (200, 150, 100, 50)
    ] + [('eval', 'student_base'), ('train', 'token_opd')] + [
        ('eval', f'token_opd_step{s}') for s in (200, 150, 100, 50)]
    assert len(results) == 9


@pytest.mark.parametrize('failure', ['train', 'block3_mean_step100', 'student_base'])
def test_no_token_after_failed_block_or_eval(subject, failure):
    events = []
    def train(v):
        events.append(v)
        return {'passed': failure != 'train'}
    def evaluate(name):
        return {'passed': name != failure}
    with pytest.raises(ValueError):
        subject.execute_ordered(train, evaluate)
    assert events == ['block3_mean']


@pytest.mark.parametrize('variant', ['block3_mean', 'token_opd'])
@pytest.mark.parametrize('probe', [None, 1, 2])
def test_environment_and_checkpoint_policy(subject, tmp_path, variant, probe):
    env = subject.training_env(tmp_path/'runtime', 'abc', tmp_path, variant, probe)
    assert env['STUDENT_MODEL'] == str(subject.STUDENT)
    assert env['MATH_TEACHER'] == str(subject.TEACHER)
    assert env['STUDENT_MODEL_REVISION'] == subject.STUDENT_REVISION
    assert env['TEACHER_MODEL_REVISION'] == subject.TEACHER_REVISION
    assert env['TOTAL_TRAINING_STEPS'] == ('200' if probe is None else '2')
    assert env['DIAGNOSTIC_SAVE_STEPS'] == {None:'50,100,150,200',1:'1',2:'1,2'}[probe]
    assert env['OPD_DIAG_INTERVAL'] == '1'
    assert env['ROLLOUT_GPU_MEMORY_UTILIZATION'] == '0.6'
    assert env['ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU'] == '1'
    assert env['REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU'] == '1'
    assert env['RESUME_MODE'] == ('resume_path' if probe == 2 else 'disable')
    assert env['OPD_PROMPT_PROTOCOL'] == subject.PROTOCOL
    assert env['OPD_REQUEST_SEED_RULE'] == subject.SEED_RULE
    if probe == 2:
        assert env['RESUME_FROM_PATH'] == str(tmp_path/variant/'checkpoints/global_step_1')
    else:
        assert env['RESUME_FROM_PATH'] == ''


def test_prepared_card_pair_and_drift(subject, tmp_path):
    cards = {v: subject.expected_card(tmp_path, v, 'abc') for v in subject.VARIANTS}
    subject.validate_pair(cards, tmp_path, 'abc')
    bad = copy.deepcopy(cards)
    bad['token_opd']['learning_rate'] *= 3
    with pytest.raises(ValueError):
        subject.validate_pair(bad, tmp_path, 'abc')
    bad = copy.deepcopy(cards)
    bad['block3_mean']['diagnostic_save_steps'] = '50,100,200'
    with pytest.raises(ValueError):
        subject.validate_pair(bad, tmp_path, 'abc')


def test_model_paths_never_warm_start(subject, tmp_path):
    assert subject.model_paths(tmp_path, 'student_base') == (None, subject.STUDENT)
    actor, model = subject.model_paths(tmp_path, 'block3_mean_step150')
    assert actor == tmp_path/'block3_mean/checkpoints/global_step_150/actor'
    assert model == tmp_path/'merged/block3_mean_step150'
    for name in ['teacher', 'block3_mean_step1', '../other', 'token_opd_step250']:
        with pytest.raises(ValueError):
            subject.model_paths(tmp_path, name)


def test_eval_command_is_complete_and_external(subject, tmp_path):
    cmd = subject.evaluation_command(tmp_path/'runtime', tmp_path/'model', tmp_path/'out')
    assert cmd[cmd.index('--grader')+1] == 'external'
    assert cmd[cmd.index('--n')+1] == '8'
    assert cmd[cmd.index('--prompt-protocol')+1] == subject.PROTOCOL
    assert '--retain-rollouts' in cmd
    assert cmd[cmd.index('--tasks')+1:cmd.index('--n')] == ['math500','aime24','aime25','amc23']


@pytest.mark.parametrize('probe', [None,1,2])
def test_audit_has_all_milestones_and_every_step(subject, tmp_path, probe):
    cmd = subject.audit_command(tmp_path/'runtime', tmp_path/'block3_mean', 'abc', probe)
    assert cmd[cmd.index('--checkpoint-steps')+1] == {None:'50,100,150,200',1:'1',2:'1,2'}[probe]
    assert cmd[cmd.index('--expected-diag-interval')+1] == '1'
    steps=cmd[cmd.index('--expected-diagnostic-steps')+1]
    assert steps == {None:','.join(map(str,range(1,201))),1:'1',2:'1,2'}[probe]
    assert cmd[cmd.index('--expected-teacher-model-revision')+1] == subject.TEACHER_REVISION


def test_capability_must_be_exact_pinned_instruct_gate(subject):
    good={'pair':'instruct','passed':True,'status':'passed','failures':[],
          'selection_sha256':subject.qualify.SOURCE_SELECTION_SHA,
          'protocol':subject.qualify.protocol('instruct')}
    subject.require_capability(good, subject.CAPABILITY_SHA)
    for patch in [{'pair':'base'}, {'passed':False}, {'status':'inconclusive'}, {'failures':['bad']}]:
        with pytest.raises(ValueError):
            subject.require_capability({**good,**patch},subject.CAPABILITY_SHA)
    with pytest.raises(ValueError):
        subject.require_capability(good, '0'*64)


def sample_rollout(subject, terminal=151645, length=2):
    identity=subject.request_identities([{'index':1,'question':'2+2?'}],group_size=1,global_seed=21,step=1)[0]
    row={'protocol':subject.PROTOCOL,'enable_thinking':False,'step':1,
         'prompt_token_ids':[10,20],'request_identity':identity,'request_turn':0,
         'eos_token_id':151645,'response_length':length,
         'response_token_ids':[3]*(length-1)+[terminal],
         'response_tensor_width':16384,'padding_length':16384-length,
         'response_mask':[1]*length,'rollout_log_probs':[-.5]*length,
         'finish_reason':'length' if length==16384 else 'stop',
         'stop_reason':None if terminal==151645 or length==16384 else terminal,
         'sampling':{'temperature':1.,'top_p':.9,'top_k':-1,'max_tokens':16384,'n':1,
                     'ignore_eos':False,'stop_token_ids':[151645,151643],
                     'seed':subject.request_seed(identity,turn=0)}}
    return row,identity


@pytest.mark.parametrize('terminal,length',[(151645,2),(151643,2),(151645,16384),(3,16384)])
def test_native_eos_secondary_stop_and_cap(subject,terminal,length):
    row,identity=sample_rollout(subject,terminal,length)
    subject.validate_rollout(row,[10,20],identity,1)


@pytest.mark.parametrize('patch',[
    {'enable_thinking':True},{'response_mask':[1,0]}, {'padding_length':0},
    {'rollout_log_probs':[-.5,float('nan')]},{'prompt_token_ids':[20,10]},
    {'stop_reason':151645},{'finish_reason':'length'}, {'request_turn':1},
    {'response_token_ids':[3,4]}, {'response_tensor_width':2},
])
def test_reject_corrupt_rollout(subject,patch):
    row,identity=sample_rollout(subject)
    row.update(patch)
    with pytest.raises(ValueError):
        subject.validate_rollout(row,[10,20],identity,1)


def test_reject_changed_request_seed(subject):
    row,identity=sample_rollout(subject)
    row['sampling']['seed']+=1
    with pytest.raises(ValueError):
        subject.validate_rollout(row,[10,20],identity,1)


def test_paired_rollouts_require_every_identical_input(subject):
    evidence={'passed':True,'total_rollouts':6400,'steps':[
        {'step':s,'count':32,'paired_input_sha256':str(s)} for s in range(1,201)]}
    assert subject.validate_paired_rollouts(evidence,copy.deepcopy(evidence))['passed']
    changed=copy.deepcopy(evidence)
    changed['steps'][149]['paired_input_sha256']='changed'
    with pytest.raises(ValueError):
        subject.validate_paired_rollouts(evidence,changed)


def test_cleanup_own_process_group_only(subject,monkeypatch):
    events=[]
    monkeypatch.setattr(subject.os,'killpg',lambda pid,sig:events.append((pid,sig)))
    monkeypatch.setattr(subject.time,'sleep',lambda seconds:None)
    subject.cleanup_failed_group(123)
    assert events==[(123,subject.signal.SIGTERM),(123,subject.signal.SIGKILL)]


def test_runtime_helpers_do_not_import_fixed_runtime(subject):
    import inspect
    assert 'from run_qwen4_completion_block3 import' not in inspect.getsource(subject)


@pytest.mark.parametrize('fstype,options,free',[
    ('nfs','rw',100_000_000_000),('tmpfs','rw,noexec',100_000_000_000),('tmpfs','rw',1)])
def test_reject_unsafe_cache(subject,fstype,options,free):
    with pytest.raises(ValueError):
        subject.validate_cache_mount(fstype,options,free)

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def subject():
    assert importlib.util.find_spec('run_qwen17_base_grpo_pair') is not None
    return importlib.import_module('run_qwen17_base_grpo_pair')


def test_isolated_base_profile_and_prior_instruct_defaults():
    m = subject()
    import run_qwen17_instruct_pair as old
    q = m.configured_runner()
    assert q is not old
    assert q.STUDENT.name == 'Qwen3-1.7B-Base'
    assert q.TEACHER.name == 'Qwen3-4B-Base-GRPO'
    assert q.STUDENT_REVISION == 'b0786a09cd6ee101cd8c90e30a5727beb8230544'
    assert q.TEACHER_REVISION == ''
    assert q.EOS_TOKEN_ID == 151643 and old.EOS_TOKEN_ID == 151645
    assert old.STUDENT.name == 'Qwen3-1.7B'
    assert q.PROTOCOL == 'qwen3_completion_boxed_v1'
    assert q.RUN_ROOT == m.RUN_ROOT
    assert q.AUTHORIZATION['capability_passed'] is False
    assert q.AUTHORIZATION['training_authorized'] is True
    assert old.AUTHORIZATION == {}


@pytest.mark.parametrize('variant', ['block3_mean', 'token_opd'])
@pytest.mark.parametrize('probe', [None, 1, 2])
def test_fixed_training_recipe(variant, probe, tmp_path):
    m = subject()
    import run_qwen17_instruct_pair as old
    q = m.configured_runner()
    a = q.training_env(tmp_path, 'abc', tmp_path, variant, probe)
    b = old.training_env(tmp_path, 'abc', tmp_path, variant, probe)
    allowed = {'STUDENT_MODEL', 'MATH_TEACHER', 'STUDENT_MODEL_REVISION', 'TEACHER_MODEL_REVISION',
               'PROJECT_NAME', 'EXP_NAME', 'LOCAL_CACHE_ROOT', 'OPD_PROMPT_PROTOCOL', 'BASELINE_ALIGNMENT'}
    assert {k:v for k,v in a.items() if k not in allowed} == {k:v for k,v in b.items() if k not in allowed}
    assert a['ENV_SEED'] == '21' and a['OPD_DIAG_INTERVAL'] == '1'
    assert a['DIAGNOSTIC_SAVE_STEPS'] == {None:'50,100,150,200',1:'1',2:'1,2'}[probe]
    assert a['TOTAL_TRAINING_STEPS'] == ('2' if probe else '200')
    assert a['RESUME_MODE'] == ('resume_path' if probe == 2 else 'disable')
    q.validate_pair({v:q.expected_card(tmp_path,v,'abc') for v in q.VARIANTS},tmp_path,'abc')


def test_order_and_full_evaluation(tmp_path):
    q = subject().configured_runner()
    events = []
    q.execute_ordered(lambda v:events.append(('train',v)) or {'passed':True},
                      lambda v:events.append(('eval',v)) or {'passed':True})
    assert events == [('train','block3_mean')] + [('eval',f'block3_mean_step{s}') for s in (200,150,100,50)] + [
        ('eval','student_base'),('train','token_opd')] + [('eval',f'token_opd_step{s}') for s in (200,150,100,50)]
    cmd = list(map(str,q.evaluation_command(tmp_path,tmp_path/'model',tmp_path/'eval')))
    assert cmd[cmd.index('--prompt-protocol')+1] == q.PROTOCOL
    assert cmd[cmd.index('--grader')+1] == 'external' and '--retain-rollouts' in cmd
    assert q.shared.TASK_COUNTS == {'math500':500,'aime24':30,'aime25':30,'amc23':83}


def test_only_exact_inconclusive_screen_is_authorized():
    m = subject()
    s = json.loads((ROOT/'results/qwen17_base_grpo_screen_20260923/final/pair_summary.json').read_text())
    before = copy.deepcopy(s)
    m.validate_screen(s,m.CAPABILITY_SHA)
    assert s == before and not s['training_started']
    for key,value in [('passed',True),('status','passed'),('failures',[]),('teacher','i4')]:
        bad=copy.deepcopy(s); bad['reports']['g4_b17'][key]=value
        with pytest.raises(ValueError): m.validate_screen(bad,m.CAPABILITY_SHA)
    bad=copy.deepcopy(s); bad['cell_errors']={'g4_b17_direct':'broken'}
    with pytest.raises(ValueError): m.validate_screen(bad,m.CAPABILITY_SHA)
    with pytest.raises(ValueError): m.validate_screen(s,'wrong')


class BaseTokenizer:
    bos_token_id = None
    eos_token_id = 151643

    def encode(self,text,add_special_tokens=False):
        return [ord(c) for c in text]


def test_base_eval_prompt_no_chat_and_native_eos_only():
    q=subject().configured_runner()
    from opd_ext.math_protocol import completion_math_prompt
    tok=BaseTokenizer()
    ids=q.prompt_input_ids(tok,'What is 1+1?')
    row=dict(problem='What is 1+1?',prompt_protocol=q.PROTOCOL,enable_thinking=False,
             rendered_prompt=completion_math_prompt('What is 1+1?'),prompt_token_ids=ids,
             eos_token_id=151643,sampling={'stop_token_ids':[]})
    q.validate_eval_row(row,tok)
    assert '<think>' not in row['rendered_prompt'] and '<|im_start|>' not in row['rendered_prompt']
    for key,value in [('eos_token_id',151645),('enable_thinking',True),('sampling',{'stop_token_ids':[151645]})]:
        with pytest.raises(ValueError): q.validate_eval_row({**row,key:value},tok)


def test_base_actual_rollout_mask_and_termination():
    q=subject().configured_runner()
    from opd_ext.request_seeds import request_identities,request_seed
    identity=request_identities([{'index':0,'question':'1+1'}],group_size=8,global_seed=21,step=1)[0]
    ids=[7,151643]+[151643]*16382
    row=dict(protocol=q.PROTOCOL,enable_thinking=False,step=1,mask_policy='historical_eos_mask',
        eos_token_id=151643,request_identity=identity,request_turn=0,prompt_token_ids=[11],
        sampling=dict(temperature=1.,top_p=.9,top_k=-1,max_tokens=16384,n=1,ignore_eos=False,
                      stop_token_ids=[],seed=request_seed(identity,turn=0)),
        response_token_ids=ids[:2],response_length=2,response_tensor_width=16384,padding_length=16382,
        response_mask=[1,1],rollout_log_probs=[-.1,-.2],training_response_token_ids=ids,
        training_response_mask=[1,1]+[0]*16382,training_rollout_log_probs=[-.1,-.2]+[0.]*16382,
        finish_reason='stop',stop_reason=None)
    q.validate_rollout(row,[11],identity,1)
    for key,value in [('prompt_token_ids',[12]),('stop_reason',151645),('mask_policy','length'),
                      ('training_response_mask',[1]*16384),('training_rollout_log_probs',[float('nan')]*16384)]:
        with pytest.raises(ValueError): q.validate_rollout({**row,key:value},[11],identity,1)


def test_ray_warmup_precedes_original_training(tmp_path,monkeypatch):
    m=subject(); q=m.configured_runner(); calls=[]
    monkeypatch.setattr(q.base,'read_json',lambda p:{'passed':True})
    def original(*args): calls.append(('train',args[1])); return {'passed':True}
    def runner(cmd,path,**kwargs): calls.append(('gate',list(map(str,cmd)),kwargs))
    result=m.train_with_ray_gate(q,original,tmp_path,'block3_mean',tmp_path,'abc',runner,{})
    assert result['passed'] and calls[1]==('train','block3_mean')
    assert 'run_qwen17_base_grpo_pair.py' in calls[0][1][1]
    assert calls[0][2]['job_env']['CUDA_VISIBLE_DEVICES']==''

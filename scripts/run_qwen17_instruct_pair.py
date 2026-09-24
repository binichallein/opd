#!/usr/bin/env python3
"""ml2: original Instruct Block3 training/eval, then independent Token training/eval."""

import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import prepare_qwen17_pair_assets as assets
import qualify_qwen17_pairs as qualify
import run_qwen06_pair as base
import run_nonthinking_eval_block3 as shared
import run_historical17_reeval_llama as historical
import run_window_queue as jobs
from opd_ext.request_seeds import SEED_RULE, request_identities, request_seed
from opd_ext.math_protocol import qwen_instruct_input_ids as prompt_input_ids

ROOT = assets.ROOT
RUN_ROOT = ROOT / 'runs/20260921v1_qwen17_instruct_blockfirst_seed21_ml2'
CACHE = Path('/dev/shm/q17i')
STUDENT = assets.MODELS['instruct_student']['path']
TEACHER = assets.MODELS['instruct_teacher']['path']
STUDENT_REVISION = assets.MODELS['instruct_student']['revision']
TEACHER_REVISION = assets.MODELS['instruct_teacher']['revision']
PROTOCOL = 'qwen3_native_chat_no_thinking_boxed_v1'
EVALUATION_PROTOCOL = None  # None keeps the training protocol for existing adapters.
TRAIN_EVAL_PROMPT_MATCH = True
EXPECTED_EVAL_COUNT = 9
ORDERING = 'Block3 probe/train/eval -> original student eval -> Token probe/train/eval'
VARIANTS = ('block3_mean', 'token_opd')
STEPS = (200, 150, 100, 50)
SAVE_STEPS = '50,100,150,200'
QUALIFICATION = ROOT / 'runs/20260921v1_qwen8_to17_diagnostics_ml2/qualification/instruct'
CAPABILITY_SHA = '57f4a0559e5d505d4f7b67f31d22ec30e2ec77138b26c5b61f82f98c27052cdb'
HISTORICAL_INPUTS = ROOT / 'runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd'
LOSS_REFERENCE = ROOT / 'deployments/0f9161f02f08287fb07f0375ad0a6bda81133ff0'
ML2_HOST = 'di-20260407234928-vrvxk'
PROJECT_NAME = 'opd_qwen17_instruct'
EXPERIMENT_PREFIX = 'qwen17-instruct'
DISPLAY_LABEL = 'Qwen1.7 Instruct'
EOS_TOKEN_ID = 151645
STUDENT_INITIALIZATION = 'original ModelScope Instruct, independently per arm'
BASELINE_ALIGNMENT = 'Matched original Qwen3 Instruct pair; accepted native nonthinking prompt; historical losses unchanged'
AUTHORIZATION = {}


def validate_host(hostname):
    if hostname != ML2_HOST:
        raise ValueError('This queue is authorized only on the verified ml2 host')


def validate_benchmark_hashes(hashes):
    from audit_block10_run import EXPECTED_EVAL_SHA256
    if hashes != EXPECTED_EVAL_SHA256:
        raise ValueError('Benchmark contents differ from historical approved inputs')


def validate_cache_mount(fstype, options, free_bytes):
    if fstype != 'tmpfs' or 'noexec' in options.split(',') or free_bytes < 20_000_000_000:
        raise ValueError('Executable local tmpfs with at least 20GB free is required')


def cleanup_failed_group(pgid):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(3)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def training_env(runtime, commit, root, variant, probe_step=None):
    env = base.launcher_env(runtime, commit, root, variant, probe_step)
    env.update(PROJECT_NAME=PROJECT_NAME,
               EXP_NAME=f'{EXPERIMENT_PREFIX}-{variant}' + ('-probe' if probe_step else ''),
               STUDENT_MODEL=str(STUDENT), MATH_TEACHER=str(TEACHER),
               STUDENT_MODEL_REVISION=STUDENT_REVISION, TEACHER_MODEL_REVISION=TEACHER_REVISION,
               OPD_PROMPT_PROTOCOL=PROTOCOL, OPD_REQUEST_SEED_RULE=SEED_RULE,
               OPD_DIAG_INTERVAL='1',
               DIAGNOSTIC_SAVE_STEPS={None:SAVE_STEPS,1:'1',2:'1,2'}[probe_step],
               LOSSLESS_ROLLOUT_DIR=str(root/variant/'rollouts'),
               ROLLOUT_ATTEMPT_ID=f'probe{probe_step}' if probe_step else 'formal',
               LOCAL_CACHE_ROOT=str(CACHE/variant/'train'),
               BASELINE_ALIGNMENT=BASELINE_ALIGNMENT)
    return env


def expected_card(root, variant, commit, probe_step=None):
    if variant not in VARIANTS or probe_step not in (None,1,2):
        raise ValueError('Unapproved training variant/probe')
    run = root/variant
    return {**base.paired.EXPECTED_TRAINING_VALUES,
            'source_commit':commit, 'variant':variant, 'opd_window_mode':'fixed',
            'opd_window_seed':910021, 'ppo_epochs':1,
            'opd_block_size':3 if variant=='block3_mean' else 1,
            'opd_block_advantage_mode':'mean' if variant=='block3_mean' else 'sum',
            'student_model':str(STUDENT), 'teacher_model':str(TEACHER),
            'student_model_revision':STUDENT_REVISION, 'teacher_model_revision':TEACHER_REVISION,
            'train_data':str(base.DATA/'train.parquet'), 'val_data':str(base.DATA/'test.parquet'),
            'expected_train_sha256':jobs.TRAIN_SHA, 'opd_prompt_protocol':PROTOCOL,
            'request_seed_rule':SEED_RULE, 'opd_diag_interval':1,
            'total_training_steps':2 if probe_step else 200,
            'diagnostic_save_steps':{None:SAVE_STEPS,1:'1',2:'1,2'}[probe_step],
            'resume_mode':'resume_path' if probe_step==2 else 'disable',
            'resume_from_path':str(run/'checkpoints/global_step_1') if probe_step==2 else '',
            'stop_after_step':1 if probe_step==1 else -1,
            'save_freq':-1, 'test_freq':-1,
            'diagnostic_output_dir':str(run/'diagnostics'), 'lossless_rollout_dir':str(run/'rollouts'),
            'rollout_attempt_id':f'probe{probe_step}' if probe_step else 'formal'}


def validate_card(card, root, variant, commit, probe_step=None):
    for key, expected in expected_card(root, variant, commit, probe_step).items():
        if card.get(key) != expected:
            raise ValueError(f'{variant}: {key} differs: {card.get(key)!r} != {expected!r}')


def validate_pair(cards, root, commit):
    if set(cards) != set(VARIANTS):
        raise ValueError('Both original comparison methods required')
    allowed = {'variant','opd_block_size','opd_block_advantage_mode','experiment_name',
               'diagnostic_output_dir','lossless_rollout_dir'}
    left, right = (cards[v] for v in VARIANTS)
    if any(left.get(k) != right.get(k) for k in (set(left)|set(right))-allowed):
        raise ValueError('Unapproved paired configuration difference')
    for variant in VARIANTS:
        validate_card(cards[variant], root, variant, commit)


def require_capability(result, digest):
    if (digest != CAPABILITY_SHA or result.get('pair')!='instruct' or result.get('passed') is not True
            or result.get('status')!='passed' or result.get('failures')!=[]
            or result.get('selection_sha256')!=qualify.SOURCE_SELECTION_SHA
            or result.get('protocol')!=qualify.protocol('instruct')):
        raise ValueError('The pinned accepted Instruct teacher gate is required')


def execute_ordered(train, evaluate):
    results = {}
    for variant in VARIANTS:
        if train(variant).get('passed') is not True:
            raise ValueError(f'Training acceptance failed: {variant}')
        names = [f'{variant}_step{s}' for s in STEPS]
        if variant=='block3_mean':
            names.append('student_base')
        for name in names:
            result = evaluate(name)
            if result.get('passed') is not True:
                raise ValueError(f'Incomplete evaluation: {name}')
            results[name] = result
    return results


def model_paths(root, name):
    if name=='student_base':
        return None, STUDENT
    if name not in {f'{v}_step{s}' for v in VARIANTS for s in STEPS}:
        raise ValueError('Unapproved evaluation model')
    variant, step = name.rsplit('_step',1)
    return root/variant/f'checkpoints/global_step_{step}/actor', root/'merged'/name


def evaluation_command(runtime, model, output):
    cmd = jobs.eval_command(runtime, base.PYTHON, model, base.DATA/'eval_jsonl', output, STUDENT)
    cmd[cmd.index('--grader')+1] = 'external'
    return cmd + ['--retain-rollouts','--prompt-protocol',EVALUATION_PROTOCOL or PROTOCOL]


def audit_command(runtime, run, commit, probe_step=None):
    cmd = list(map(str, base.audit_command(runtime, run, commit, probe_step=probe_step)))
    changes = {'--expected-student-model-suffix':STUDENT.name,
               '--expected-teacher-model-suffix':TEACHER.name,
               '--expected-student-model-revision':STUDENT_REVISION,
               '--checkpoint-steps':{None:SAVE_STEPS,1:'1',2:'1,2'}[probe_step]}
    for flag, value in changes.items():
        cmd[cmd.index(flag)+1] = value
    cmd += ['--expected-teacher-model-revision',TEACHER_REVISION]
    if not probe_step:
        cmd += ['--expected-diag-interval','1','--expected-diagnostic-steps',','.join(map(str,range(1,201)))]
    return cmd


def preflight(runtime):
    old = qualify.old
    gate = old.read_sealed(QUALIFICATION/'gate_acceptance.json')
    require_capability(gate, assets.sha256(QUALIFICATION/'gate_acceptance.json'))
    protected = {}
    for key in ('instruct_student','instruct_teacher'):
        protected.update(assets.ensure_asset(key))
    protected[str(QUALIFICATION/'gate_acceptance.json')] = CAPABILITY_SHA
    old.verify_hashes(gate['evidence'])
    protected.update(gate['evidence'])
    for label in ('student','teacher'):
        path = QUALIFICATION/'smoke'/label/'non_thinking_acceptance.json'
        accepted = old.read_sealed(path)
        if accepted.get('passed') is not True or accepted.get('num_rollouts')!=4:
            raise ValueError('Missing actual GPU non-thinking acceptance')
        protected[str(path)] = assets.sha256(path)
    protected.update(common_preflight(runtime))
    return protected


def common_preflight(runtime):
    protected = {}
    for path in [base.DATA/'train.parquet',base.DATA/'test.parquet',base.DATA/'manifest.json',base.GRADER,
                 *[base.DATA/'eval_jsonl'/f'{task}.jsonl' for task in jobs.TASKS]]:
        protected[str(path)] = assets.sha256(path)
    if protected[str(base.DATA/'train.parquet')] != jobs.TRAIN_SHA:
        raise ValueError('Changed DAPO training pool')
    validate_benchmark_hashes({task:protected[str(base.DATA/'eval_jsonl'/f'{task}.jsonl')]
                              for task in jobs.TASKS})
    shared.grading.validate_grader_hash(base.GRADER, shared.grading.HISTORICAL_GRADER_SHA256)
    qualify.runtime_versions()
    for name in ('opd_ext/window_supervision.py','opd_ext/diagnostics.py',
                 'external/revisiting_opd/verl/trainer/ppo/core_algos.py',
                 'external/revisiting_opd/verl/workers/actor/dp_actor.py',
                 'external/revisiting_opd/verl/workers/fsdp_workers.py'):
        if assets.sha256(runtime/name)!=assets.sha256(LOSS_REFERENCE/name):
            raise ValueError(f'Unexpected historical loss/diagnostics change: {name}')
    return protected


def input_contract():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl import DataProto
    from agent_system.environments.env_manager import MathEnvironmentManager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from opd_ext.math_protocol import qwen_instruct_input_ids
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(TEACHER, local_files_only=True)
    if tokenizer.get_vocab()!=teacher.get_vocab() or tokenizer.chat_template!=teacher.chat_template:
        raise ValueError('Teacher/student tokenizer or native template differs')
    config = OmegaConf.create({'data':{'opd_prompt_protocol':PROTOCOL,
        'apply_chat_template_kwargs':{'enable_thinking':False},'max_prompt_length':2048,
        'truncation':'middle','return_raw_chat':True}})
    manager = MathEnvironmentManager.__new__(MathEnvironmentManager)
    manager.config = config
    collector = TrajectoryCollector(config,tokenizer)
    selected = qualify.old.read_json(QUALIFICATION/'selected.json')
    for row in selected:
        ids = qwen_instruct_input_ids(tokenizer,row['question'])
        if ids != qualify.prompt_input(tokenizer,row['question'],'instruct')['base_prompt_token_ids']:
            raise ValueError('Training prompt differs from accepted teacher qualification')
    counts, digest = {}, hashlib.sha256()
    for task, n in shared.TASK_COUNTS.items():
        rows=shared.grading.load_jsonl(base.DATA/'eval_jsonl'/f'{task}.jsonl')
        if len(rows)!=n or len({str(r['id']) for r in rows})!=n:
            raise ValueError('Incomplete benchmark identities')
        manager.tasks=[row['problem'] for row in rows]
        observations=manager.build_text_obs(manager.tasks)
        for row, observation in zip(rows,observations):
            question=row['problem']
            raw=np.empty(1,dtype=object); raw[0]=[{'role':'user','content':question}]
            gen=DataProto.from_single_dict({'input_ids':torch.zeros((1,1),dtype=torch.long),
                'raw_prompt':raw,'data_source':np.array(['dapo-math-17k'],dtype=object)})
            processed=collector.preprocess_single_sample(0,gen,{'text':[observation]})
            ids=qwen_instruct_input_ids(tokenizer,question)
            if (processed['raw_prompt_ids']!=ids
                    or processed['input_ids'][processed['attention_mask'].bool()].tolist()!=ids
                    or ids!=qwen_instruct_input_ids(teacher,question)):
                raise ValueError('Actual training collector/eval/teacher tokenization mismatch')
            digest.update(json.dumps([task,row['id'],ids]).encode())
        counts[task]=len(rows)
    return {'passed':True,'actual_collector_exercised':True,'eval_questions_checked':counts,
            'qualification_questions_checked':len(selected),'prompt_ids_sha256':digest.hexdigest(),
            'protocol':PROTOCOL,'enable_thinking':False,'stop_token_ids':[151645,151643]}


def read_archive(path):
    digest=path.with_suffix(path.suffix+'.sha256').read_text().split()[0]
    if assets.sha256(path)!=digest:
        raise ValueError('Raw archive hash mismatch')
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream]


def validate_rollout(row, ids, identity, step):
    count=row['response_length']
    sampling={'temperature':1.,'top_p':.9,'top_k':-1,'max_tokens':16384,'n':1,
              'ignore_eos':False,'stop_token_ids':[151645,151643],'seed':request_seed(identity,turn=0)}
    if (row['protocol']!=PROTOCOL or row['enable_thinking'] is not False
            or row['step']!=step or row['prompt_token_ids']!=ids or row['request_identity']!=identity
            or row['request_turn']!=0 or row['eos_token_id']!=151645
            or any(row['sampling'].get(k)!=v for k,v in sampling.items())
            or count!=len(row['response_token_ids']) or not 0<count<=16384
            or row['response_tensor_width']!=16384 or row['padding_length']!=16384-count
            or row['response_mask']!=[1]*count or len(row['rollout_log_probs'])!=count
            or not all(math.isfinite(v) for v in row['rollout_log_probs'])):
        raise ValueError('Actual rollout input, seed, mask or logprob contract violated')
    if row['finish_reason']=='length':
        if count!=16384 or row['stop_reason'] is not None:
            raise ValueError('Invalid length termination')
    elif row['finish_reason']=='stop':
        terminal=row['response_token_ids'][-1]
        if count>=16384 or not ((terminal==151645 and row['stop_reason'] is None)
                              or (terminal==151643 and row['stop_reason']==151643)):
            raise ValueError('Invalid native EOS/explicit stop termination')
    else:
        raise ValueError('Unexpected generation finish reason')


def audit_rollouts(run, steps, *, probe=False):
    from transformers import AutoTokenizer
    from diagnose_token_truncation import analyze_tokens
    tokenizer=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    evidence=[]
    for step in steps:
        paths=list((run/'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(paths)!=1:
            raise ValueError('Missing or duplicate archived training step')
        rows=read_archive(paths[0])
        original=read_archive(HISTORICAL_INPUTS/f'rollouts/formal/step_{step:06d}/raw.jsonl.gz')
        if len(rows)!=32 or len(original)!=32 or len({r['traj_uid'] for r in rows})!=32:
            raise ValueError('Incomplete training trajectories')
        sources=[original[i]['source_extra_info'] for i in range(0,32,8)]
        identities=request_identities(sources,group_size=8,global_seed=21,step=step)
        fingerprint=[]
        for row, old, identity in zip(rows,original,identities):
            if row['source_extra_info']!=old['source_extra_info']:
                raise ValueError('Historical DAPO source order changed')
            validate_rollout(row,prompt_input_ids(tokenizer,row['source_extra_info']['question']),identity,step)
            if probe and row['generated_think_tags']:
                raise ValueError('GPU training rollout generated thinking tags; preserve probe and stop')
            fingerprint.append([row[k] for k in ('source_extra_info','prompt_token_ids','request_identity','sampling')])
        evidence.append({'step':step,'count':32,'sha256':assets.sha256(paths[0]),
            'paired_input_sha256':qualify.old.object_hash(fingerprint),
            'length_stops':sum(r['finish_reason']=='length' for r in rows),
            'generated_think_tags':sum(r['generated_think_tags'] for r in rows),
            'periodic_tails':sum(analyze_tokens(r['response_token_ids'],16384)['tail_period'] is not None for r in rows)})
    return {'passed':True,'steps':evidence,'total_rollouts':32*len(evidence)}


def validate_paired_rollouts(block, token):
    if (block.get('passed') is not True or token.get('passed') is not True
            or block.get('total_rollouts')!=6400 or token.get('total_rollouts')!=6400
            or len(block['steps'])!=200 or len(token['steps'])!=200):
        raise ValueError('Both complete 200-step rollout audits required')
    for step, (left,right) in enumerate(zip(block['steps'],token['steps']),1):
        if (left['step']!=step or right['step']!=step or left['count']!=32 or right['count']!=32
                or left['paired_input_sha256']!=right['paired_input_sha256']):
            raise ValueError(f'Paired data/prompt/request-seed drift at step {step}')
    return {'passed':True,'steps':200,'trajectories_per_arm':6400}


def validate_eval_row(row, tokenizer):
    from opd_ext.math_protocol import qwen_instruct_input_ids, qwen_instruct_render
    if (row.get('prompt_protocol')!=PROTOCOL or row.get('enable_thinking') is not False
            or row.get('rendered_prompt')!=qwen_instruct_render(tokenizer,row['problem'])
            or row.get('prompt_token_ids')!=qwen_instruct_input_ids(tokenizer,row['problem'])
            or row.get('eos_token_id')!=151645
            or row.get('sampling',{}).get('stop_token_ids')!=[151645,151643]):
        raise ValueError('Evaluation input/thinking/stopping differs from training')


def evaluate_model(root, name, runtime, commit, runner, protected):
    from transformers import AutoTokenizer
    shared.verify_hashes(protected)
    folder=root/'evaluations'/name
    folder.mkdir(parents=True,exist_ok=False)
    actor, model=model_paths(root,name)
    if actor is not None:
        if model.exists():
            raise FileExistsError(model)
        checkpoint_hashes={str(actor/f'model_world_size_4_rank_{rank}.pt'):
                           assets.sha256(actor/f'model_world_size_4_rank_{rank}.pt') for rank in range(4)}
        jobs.write_json(folder/'checkpoint_identity.json',{'checkpoint':str(actor),'sha256':checkpoint_hashes})
        runner([base.PYTHON,runtime/'external/revisiting_opd/scripts/model_merger.py','merge',
                '--backend','fsdp','--local_dir',actor,'--target_dir',model],
               root/f'queue_jobs/merge_{name}',job_env={'CUDA_VISIBLE_DEVICES':''})
        shared.verify_hashes(checkpoint_hashes)
    if not list(model.glob('*.safetensors')):
        raise ValueError('Missing merged/original evaluation weights')
    hashes={str(p):assets.sha256(p) for p in model.iterdir() if p.is_file()}
    jobs.write_json(folder/'model_identity.json',{'model':str(model),'sha256':hashes})
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    reference=AutoTokenizer.from_pretrained(STUDENT,local_files_only=True)
    if (tokenizer.get_vocab()!=reference.get_vocab() or tokenizer.chat_template!=reference.chat_template
            or tokenizer.eos_token_id!=EOS_TOKEN_ID):
        raise ValueError('Merged model tokenizer/template/EOS changed')
    jobs.write_json(folder/'eval_card.json',{**shared.EVAL_VALUES,'model':str(model),'role':name,
        'source_commit':commit,'tasks':shared.TASK_COUNTS,'retain_rollouts':True,
        'prompt_protocol':EVALUATION_PROTOCOL or PROTOCOL,'grader_sha256':shared.grading.HISTORICAL_GRADER_SHA256,
        'reporting':'Each benchmark separately; no macro score'})
    runner(evaluation_command(runtime,model,folder/'outputs'),folder,gpu=True,
           pid_name='eval.pid',log_name='eval.log')
    shared.verify_hashes(hashes)
    shared.verify_hashes(protected)
    config=base.read_json(folder/'outputs/eval_config.json')
    if config.get('prompt_protocol')!=(EVALUATION_PROTOCOL or PROTOCOL) or config.get('retain_rollouts') is not True:
        raise ValueError('Wrong full evaluation prompt/retention protocol')
    accepted=shared.audit_evaluation(folder/'outputs',base.DATA/'eval_jsonl',model)
    for task in shared.TASK_COUNTS:
        rows=shared.grading.load_jsonl(shared.grading.raw_output_path(folder/'outputs',task,config))
        for row in rows:
            validate_eval_row(row,tokenizer)
        accepted['per_task'][task]['generated_think_tag_count']=sum(
            any(t in (151667,151668) for t in row['response_token_ids']) for row in rows)
    accepted['train_eval_prompt_verified']=TRAIN_EVAL_PROMPT_MATCH
    accepted['evaluation_protocol_verified']=True
    return historical.accept_archive(folder,accepted)


def write_comparison(root, results):
    expected={'student_base'} | {f'{v}_step{s}' for v in VARIANTS for s in STEPS}
    if set(results)!=expected or any(r.get('passed') is not True for r in results.values()):
        raise ValueError('All nine complete evaluations required for comparison')
    report={'student':str(STUDENT),'teacher':str(TEACHER),'protocol':PROTOCOL,
            'training_seed':21,'primary_checkpoint':200,'per_benchmark':{}}
    for task in shared.TASK_COUNTS:
        values={name:result['per_task'][task] for name,result in results.items()}
        for step in sorted(STEPS):
            block,token=(results[f'{v}_step{step}']['per_task'][task] for v in VARIANTS)
            values[f'block_minus_token_step{step}_pp']={key:100*(block[key]-token[key])
                                                      for key in ('avg_at_8','pass_at_8')}
        report['per_benchmark'][task]=values
    jobs.write_json(root/'paired_comparison.json',report)


def train_variant(root, variant, runtime, commit, runner, protected):
    import run_qwen8_teacher_pair as checkpoint_helpers
    probe_root=root/'probes'
    probe=probe_root/variant
    for step in (1,2):
        runner(['bash',runtime/'scripts/launch_revisiting_block_opd_formal_train.sh'],
               root/f'queue_jobs/prepare_{variant}_probe{step}',
               job_env=training_env(runtime,commit,probe_root,variant,step))
        validate_card(base.read_json(probe/'run_card.json'),probe_root,variant,commit,step)
        job=root/f'queue_jobs/{variant}_probe{step}'
        runner(['bash',probe/'command.sh'],job,gpu=True)
        shutil.copyfile(job/'logs/job.log',probe/'logs/nohup.log')
        runner(audit_command(runtime,probe,commit,step),root/f'queue_jobs/{variant}_probe_audit{step}')
        jobs.write_json(probe/f'rollout_acceptance_step{step}.json',audit_rollouts(probe,range(1,step+1),probe=True))
        checkpoint_helpers.verify_optimizer_state(probe,(step,))
    base.audit_resume(probe)
    shared.verify_hashes(protected)
    run=root/variant
    validate_card(base.read_json(run/'run_card.json'),root,variant,commit)
    runner(['bash',run/'command.sh'],run,gpu=True,pid_name='train.pid',log_name='nohup.log')
    runner(audit_command(runtime,run,commit),root/f'queue_jobs/{variant}_checkpoint_audit')
    checkpoint_helpers.verify_formal_training_states(run)
    evidence=audit_rollouts(run,range(1,201))
    jobs.write_json(run/'rollout_acceptance.json',evidence)
    if variant=='token_opd':
        paired=validate_paired_rollouts(base.read_json(root/'block3_mean/rollout_acceptance.json'),evidence)
        jobs.write_json(root/'paired_rollout_acceptance.json',paired)
    runner([base.PLOT_PYTHON,runtime/'scripts/analyze_single_opd_diagnostics.py',
            '--run-dir',run,'--output-dir',run/'figures','--label',f'{DISPLAY_LABEL} {variant} seed21'],
           root/f'queue_jobs/{variant}_figures')
    shared.verify_hashes(protected)
    return {'passed':True,'variant':variant,'steps':200,'checkpoint_steps':sorted(STEPS)}


def main():
    from recover_historical17_llama import check_socket_budget
    validate_host(socket.gethostname())
    runtime=Path(__file__).resolve().parents[1]
    commit=(runtime/'DEPLOYED_COMMIT').read_text().strip()
    if runtime!=ROOT/'deployments'/commit:
        raise ValueError('New immutable ml2 training runtime required')
    sys.path.insert(0,str(runtime/'external/revisiting_opd'))
    def interrupted(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state=RUN_ROOT/'queue_state.json'
    with (RUN_ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN_ROOT/'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt: never overwrite, retry or duplicate queue')
        (RUN_ROOT/'queue.pid').write_text(str(os.getpid())+'\n')
        jobs.write_json(RUN_ROOT/'queue_manifest.json',{
            'source_commit':commit,'created_at':jobs.now(),'student':str(STUDENT),'teacher':str(TEACHER),
            'student_revision':STUDENT_REVISION,'teacher_revision':TEACHER_REVISION,
            'variants':VARIANTS,'total_training_steps':200,'checkpoint_steps':sorted(STEPS),
            'eval_steps':STEPS,'eval_tasks':shared.TASK_COUNTS,'eval_values':shared.EVAL_VALUES,
            'ordering':ORDERING,
            'retain_all_checkpoints':True,'retain_all_rollouts':True,'full_eval_autostart':True,
            'student_initialization':STUDENT_INITIALIZATION,
            'prompt_protocol':PROTOCOL,'request_seed_rule':SEED_RULE,'capability_gate_sha256':CAPABILITY_SHA,
            'authorization':AUTHORIZATION})
        try:
            jobs.wait_for_idle()
            if shutil.disk_usage(ROOT).free<1_000_000_000_000:
                raise ValueError('Less than 1TB free; checkpoint pruning is not authorized')
            CACHE.mkdir(exist_ok=True)
            mount=subprocess.run(['findmnt','-T',str(CACHE),'-n','-o','FSTYPE,OPTIONS'],
                                 check=True,capture_output=True,text=True).stdout.strip().split(maxsplit=1)
            validate_cache_mount(*mount,shutil.disk_usage(CACHE).free)
            for variant in VARIANTS:
                check_socket_budget(CACHE/variant/'train/tmp')
            env=dict(os.environ,PATH=str(base.PYTHON.parent)+':'+os.environ.get('PATH',''),
                PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd',CUDA_VISIBLE_DEVICES='0,1,2,3',
                PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
                RAY_DEDUP_LOGS='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',ENGINE='vllm',
                VLLM_WORKER_MULTIPROC_METHOD='spawn',EVAL_GRADE_UTILS_PATH=str(base.GRADER))
            if env.get('RAY_TMPDIR'):
                raise ValueError('Unexpected inherited RAY_TMPDIR')
            for key,suffix in {'TMPDIR':'tmp','VLLM_CACHE_ROOT':'vllm','TRITON_CACHE_DIR':'triton',
                               'TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda',
                               'OUTLINES_CACHE_DIR':'outlines'}.items():
                (CACHE/suffix).mkdir(exist_ok=True)
                env[key]=str(CACHE/suffix)
            def runner(command,job,job_env=None,**kwargs):
                try:
                    jobs.run_job(command,job,runtime,state,{**env,**(job_env or {})},**kwargs)
                except BaseException:
                    pidfile=job/kwargs.get('pid_name','job.pid')
                    if kwargs.get('gpu') and pidfile.exists():
                        cleanup_failed_group(int(pidfile.read_text()))
                    raise
            runner(['sha256sum','-c','.expected.sha256'],RUN_ROOT/'queue_jobs/runtime_hashes')
            protected=preflight(runtime)
            jobs.write_json(RUN_ROOT/'protected_inputs.json',protected)
            jobs.write_json(RUN_ROOT/'input_contract.json',input_contract())
            cards={}
            for variant in VARIANTS:
                runner(['bash',runtime/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                       RUN_ROOT/f'queue_jobs/prepare_{variant}_formal',
                       job_env=training_env(runtime,commit,RUN_ROOT,variant))
                cards[variant]=base.read_json(RUN_ROOT/variant/'run_card.json')
                for name in ('run_card.json','command.sh','artifact_hashes.sha256','script_hashes.sha256'):
                    protected[str(RUN_ROOT/variant/name)]=assets.sha256(RUN_ROOT/variant/name)
            validate_pair(cards,RUN_ROOT,commit)
            for name in ('artifact_hashes.sha256','script_hashes.sha256'):
                manifests=[base.paired.load_sha256_manifest(RUN_ROOT/v/name) for v in VARIANTS]
                if manifests[0]!=manifests[1]:
                    raise ValueError('Paired artifact/runtime manifests differ')
            jobs.write_json(RUN_ROOT/'paired_preflight.json',{'passed':True,'cards':cards})
            jobs.write_json(RUN_ROOT/'protected_inputs.json',protected)
            results={}
            def train(variant):
                return train_variant(RUN_ROOT,variant,runtime,commit,runner,protected)
            def evaluate(name):
                result=evaluate_model(RUN_ROOT,name,runtime,commit,runner,protected)
                results[name]=result
                jobs.write_json(RUN_ROOT/'evaluation_acceptance.json',{'passed':True,'complete':len(results)==EXPECTED_EVAL_COUNT,'models':results})
                jobs.write_json(RUN_ROOT/'per_benchmark_results.json',{
                    task:{n:r['per_task'][task] for n,r in results.items()} for task in shared.TASK_COUNTS})
                print(json.dumps({'evaluated':name,'per_task':result['per_task']}),flush=True)
                return result
            execute_ordered(train,evaluate)
            write_comparison(RUN_ROOT,results)
            shared.verify_hashes(protected)
            jobs.wait_for_idle()
            jobs.write_json(state,{'status':'complete','updated_at':jobs.now(),'variants':VARIANTS,
                'checkpoint_steps':sorted(STEPS),'evaluation_models':list(results),'protected_inputs_verified':True})
        except BaseException as error:
            previous=base.read_json(state) if state.exists() else {}
            jobs.write_json(state,{**previous,'status':'failed','error':str(error),'updated_at':jobs.now()})
            raise


if __name__=='__main__':
    main()

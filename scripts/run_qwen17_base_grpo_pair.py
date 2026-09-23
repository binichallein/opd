#!/usr/bin/env python3
"""User-authorized Base1.7B/GRPO4B comparison despite inconclusive screening."""

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qualify_qwen17_base_grpo as qualification
import run_qwen17_instruct_pair as previous
from opd_ext.math_protocol import QWEN_COMPLETION_PROTOCOL, completion_input_ids, completion_math_prompt

ROOT = previous.ROOT
RUN_ROOT = ROOT / 'runs/20260923v4_qwen17_base_grpo_blockfirst_seed21_ml2'
CACHE = Path('/dev/shm/q17g')
RAY_CACHES = {'block3_mean': CACHE/'rb', 'token_opd': CACHE/'rt'}
QUALIFICATION = ROOT / 'runs/20260923v3_qwen17_base_grpo_teacher_screen_ml2/qualification'
CAPABILITY_SHA = 'd8ebca3e3edea8f0af35c69c192dd20d5a5e414304099735276358fc35d9fab0'
RECIPE_REFERENCE = ROOT / 'deployments/86a4122df91c5d1b76091619ad13048335d3b949'
RECIPE_FILES = ('opd_ext/math_protocol.py', 'opd_ext/request_seeds.py',
                'scripts/launch_revisiting_block_opd_formal_train.sh', 'scripts/eval_qwen3_math_vllm.py')
AUTHORIZATION = dict(training_authorized=True, capability_passed=False,
    capability_status='inconclusive', capability_failures=['continuation_gain'],
    capability_summary_sha256=CAPABILITY_SHA,
    user_instruction='Start training despite this result, using the previous complete comparison workflow.',
    scope='Only this pair and pinned screening result; engineering gates still required')


def validate_screen(summary, digest):
    q = qualification.configured_qualifier()
    r = summary.get('reports', {}).get('g4_b17', {})
    checks = r.get('checks', {})
    if (digest != CAPABILITY_SHA or summary.get('cell_errors') != {}
            or summary.get('protocol') != q.protocol() or summary.get('training_started') is not False
            or r.get('status') != 'inconclusive' or r.get('passed') is not False
            or r.get('teacher') != 'g4' or r.get('student') != 'b17'
            or r.get('failures') != ['continuation_gain'] or not checks
            or {k for k,v in checks.items() if v is not True} != {'continuation_gain'}):
        raise ValueError('Only the exact pinned inconclusive screen has a user override')


def preflight(run, runtime):
    if os.environ.get('RAY_ADDRESS') or os.environ.get('RAY_TMPDIR'):
        raise ValueError('Unexpected inherited Ray state')
    q = qualification.configured_qualifier()
    summary = q.old.read_sealed(QUALIFICATION/'pair_summary.json')
    preparation = q.old.read_sealed(QUALIFICATION/'prepare_manifest.json')
    validate_screen(summary, q.old.sha256(QUALIFICATION/'pair_summary.json'))
    if (preparation['protocol'] != q.protocol()
            or summary['prepare_sha256'] != q.old.sha256(QUALIFICATION/'prepare_manifest.json')
            or preparation['selected_sha256'] != q.old.sha256(QUALIFICATION/'selected.json')
            or q.old.object_hash(q.old.read_json(QUALIFICATION/'selected.json')) != q.prior.SOURCE_SELECTION_SHA
            or set(summary['evidence']) != set(preparation['cells'])):
        raise ValueError('Screening protocol or evidence changed')
    protected = run.common_preflight(runtime)
    for key in ('b17', 'g4'):
        model = q.assets.ensure_asset(key)
        if model != preparation['models'][key]:
            raise ValueError('Training model differs from screened bytes or identity')
        protected.update(model['hashes'])
    for name in ('pair_summary.json', 'prepare_manifest.json', 'selected.json'):
        protected[str(QUALIFICATION/name)] = q.old.sha256(QUALIFICATION/name)
    for key, digest in summary['evidence'].items():
        folder = QUALIFICATION/'cells'/key
        completion = q.old.read_sealed(folder/'completion.json')
        if q.old.sha256(folder/'completion.json') != digest or not completion['complete'] or completion['cell'] != key:
            raise ValueError('Screening completion changed')
        protected[str(folder/'completion.json')] = digest
        protected.update({str(folder/name): h for name,h in completion['files'].items()})
    for name in RECIPE_FILES:
        if q.old.sha256(runtime/name) != q.old.sha256(RECIPE_REFERENCE/name):
            raise ValueError(f'Historical training/eval implementation changed: {name}')
    q.old.verify_hashes(protected)
    q.old.seal_json(RUN_ROOT/'training_authorization.json', AUTHORIZATION)
    protected[str(RUN_ROOT/'training_authorization.json')] = q.old.sha256(RUN_ROOT/'training_authorization.json')
    return protected


def input_contract(run):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl import DataProto
    from agent_system.environments.env_manager import MathEnvironmentManager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    tokenizer = AutoTokenizer.from_pretrained(run.STUDENT, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(run.TEACHER, local_files_only=True)
    if tokenizer.get_vocab() != teacher.get_vocab() or teacher.eos_token_id != 151643:
        raise ValueError('Teacher/student token space or Base EOS differs')
    config = OmegaConf.create({'data':{'opd_prompt_protocol':run.PROTOCOL,
        'apply_chat_template_kwargs':{'enable_thinking':False},'max_prompt_length':2048,
        'truncation':'middle','return_raw_chat':True}})
    manager = MathEnvironmentManager.__new__(MathEnvironmentManager)
    manager.config = config
    collector = TrajectoryCollector(config,tokenizer)
    selected = run.qualify.old.read_json(QUALIFICATION/'selected.json')
    counts, digest = {}, hashlib.sha256()
    groups = {'qualification': [(str(i),r['question']) for i,r in enumerate(selected)]}
    for task,n in run.shared.TASK_COUNTS.items():
        rows=run.shared.grading.load_jsonl(run.base.DATA/'eval_jsonl'/f'{task}.jsonl')
        if len(rows)!=n or len({str(r['id']) for r in rows})!=n:
            raise ValueError('Incomplete benchmark identities')
        groups[task]=[(str(r['id']),r['problem']) for r in rows]
    for task,rows in groups.items():
        manager.tasks=[question for _,question in rows]
        for (identity,question),observation in zip(rows,manager.build_text_obs(manager.tasks)):
            raw=np.empty(1,dtype=object); raw[0]=[{'role':'user','content':question}]
            gen=DataProto.from_single_dict({'input_ids':torch.zeros((1,1),dtype=torch.long),
                'raw_prompt':raw,'data_source':np.array(['dapo-math-17k'],dtype=object)})
            processed=collector.preprocess_single_sample(0,gen,{'text':[observation]})
            ids=completion_input_ids(tokenizer,question)
            if (processed['raw_prompt_ids']!=ids
                    or processed['input_ids'][processed['attention_mask'].bool()].tolist()!=ids
                    or ids!=completion_input_ids(teacher,question)):
                raise ValueError('Actual Base collector/eval/teacher input mismatch')
            digest.update(json.dumps([task,identity,ids]).encode())
        counts[task]=len(rows)
    return dict(passed=True,actual_collector_exercised=True,questions_checked=counts,
                protocol=run.PROTOCOL,enable_thinking=False,chat_template_used=False,
                eos_token_id=151643,stop_token_ids=[],prompt_ids_sha256=digest.hexdigest())


def validate_rollout(row, prompt_ids, identity, step):
    from run_qwen_completion_gate import validate_seed_settings
    validate_seed_settings(row,identity,step)
    ids,count=row['training_response_token_ids'],row['response_length']
    first_eos=ids.index(151643)+1 if 151643 in ids else len(ids)
    mask=[1]*first_eos+[0]*(len(ids)-first_eos)
    if (row['prompt_token_ids']!=prompt_ids or len(ids)!=16384
            or row['response_tensor_width']!=16384 or not 0<count<=16384
            or count!=len(row['response_token_ids']) or ids[:count]!=row['response_token_ids']
            or row['padding_length']!=16384-count or mask!=row['training_response_mask']
            or row['response_mask']!=mask[:count]
            or len(row['training_rollout_log_probs'])!=16384
            or row['rollout_log_probs']!=row['training_rollout_log_probs'][:count]
            or not all(math.isfinite(p) for p in row['training_rollout_log_probs'])):
        raise ValueError('Base rollout prompt, mask or logprob contract violated')
    if row['finish_reason']=='length':
        if count!=16384 or row['stop_reason'] is not None:
            raise ValueError('Invalid length termination')
    elif row['finish_reason']=='stop':
        if row['response_token_ids'][-1]!=151643 or row['stop_reason'] is not None:
            raise ValueError('Invalid Base native EOS termination')
    else:
        raise ValueError('Unexpected finish reason')


def validate_eval_row(row, tokenizer):
    if (row.get('prompt_protocol')!=QWEN_COMPLETION_PROTOCOL or row.get('enable_thinking') is not False
            or row.get('rendered_prompt')!=completion_math_prompt(row['problem'])
            or row.get('prompt_token_ids')!=completion_input_ids(tokenizer,row['problem'])
            or row.get('eos_token_id')!=151643 or row.get('sampling',{}).get('stop_token_ids')!=[]):
        raise ValueError('Base evaluation input/stop contract changed')


def train_with_ray_gate(run, original, root, variant, runtime, commit, runner, protected):
    output=root/f'{variant}_ray_gate.json'
    runner([run.base.PYTHON,runtime/'scripts/run_qwen17_base_grpo_pair.py',
            '--ray-gate-output',output,'--ray-gate-variant',variant],
           root/f'queue_jobs/{variant}_ray_warmup',job_env={'CUDA_VISIBLE_DEVICES':''},
           deadline_epoch=time.time()+360)
    if run.base.read_json(output).get('passed') is not True:
        raise ValueError('CPU Ray worker startup gate failed')
    return original(root,variant,runtime,commit,runner,protected)


def configured_runner():
    run=qualification.private_module('_qwen17_base_grpo_queue',previous.__file__)
    specs=qualification.specifications()
    run.RUN_ROOT,run.CACHE=RUN_ROOT,CACHE
    run.STUDENT=ROOT/'models'/specs['b17']['repo'].split('/')[1]
    run.TEACHER=ROOT/'models'/specs['g4']['repo'].split('/')[1]
    run.STUDENT_REVISION=specs['b17']['revision']
    # Teacher provenance is pinned by screening hashes, not a forged SOURCE_REVISION marker.
    run.TEACHER_REVISION=''
    run.QUALIFICATION,run.CAPABILITY_SHA=QUALIFICATION,CAPABILITY_SHA
    run.PROTOCOL=QWEN_COMPLETION_PROTOCOL
    run.EOS_TOKEN_ID=151643
    run.STUDENT_INITIALIZATION='original ModelScope Qwen3-1.7B-Base, independently per arm'
    run.BASELINE_ALIGNMENT='Matched original Base student; completion protocol; unchanged historical losses; explicit capability override'
    run.AUTHORIZATION=AUTHORIZATION
    run.PROJECT_NAME='opd_qwen17_base_grpo'
    run.EXPERIMENT_PREFIX='qwen17-base-grpo'
    run.DISPLAY_LABEL='Qwen1.7 Base GRPO4B'
    run.preflight=partial(preflight,run)
    run.input_contract=partial(input_contract,run)
    run.prompt_input_ids=completion_input_ids
    run.validate_rollout=validate_rollout
    run.validate_eval_row=validate_eval_row
    run.train_variant=partial(train_with_ray_gate,run,run.train_variant)
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray-gate-output',type=Path)
    parser.add_argument('--ray-gate-variant',choices=previous.VARIANTS)
    args=parser.parse_args()
    if args.ray_gate_output:
        import socket
        import recover_qwen17_instruct_token as recovery
        previous.validate_host(socket.gethostname())
        if args.ray_gate_variant is None or args.ray_gate_output!=RUN_ROOT/f'{args.ray_gate_variant}_ray_gate.json':
            raise ValueError('Only this attempt may receive Ray gate artifacts')
        recovery.CACHE=RAY_CACHES[args.ray_gate_variant]
        recovery.ray_gate(args.ray_gate_output)
    else:
        configured_runner().main()


if __name__=='__main__':
    main()

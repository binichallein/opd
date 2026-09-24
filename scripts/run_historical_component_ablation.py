#!/usr/bin/env python3
"""Historical Qwen Base B/C ablations, with intentional legacy prompt/seed behavior."""

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qualify_qwen17_base_grpo as modules
import run_qwen17_instruct_pair as previous
import run_historical17_reeval_llama as historical
from opd_ext.math_protocol import (
    QWEN_HISTORICAL_PROTOCOL, historical_math_prompt, render_qwen_historical,
    validate_qwen_historical_prompt, render_nonthinking,
)

ROOT = previous.ROOT
RUN_ROOT = ROOT / 'runs/20260924v1_historical17_components_seed21_ml2'
CACHE = Path('/dev/shm/ha')
MODES = {'adv3': 'adv_only', 'joint3': 'joint_tokenmean'}
AUTHORIZATION = {
    'user_instruction': 'Run B and C on the old 1.7B Base / 4B GRPO setup.',
    'scope': 'Only B and C: 200 steps each, independent original initialization, full milestone evaluation.',
    'historical_quirks_intentional': ['think-required ChatML training versus nonthinking chat evaluation',
                                    'constant request seed21', 'historical EOS-derived mask'],
    'not_recommended_default': True,
    'extra_observability': 'Every raw rollout; same interval5 diagnostics; extra Step150 full checkpoint.',
}


def training_env(run, runtime, commit, root, variant, probe_step=None):
    if variant not in MODES:
        raise ValueError('Only B/C variants authorized')
    env = run.base.launcher_env(runtime, commit, root, 'block3_mean', probe_step)
    env.update(VARIANT=variant, PROJECT_NAME='opd_historical_components',
        EXP_NAME=f'historical17-{variant}' + ('-probe' if probe_step else ''),
        STUDENT_MODEL=str(run.STUDENT), MATH_TEACHER=str(run.TEACHER),
        STUDENT_MODEL_REVISION='', TEACHER_MODEL_REVISION='',
        OPD_PROMPT_PROTOCOL=QWEN_HISTORICAL_PROTOCOL, OPD_REQUEST_SEED_RULE='legacy',
        DIAGNOSTIC_SAVE_STEPS={None:run.SAVE_STEPS,1:'1',2:'1,2'}[probe_step],
        OPD_DIAG_OUTPUT_DIR=str(root/variant/'diagnostics'),
        LOSSLESS_ROLLOUT_DIR=str(root/variant/'rollouts'),
        ROLLOUT_ATTEMPT_ID=f'probe{probe_step}' if probe_step else 'formal',
        LOCAL_CACHE_ROOT=str(CACHE/variant/'train'),
        RESUME_FROM_PATH=str(root/variant/'checkpoints/global_step_1') if probe_step==2 else '',
        BASELINE_ALIGNMENT='Historical July1.7B Base / GRPO4B recipe; B/C loss-only ablations; old seed and prompt preserved')
    return env


def expected_card(run, original, root, variant, commit, probe_step=None):
    card = original(root, variant, commit, probe_step)
    card.update(opd_block_size=3, opd_block_advantage_mode='mean', opd_block_ablation=MODES[variant],
                request_seed_rule='legacy', opd_diag_interval=1 if probe_step else 5,
                ray_num_cpus=64, val_n=1)
    return card


def validate_pair(run, cards, root, commit):
    if set(cards) != set(MODES):
        raise ValueError('Only the matched B/C pair is authorized')
    allowed={'variant','opd_block_ablation','experiment_name','diagnostic_output_dir','lossless_rollout_dir'}
    left,right=(cards[v] for v in MODES)
    if any(left.get(k)!=right.get(k) for k in (set(left)|set(right))-allowed):
        raise ValueError('Unapproved B/C configuration difference')
    for variant in MODES:
        run.validate_card(cards[variant],root,variant,commit)


def manifest_hashes(path):
    return {str(file):digest for digest,file in previous.base.paired.load_sha256_manifest(path)}


def preflight(run, runtime):
    protected={}
    old_cards={}
    for variant, folder in historical.HISTORICAL_RUNS.items():
        old_cards[variant]=run.base.read_json(folder/'run_card.json')
        if old_cards[variant]['source_commit']!=historical.HISTORICAL_COMMIT:
            raise ValueError('Historical training reference changed')
        manifest=manifest_hashes(folder/'artifact_hashes.sha256')
        protected.update(manifest)
        for name in ('run_card.json','artifact_hashes.sha256','command.sh','env.txt'):
            protected[str(folder/name)]=run.assets.sha256(folder/name)
    run.shared.verify_hashes(protected)
    expected=run.expected_card(RUN_ROOT,'adv3',runtime.name)
    common=('seed','data_seed','rollout_seed','environment_seed','n_gpus_per_node','ray_num_cpus',
        'student_model','teacher_model','train_data','val_data','train_batch_size','ppo_mini_batch_size',
        'rollout_group_size','max_prompt_length','max_response_length','learning_rate','total_training_steps',
        'save_freq','test_freq','val_n','rollout_gpu_memory_utilization','rollout_max_num_batched_tokens',
        'rollout_temperature','rollout_top_p','actor_ppo_micro_batch_size_per_gpu',
        'rollout_log_prob_micro_batch_size_per_gpu','ref_log_prob_micro_batch_size_per_gpu',
        'opd_diagnostics','opd_diag_interval','opd_diag_topk','opd_diag_position_stride','filter_overlong_prompts')
    for card in old_cards.values():
        for key in common:
            if card[key]!=expected[key]:
                raise ValueError(f'Historical recipe differs: {key}')
    if run.assets.sha256(run.base.DATA/'train.parquet')!=run.jobs.TRAIN_SHA:
        raise ValueError('Changed DAPO bytes')
    run.validate_benchmark_hashes({task:run.assets.sha256(run.base.DATA/'eval_jsonl'/f'{task}.jsonl')
                                 for task in run.shared.TASK_COUNTS})
    run.shared.grading.validate_grader_hash(run.base.GRADER,run.shared.grading.HISTORICAL_GRADER_SHA256)
    protected[str(run.base.GRADER)]=run.shared.grading.HISTORICAL_GRADER_SHA256
    versions=run.qualify.runtime_versions()
    run.jobs.write_json(RUN_ROOT/'environment.json',dict(packages=versions,python=sys.version))
    run.jobs.write_json(RUN_ROOT/'historical_alignment.json',dict(passed=True,matched_fields=common,
        historical_cards=old_cards,authorization=AUTHORIZATION,
        limitations=['Not bitwise replay of July runtime; all dependency versions retained.',
                     'A/D are historical controls, not newly trained in this queue.',
                     'Step150 has no historical A/D counterpart.']))
    return protected


def input_contract(run):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl import DataProto
    from agent_system.environments.env_manager import MathEnvironmentManager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    tokenizer=AutoTokenizer.from_pretrained(run.STUDENT,local_files_only=True)
    teacher=AutoTokenizer.from_pretrained(run.TEACHER,local_files_only=True)
    if tokenizer.get_vocab()!=teacher.get_vocab() or tokenizer.eos_token_id!=151643:
        raise ValueError('Historical token space or EOS changed')
    config=OmegaConf.create({'data':{'opd_prompt_protocol':QWEN_HISTORICAL_PROTOCOL,
        'apply_chat_template_kwargs':{},'max_prompt_length':2048,'truncation':'middle','return_raw_chat':True}})
    manager=MathEnvironmentManager.__new__(MathEnvironmentManager)
    manager.config=config
    collector=TrajectoryCollector(config,tokenizer)
    digest=hashlib.sha256()
    count=0
    for task in run.shared.TASK_COUNTS:
        rows=run.shared.grading.load_jsonl(run.base.DATA/'eval_jsonl'/f'{task}.jsonl')
        manager.tasks=[r['problem'] for r in rows]
        for row,observation in zip(rows,manager.build_text_obs(manager.tasks)):
            raw=np.empty(1,dtype=object); raw[0]=[{'role':'user','content':row['problem']}]
            gen=DataProto.from_single_dict({'input_ids':torch.zeros((1,1),dtype=torch.long),
                'raw_prompt':raw,'data_source':np.array(['dapo-math-17k'],dtype=object)})
            processed=collector.preprocess_single_sample(0,gen,{'text':[observation]})
            ids=processed['input_ids'][processed['attention_mask'].bool()].tolist()
            validate_qwen_historical_prompt(tokenizer,row['problem'],ids,max_prompt_length=2048)
            if ids!=processed['raw_prompt_ids']:
                raise ValueError('Historical generation and scoring prompts differ')
            digest.update(json.dumps([task,row['id'],ids]).encode()); count+=1
    return dict(passed=True,actual_collector_exercised=True,training_examples_checked=count,
        training_prompt_ids_sha256=digest.hexdigest(),training_enable_thinking=None,
        evaluation=historical.historical_prompt_contract(run.STUDENT),
        train_eval_prompt_equal=False, intentional_historical_mismatch=True)


def audit_rollouts(run, folder, steps, *, probe=False):
    from transformers import AutoTokenizer
    from diagnose_token_truncation import analyze_tokens
    tokenizer=AutoTokenizer.from_pretrained(run.STUDENT,local_files_only=True)
    evidence=[]
    for step in steps:
        paths=list((folder/'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(paths)!=1:
            raise ValueError('Missing or duplicate raw step')
        rows=run.read_archive(paths[0])
        source=run.read_archive(run.HISTORICAL_INPUTS/f'rollouts/formal/step_{step:06d}/raw.jsonl.gz')
        if len(rows)!=32 or len(source)!=32 or len({r['traj_uid'] for r in rows})!=32:
            raise ValueError('Incomplete batch')
        fingerprints=[]
        for row,reference in zip(rows,source):
            if row['source_extra_info']!=reference['source_extra_info']:
                raise ValueError('DAPO prompt sequence changed')
            validate_qwen_historical_prompt(tokenizer,row['source_extra_info']['question'],
                                           row['prompt_token_ids'],max_prompt_length=2048)
            validate_legacy_rollout(row,step)
            fingerprints.append([row[k] for k in ('source_extra_info','prompt_token_ids','sampling')])
        evidence.append(dict(step=step,count=32,sha256=run.assets.sha256(paths[0]),
            paired_input_sha256=run.qualify.old.object_hash(fingerprints),
            length_stops=sum(r['finish_reason']=='length' for r in rows),
            generated_think_tags=sum(r['generated_think_tags'] for r in rows),
            periodic_tails=sum(analyze_tokens(r['response_token_ids'],16384)['tail_period'] is not None for r in rows),
            unique_response_count=len({tuple(r['response_token_ids']) for r in rows})))
    return dict(passed=True,steps=evidence,total_rollouts=32*len(evidence))


def validate_legacy_rollout(row,step):
    sampling=dict(temperature=1.,top_p=.9,top_k=-1,max_tokens=16384,n=1,
                  ignore_eos=False,stop_token_ids=[],seed=21)
    ids=row['training_response_token_ids']; count=row['response_length']
    end=ids.index(151643)+1 if 151643 in ids else len(ids)
    mask=[1]*end+[0]*(len(ids)-end)
    if (row['protocol']!=QWEN_HISTORICAL_PROTOCOL or row['enable_thinking'] is not None
            or row['step']!=step or row['eos_token_id']!=151643
            or any(row['sampling'].get(k)!=v for k,v in sampling.items())
            or len(ids)!=16384 or row['response_tensor_width']!=16384
            or not 0<count<=16384 or row['padding_length']!=16384-count
            or len(row['response_token_ids'])!=count or ids[:count]!=row['response_token_ids']
            or row['training_response_mask']!=mask or row['response_mask']!=mask[:count]
            or len(row['training_rollout_log_probs'])!=16384
            or row['rollout_log_probs']!=row['training_rollout_log_probs'][:count]
            or not all(math.isfinite(p) for p in row['training_rollout_log_probs'])):
        raise ValueError('Historical rollout seed/prompt/mask contract violated')
    if row['finish_reason']=='length':
        if count!=16384 or row['stop_reason'] is not None:
            raise ValueError('Invalid capped termination')
    elif row['finish_reason']=='stop':
        if row['response_token_ids'][-1]!=151643 or row['stop_reason'] is not None:
            raise ValueError('Invalid historical EOS termination')
    else:
        raise ValueError('Unexpected termination')


def validate_eval_row(row,tokenizer):
    text=render_nonthinking(tokenizer,row['prompt'])
    if (row.get('enable_thinking') is not False or row.get('rendered_prompt')!=text
            or row.get('prompt_token_ids')!=tokenizer.encode(text)
            or row.get('eos_token_id')!=151643
            or row.get('sampling',{}).get('stop_token_ids')!=[151645,151643]):
        raise ValueError('Historical evaluation protocol changed')


def audit_command(run,runtime,folder,commit,probe_step=None):
    cmd=list(map(str,run.base.audit_command(runtime,folder,commit,probe_step=probe_step)))
    for flag,value in {'--expected-student-model-suffix':run.STUDENT.name,
                      '--expected-student-model-revision':'',
                      '--checkpoint-steps':{None:run.SAVE_STEPS,1:'1',2:'1,2'}[probe_step]}.items():
        cmd[cmd.index(flag)+1]=value
    return cmd


def execute_ordered(train,evaluate):
    results={}
    for variant in MODES:
        if train(variant).get('passed') is not True:
            raise ValueError('Training acceptance failed')
        for step in previous.STEPS:
            name=f'{variant}_step{step}'
            results[name]=evaluate(name)
            if results[name].get('passed') is not True:
                raise ValueError('Evaluation acceptance failed')
    return results


def write_comparison(run,root,results):
    if set(results)!={f'{v}_step{s}' for v in MODES for s in run.STEPS}:
        raise ValueError('Eight complete evaluations required')
    paired=run.validate_paired_rollouts(*(run.base.read_json(root/v/'rollout_acceptance.json') for v in MODES))
    run.jobs.write_json(root/'paired_rollout_acceptance.json',paired)
    report=dict(training_protocol=QWEN_HISTORICAL_PROTOCOL,evaluation_protocol='legacy',
                training_seed=21,primary_checkpoint=200,per_benchmark={})
    for task in run.shared.TASK_COUNTS:
        values={name:r['per_task'][task] for name,r in results.items()}
        for step in sorted(run.STEPS):
            values[f'C_minus_B_step{step}_pp']={key:100*(values[f'joint3_step{step}'][key]-values[f'adv3_step{step}'][key])
                                               for key in ('avg_at_8','pass_at_8')}
        report['per_benchmark'][task]=values
    run.jobs.write_json(root/'paired_comparison.json',report)


def train_with_gate(run,original,root,variant,runtime,commit,runner,protected):
    output=root/f'{variant}_ray_gate.json'
    runner([run.base.PYTHON,runtime/'scripts/run_historical_component_ablation.py',
            '--ray-gate-variant',variant],root/f'queue_jobs/{variant}_ray_gate',
           job_env={'CUDA_VISIBLE_DEVICES':''},deadline_epoch=time.time()+360)
    if run.base.read_json(output).get('passed') is not True:
        raise ValueError('CPU Ray startup gate failed')
    return original(root,variant,runtime,commit,runner,protected)


def configured_runner():
    run=modules.private_module('_historical_component_queue',previous.__file__)
    run.RUN_ROOT,run.CACHE=RUN_ROOT,CACHE
    run.STUDENT=historical.QWEN_INITIAL
    run.TEACHER=ROOT/'models/Qwen3-4B-Base-GRPO'
    run.STUDENT_REVISION=run.TEACHER_REVISION=''
    run.VARIANTS=tuple(MODES)
    run.PROTOCOL=QWEN_HISTORICAL_PROTOCOL
    run.EVALUATION_PROTOCOL='legacy'
    run.TRAIN_EVAL_PROMPT_MATCH=False
    run.EXPECTED_EVAL_COUNT=8
    run.ORDERING='B probe/train/eval200,150,100,50 -> C probe/train/eval200,150,100,50'
    run.SEED_RULE='legacy'
    run.EOS_TOKEN_ID=151643
    run.CAPABILITY_SHA=None
    run.STUDENT_INITIALIZATION='same original official Qwen3-1.7B-Base bytes as July A/D, independently per arm'
    run.AUTHORIZATION=AUTHORIZATION
    run.DISPLAY_LABEL='Historical1.7B B/C ablation'
    run.training_env=partial(training_env,run)
    run.expected_card=partial(expected_card,run,run.expected_card)
    run.validate_pair=partial(validate_pair,run)
    run.preflight=partial(preflight,run)
    run.input_contract=partial(input_contract,run)
    run.audit_rollouts=partial(audit_rollouts,run)
    run.audit_command=partial(audit_command,run)
    run.validate_eval_row=validate_eval_row
    run.execute_ordered=execute_ordered
    run.write_comparison=partial(write_comparison,run)
    run.train_variant=partial(train_with_gate,run,run.train_variant)
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray-gate-variant',choices=tuple(MODES))
    args=parser.parse_args()
    if args.ray_gate_variant:
        import socket
        import recover_qwen17_instruct_token as recovery
        previous.validate_host(socket.gethostname())
        recovery.CACHE=CACHE/args.ray_gate_variant/'warm'
        recovery.ray_gate(RUN_ROOT/f'{args.ray_gate_variant}_ray_gate.json')
    else:
        configured_runner().main()


if __name__=='__main__':
    main()

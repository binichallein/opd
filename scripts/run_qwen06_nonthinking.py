#!/usr/bin/env python3
"""Standalone authorized non-thinking Token run, with GPU and resume gates."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_qwen06_pair as base
import run_window_queue as jobs

ROOT=base.ROOT
RUN_ROOT=ROOT/'runs/20260913v4_qwen06_nonthinking_token_seed21_ml2'
CACHE=Path('/limx_embap/tos/q06/0913v4')
PROTOCOL='math_eval_nonthinking_v1'


def training_env(runtime, commit, root, probe_step=None):
    env=base.launcher_env(runtime,commit,root,'token_opd',probe_step)
    env.update({
        'PROJECT_NAME':'opd_qwen06_nonthinking',
        'EXP_NAME':'qwen06-nonthinking-token-probe' if probe_step else 'qwen06-nonthinking-token',
        'OPD_PROMPT_PROTOCOL':PROTOCOL, 'LOSSLESS_ROLLOUT_DIR':str(root/'token_opd/rollouts'),
        'ROLLOUT_ATTEMPT_ID':f'probe{probe_step}' if probe_step else 'formal',
        'LOCAL_CACHE_ROOT':str(CACHE/'train'),
        'BASELINE_ALIGNMENT':'New non-thinking math-eval prompt protocol; old Token is not a matched baseline',
    })
    return env


def audit_rollouts(run, steps):
    from transformers import AutoTokenizer
    from opd_ext.math_protocol import math_prompt, render_nonthinking
    tokenizer=AutoTokenizer.from_pretrained(str(base.assets.STUDENT),local_files_only=True)
    evidence=[]
    for step in steps:
        files=list((run/'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(files)!=1:
            raise ValueError(f'Expected exactly one archived attempt for step {step}, got {files}')
        path=files[0]
        expected=path.with_suffix(path.suffix+'.sha256').read_text().split()[0]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Raw rollout hash mismatch')
        with gzip.open(path,'rt') as f:
            rows=[json.loads(line) for line in f]
        if len(rows)!=32 or len({r['traj_uid'] for r in rows})!=32:
            raise ValueError('Missing or duplicated training trajectories')
        for row in rows:
            text=render_nonthinking(tokenizer,math_prompt(row['source_extra_info']['question']))
            ids=tokenizer.encode(text,add_special_tokens=False)
            if len(ids)>2048:
                ids=ids[:1024]+ids[-1024:]
            if ids!=row['prompt_token_ids'] or row['enable_thinking'] is not False:
                raise ValueError('Actual training prompt does not match evaluation protocol')
            if row['sampling']['stop_token_ids'] != [151643,151645]:
                raise ValueError('Train/eval stopping protocol differs')
            if row['finish_reason'] not in ('length','stop'):
                raise ValueError('Missing engine finish reason')
        evidence.append({'step':step,'n':len(rows),'generated_think_tags':sum(r['generated_think_tags'] for r in rows),
                         'length_stops':sum(r['finish_reason']=='length' for r in rows),'sha256':expected})
    return evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-rollouts',type=Path)
    parser.add_argument('--steps',default='1,2')
    args=parser.parse_args()
    if args.audit_rollouts:
        evidence=audit_rollouts(args.audit_rollouts,[int(s) for s in args.steps.split(',')])
        jobs.write_json(args.audit_rollouts/'rollout_acceptance.json',{'passed':True,'evidence':evidence})
        return
    runtime=Path(__file__).resolve().parents[1]
    commit=(runtime/'DEPLOYED_COMMIT').read_text().strip()
    if runtime != ROOT/'deployments'/commit:
        raise ValueError('An immutable ml2 runtime is required')
    RUN_ROOT.mkdir(exist_ok=True)
    state=RUN_ROOT/'queue_state.json'
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    with (RUN_ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN_ROOT/'queue_manifest.json').exists():
            raise ValueError('Existing attempt requires manual review; no automatic restart')
        jobs.write_json(RUN_ROOT/'queue_manifest.json',{'source_commit':commit,'protocol':PROTOCOL,
                        'variant':'token_opd','training_steps':200,'created_at':jobs.now(),
                        'eval_autostart':False,'old_run_untouched':str(base.RUN_ROOT)})
        (RUN_ROOT/'queue.pid').write_text(str(os.getpid())+'\n')
        env=dict(os.environ,PATH=f'{base.VENV}/bin:'+os.environ.get('PATH',''),
                 PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd',CUDA_VISIBLE_DEVICES='0,1,2,3',
                 PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
                 RAY_DEDUP_LOGS='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',ENGINE='vllm')
        for key, suffix in {'TMPDIR':'tmp','VLLM_CACHE_ROOT':'vllm','TORCHINDUCTOR_CACHE_DIR':'inductor',
                            'TRITON_CACHE_DIR':'triton','CUDA_CACHE_PATH':'cuda','OUTLINES_CACHE_DIR':'outlines'}.items():
            (CACHE/suffix).mkdir(parents=True,exist_ok=True)
            env[key]=str(CACHE/suffix)
        def runner(command,job,job_env=None,**kw):
            jobs.run_job(command,job,runtime,state,{**env,**(job_env or {})},**kw)
        try:
            runner(['sha256sum','-c','.expected.sha256'],RUN_ROOT/'queue_jobs/runtime_verification')
            jobs.write_json(RUN_ROOT/'protected_inputs.json',base.preflight_assets())
            runner([base.PYTHON,runtime/'scripts/verify_nonthinking_gpu.py','--model',base.assets.STUDENT,
                    '--cohort',ROOT/'analyses/20260913_qwen06_truncation_probe_v2/prompts.jsonl',
                    '--eval-data',base.DATA/'eval_jsonl','--output',RUN_ROOT/'gpu_gate'],
                   RUN_ROOT/'queue_jobs/gpu_gate',gpu=True)
            if not base.read_json(RUN_ROOT/'gpu_gate/summary.json')['passed']:
                raise RuntimeError('Non-thinking GPU gate failed')
            probe=RUN_ROOT/'probes/token_opd'
            for step in (1,2):
                runner(['bash',runtime/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                       RUN_ROOT/f'queue_jobs/prepare_probe{step}',
                       job_env=training_env(runtime,commit,RUN_ROOT/'probes',step))
                job=RUN_ROOT/f'queue_jobs/probe{step}'
                runner(['bash',probe/'command.sh'],job,gpu=True)
                shutil.copyfile(job/'logs/job.log',probe/'logs/nohup.log')
                runner(base.audit_command(runtime,probe,commit,probe_step=step),RUN_ROOT/f'queue_jobs/audit_probe{step}')
            base.audit_resume(probe)
            evidence=audit_rollouts(probe,[1,2])
            jobs.write_json(probe/'rollout_acceptance.json',{'passed':True,'evidence':evidence})
            run=RUN_ROOT/'token_opd'
            runner(['bash',runtime/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                   RUN_ROOT/'queue_jobs/prepare_formal',job_env=training_env(runtime,commit,RUN_ROOT))
            runner(['bash',run/'command.sh'],run,gpu=True,pid_name='train.pid',log_name='nohup.log')
            runner(base.audit_command(runtime,run,commit),RUN_ROOT/'queue_jobs/checkpoints')
            jobs.write_json(run/'rollout_acceptance.json',{'passed':True,'evidence':audit_rollouts(run,range(1,201))})
            runner([base.PLOT_PYTHON,runtime/'scripts/analyze_single_opd_diagnostics.py',
                    '--run-dir',run,'--output-dir',run/'figures','--label','Qwen06 non-thinking Token seed21'],
                   RUN_ROOT/'queue_jobs/figures')
            for name,sha in base.read_json(RUN_ROOT/'protected_inputs.json').items():
                if base.assets.sha256(Path(name))!=sha:
                    raise ValueError('Protected input changed')
            jobs.write_json(state,{'status':'complete','updated_at':jobs.now(),'training_steps':200,
                                  'full_eval_started':False,'protected_inputs_verified':True})
        except BaseException as error:
            previous=base.read_json(state) if state.exists() else {}
            jobs.write_json(state,{**previous,'status':'failed','error':str(error),'updated_at':jobs.now()})
            raise


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Recover the pre-update Ray stall without changing the Base/GRPO experiment."""

import argparse
from functools import partial
import os
from pathlib import Path
import shutil
import socket
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_qwen17_base_grpo_pair as profile

SOURCE = profile.RUN_ROOT
SOURCE_COMMIT = '26382fcff692667269119a95b52bebceb40d3ced'
RUN_ROOT = profile.ROOT/'runs/20260923v5_qwen17_base_grpo_blockfirst_seed21_ml2'
CACHE = Path('/dev/shm/q17h')
INCIDENT = 'supervision/ray_stall_20260923T1220'
RAY_ENV = {'RAY_prestart_worker_first_driver':'0', 'RAY_enable_worker_prestart':'0'}
RAY_CONFIG = {'prestart_worker_first_driver':False, 'enable_worker_prestart':False}


def validate_failure(source):
    read = profile.previous.base.read_json
    run = source/'block3_mean'
    state = read(source/'queue_state.json')
    if (state.get('status') != 'failed' or state.get('job') != str(run)
            or read(source/'queue_manifest.json').get('source_commit') != SOURCE_COMMIT
            or 'RegisterClient' not in (source/INCIDENT/'driver_stack.txt').read_text()
            or 'have not registered within the timeout' not in (source/INCIDENT/'ray_logs/raylet.err').read_text()):
        raise ValueError('Only the evidenced driver-registration stall may be recovered')
    if list(run.rglob('raw.jsonl.gz')) or list(run.rglob('global_step_*')):
        raise ValueError('Formal training already produced artifacts; initialization restart forbidden')
    for file in (source/'queue.pid', run/'train.pid'):
        if file.exists():
            proc = Path(f'/proc/{int(file.read_text())}/cmdline')
            if proc.exists() and proc.read_bytes():
                raise ValueError('Source job is still alive; duplicate queue forbidden')


def validate_resume_gate(gate):
    expected = [dict(step=s,rank=r,scheduler_step=s) for s in (1,2) for r in range(4)]
    if gate.get('passed') is not True or gate.get('states') != expected:
        raise ValueError('All saved/resumed rank states are required')


def validate_relocation(q, old, new, commit):
    q.validate_card(old, SOURCE, 'block3_mean', SOURCE_COMMIT)
    q.validate_card(new, RUN_ROOT, 'block3_mean', commit)
    allowed = {'source_commit','diagnostic_output_dir','lossless_rollout_dir'}
    if any(old.get(k) != new.get(k) for k in (set(old)|set(new))-allowed):
        raise ValueError('Recovery must not change experimental settings')


def verify_training_files(q, manifest, old_runtime, runtime):
    for digest, path in q.base.paired.load_sha256_manifest(manifest):
        target = runtime/path.relative_to(old_runtime) if path.is_relative_to(old_runtime) else path
        if q.assets.sha256(target) != digest:
            raise ValueError(f'Frozen training implementation changed: {target}')


def recovery_preflight(q, original, runtime):
    import torch
    validate_failure(SOURCE)
    q.shared.verify_hashes(q.base.read_json(SOURCE/'protected_inputs.json'))
    old_runtime = q.ROOT/'deployments'/SOURCE_COMMIT
    # Deployment IDs may differ, but every file used by the old launcher must be identical.
    for variant in q.VARIANTS:
        verify_training_files(q,SOURCE/variant/'script_hashes.sha256',old_runtime,runtime)
    protected = original(runtime)
    probe = SOURCE/'probes/block3_mean'
    q.validate_card(q.base.read_json(probe/'run_card.json'),SOURCE/'probes','block3_mean',SOURCE_COMMIT,2)
    gate = q.base.read_json(probe/'resume_gate.json')
    validate_resume_gate(gate)
    if q.base.read_json(probe/'acceptance.json').get('passed') is not True:
        raise ValueError('Original checkpoint acceptance failed')
    log = (probe/'logs/nohup.log').read_text()
    if f'Resuming from {probe}/checkpoints/global_step_1' not in log or 'No dataloader state found' in log:
        raise ValueError('Original resume log missing')
    for step in (1,2):
        cp = probe/f'checkpoints/global_step_{step}'
        data = torch.load(cp/'data.pt',map_location='cpu',weights_only=False)
        inspection = q.base.read_json(probe/f'optimizer_inspection_step{step}.json')
        if inspection.get('passed') is not True or [r['rank'] for r in inspection['ranks']] != list(range(4)):
            raise ValueError('Original optimizer inspection incomplete')
        for rank in range(4):
            extra = cp/'actor'/f'extra_state_world_size_4_rank_{rank}.pt'
            q.base.validate_resume_state(torch.load(extra,map_location='cpu',weights_only=False),data,step)
            for prefix in ('model','optim','extra_state'):
                file = cp/'actor'/f'{prefix}_world_size_4_rank_{rank}.pt'
                if not file.is_file() or file.stat().st_size == 0:
                    raise ValueError(f'Missing saved state: {file}')
            optim = cp/'actor'/f'optim_world_size_4_rank_{rank}.pt'
            if q.assets.sha256(optim) != inspection['ranks'][rank]['sha256']:
                raise ValueError('Saved optimizer changed')
    evidence = q.audit_rollouts(probe,(1,2),probe=True)
    for name in ('resume_gate.json','acceptance.json','run_card.json',
                 'optimizer_inspection_step1.json','optimizer_inspection_step2.json'):
        protected[str(probe/name)] = q.assets.sha256(probe/name)
    shutil.copytree(SOURCE/INCIDENT,RUN_ROOT/'failure_evidence')
    q.jobs.write_json(RUN_ROOT/'reused_resume_gate.json',dict(source=str(probe),acceptance=gate,
        rollout_acceptance=evidence,formal_initialization=str(q.STUDENT),
        note='Reuse engineering acceptance only, not probe model/optimizer weights'))
    q.jobs.write_json(RUN_ROOT/'recovery_preflight.json',dict(passed=True,source=str(SOURCE),
        source_commit=SOURCE_COMMIT,ray_environment=RAY_ENV,training_files_identical=True,
        formal_steps_before_recovery=0,old_attempt_retained=True))
    return protected


def ray_gate(q, output, variant):
    import ray
    import ray.dashboard.utils as dashboard
    from recover_historical17_llama import check_socket_budget
    if ray.__version__ != '2.55.1':
        raise ValueError('Revalidate startup on a different Ray version')
    start = time.monotonic()
    dashboard.get_all_modules(dashboard.DashboardAgentModule)
    tmp = CACHE/variant/'gate/tmp'
    check_socket_budget(tmp)
    tmp.mkdir(parents=True,exist_ok=False)
    try:
        ray.init(num_cpus=64,num_gpus=0,include_dashboard=False,_temp_dir=str(tmp/'ray'),
                 object_store_memory=100*1024*1024,_system_config=RAY_CONFIG)
        @ray.remote
        def check():
            return 'ray-worker-ok'
        results = ray.get([check.remote() for _ in range(4)],timeout=90)
        if results != ['ray-worker-ok']*4:
            raise ValueError('Ray worker execution failed')
        q.jobs.write_json(output,dict(passed=True,ray_version=ray.__version__,num_cpus=64,
            config=RAY_CONFIG,elapsed_seconds=time.monotonic()-start,results=results))
    finally:
        ray.shutdown()


def train_variant(q, original, root, variant, runtime, commit, runner, protected):
    gate = root/f'{variant}_ray_gate.json'
    runner([q.base.PYTHON,runtime/'scripts/recover_qwen17_base_grpo.py',
            '--ray-gate-output',gate,'--variant',variant],root/f'queue_jobs/{variant}_ray_gate',
           job_env={'CUDA_VISIBLE_DEVICES':''},deadline_epoch=time.time()+360)
    if q.base.read_json(gate).get('passed') is not True:
        raise ValueError('64-CPU Ray startup gate failed')
    if variant == 'token_opd':
        return original(root,variant,runtime,commit,runner,protected)
    import run_qwen8_teacher_pair as helpers
    validate_resume_gate(q.base.read_json(root/'reused_resume_gate.json')['acceptance'])
    run = root/variant
    validate_relocation(q,q.base.read_json(SOURCE/variant/'run_card.json'),
                        q.base.read_json(run/'run_card.json'),commit)
    q.shared.verify_hashes(protected)
    runner(['bash',run/'command.sh'],run,gpu=True,pid_name='train.pid',log_name='nohup.log')
    runner(q.audit_command(runtime,run,commit),root/f'queue_jobs/{variant}_checkpoint_audit')
    helpers.verify_formal_training_states(run)
    q.jobs.write_json(run/'rollout_acceptance.json',q.audit_rollouts(run,range(1,201)))
    runner([q.base.PLOT_PYTHON,runtime/'scripts/analyze_single_opd_diagnostics.py',
            '--run-dir',run,'--output-dir',run/'figures','--label',f'{q.DISPLAY_LABEL} {variant} seed21'],
           root/f'queue_jobs/{variant}_figures')
    q.shared.verify_hashes(protected)
    return dict(passed=True,variant=variant,steps=200,checkpoint_steps=sorted(q.STEPS))


def configured_runner():
    private = profile.qualification.private_module('_q17_grpo_recovery_profile',profile.__file__)
    private.RUN_ROOT,private.CACHE = RUN_ROOT,CACHE
    q = private.configured_runner()
    original_train = q.train_variant.args[1]
    q.preflight = partial(recovery_preflight,q,q.preflight)
    q.train_variant = partial(train_variant,q,original_train)
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray-gate-output',type=Path)
    parser.add_argument('--variant',choices=('block3_mean','token_opd'))
    args = parser.parse_args()
    q = configured_runner()
    q.validate_host(socket.gethostname())
    os.environ.update(RAY_ENV)
    if args.ray_gate_output:
        if args.variant is None or args.ray_gate_output != RUN_ROOT/f'{args.variant}_ray_gate.json':
            raise ValueError('Unexpected Ray gate destination')
        ray_gate(q,args.ray_gate_output,args.variant)
    else:
        q.main()


if __name__ == '__main__':
    main()

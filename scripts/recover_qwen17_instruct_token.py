#!/usr/bin/env python3
"""One authorized Token-only recovery; preserve the failed attempt and completed evals."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_qwen17_instruct_pair as q
from recover_historical17_llama import attempt_locks, check_socket_budget

base, jobs, shared = q.base, q.jobs, q.shared
SOURCE = q.RUN_ROOT
RUN_ROOT = q.ROOT / 'runs/20260922v1_qwen17_instruct_token_recovery_seed21_ml2'
TRAIN_COMMIT = 'be736b5fac4f26ff59f4e2c21abafc456c2503f6'
TRAIN_RUNTIME = q.ROOT / 'deployments' / TRAIN_COMMIT
CACHE = Path('/dev/shm/q17t')
FAILED_RAY = q.CACHE / 'token_opd/train/tmp/ray/session_2026-09-21_23-59-19_786144_2162511'
REUSED_NAMES = tuple(f'block3_mean_step{s}' for s in q.STEPS) + ('student_base',)


def formal_env():
    return {**q.training_env(TRAIN_RUNTIME, TRAIN_COMMIT, RUN_ROOT, 'token_opd'),
            'LOCAL_CACHE_ROOT': str(CACHE / 'train')}


def validate_relocated_card(old, new):
    q.validate_card(old, SOURCE, 'token_opd', TRAIN_COMMIT)
    q.validate_card(new, RUN_ROOT, 'token_opd', TRAIN_COMMIT)
    allowed = {'diagnostic_output_dir', 'lossless_rollout_dir'}
    if any(old.get(k) != new.get(k) for k in (set(old) | set(new)) - allowed):
        raise ValueError('Recovery may relocate outputs, not change training settings')


def validate_failure(source):
    run = source / 'token_opd'
    state = base.read_json(source / 'queue_state.json')
    if (state.get('status') != 'failed' or state.get('job') != str(run)
            or base.read_json(source / 'queue_manifest.json').get('source_commit') != TRAIN_COMMIT
            or (run / 'exit_code.txt').read_text().strip() != '1'
            or 'current node timed out during startup' not in (run / 'logs/nohup.log').read_text()):
        raise ValueError('Recovery requires the reviewed Ray startup failure')
    if list(run.rglob('global_step_*')) or list(run.rglob('raw.jsonl.gz')):
        raise ValueError('Unexpected training artifacts; do not restart from initialization')


def execute_remaining(reused, train, evaluate):
    if set(reused) != set(REUSED_NAMES) or any(r.get('passed') is not True for r in reused.values()):
        raise ValueError('All five original evaluations must be accepted')
    if train().get('passed') is not True:
        raise ValueError('Token training acceptance failed')
    results = dict(reused)
    for step in q.STEPS:
        name = f'token_opd_step{step}'
        result = evaluate(name)
        if result.get('passed') is not True:
            raise ValueError(f'Token evaluation failed: {name}')
        results[name] = result
    return results


def ray_gate(output):
    # Ray 2.55.1's port-file wait is a hardcoded 15 seconds, not agent_register_timeout_ms.
    # Warm shared-filesystem imports, then check an actual fresh Ray worker. No library patch.
    import ray
    import ray.dashboard.utils as dashboard
    if ray.__version__ != '2.55.1':
        raise ValueError('Revalidate Ray startup on a different installed version')
    started = time.monotonic()
    modules = dashboard.get_all_modules(dashboard.DashboardAgentModule)
    warmed = time.monotonic()
    tmp = CACHE / 'gate/tmp'
    check_socket_budget(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        ray.init(num_cpus=1, num_gpus=0, include_dashboard=False,
                 _temp_dir=str(tmp / 'ray'), object_store_memory=100 * 1024 * 1024)
        @ray.remote
        def check():
            return 'ray-worker-ok'
        if ray.get(check.remote(), timeout=60) != 'ray-worker-ok':
            raise ValueError('CPU Ray worker failed')
        jobs.write_json(output, {'passed': True, 'ray_version': ray.__version__,
            'modules': [m.__name__ for m in modules], 'import_seconds': warmed-started,
            'startup_and_worker_seconds': time.monotonic()-warmed,
            'limitation': 'Import warmup reduces cold-start risk; it does not change the 15-second timeout'})
    finally:
        ray.shutdown()


def reuse_preflight():
    from opd_ext.eval_rollout_archive import audit_archive
    validate_failure(SOURCE)
    for file in (SOURCE/'queue.pid', SOURCE/'token_opd/train.pid'):
        proc = Path(f'/proc/{int(file.read_text())}/cmdline')
        if proc.exists() and proc.read_bytes():
            raise ValueError('Original process still exists; refuse duplicate run')
    protected = base.read_json(SOURCE/'protected_inputs.json')
    shared.verify_hashes(protected)
    q.qualify.runtime_versions()
    accepted = base.read_json(SOURCE/'evaluation_acceptance.json')
    if accepted.get('passed') is not True or set(accepted.get('models',{})) != set(REUSED_NAMES):
        raise ValueError('Original five full evaluations are not accepted')
    for name in REUSED_NAMES:
        folder = SOURCE/'evaluations'/name
        result = base.read_json(folder/'acceptance.json')
        _, model = q.model_paths(SOURCE, name)
        if (result != accepted['models'][name] or result.get('passed') is not True
                or result.get('model') != str(model) or (folder/'exit_code.txt').read_text().strip() != '0'
                or result.get('train_eval_prompt_verified') is not True):
            raise ValueError(f'Original evaluation incomplete or changed: {name}')
        shared.verify_hashes(result['sha256'])
        fresh = shared.audit_evaluation(folder/'outputs', base.DATA/'eval_jsonl', model)
        for task, metrics in fresh['per_task'].items():
            if any(result['per_task'][task].get(key) != value for key,value in metrics.items()):
                raise ValueError('Original metrics differ from archived predictions')
        if result['rollout_archive'] != audit_archive(folder/'outputs',base.read_json(folder/'outputs/eval_config.json')):
            raise ValueError('Original raw evaluation archive changed')
        protected.update(result['sha256'])
        protected[str(folder/'acceptance.json')] = q.assets.sha256(folder/'acceptance.json')
    probe = SOURCE/'probes/token_opd'
    q.validate_card(base.read_json(probe/'run_card.json'),SOURCE/'probes','token_opd',TRAIN_COMMIT,2)
    gate = base.read_json(probe/'resume_gate.json')
    expected = [{'step':s,'rank':r,'scheduler_step':s} for s in (1,2) for r in range(4)]
    if (gate.get('passed') is not True or gate.get('states') != expected
            or base.read_json(probe/'acceptance.json').get('passed') is not True):
        raise ValueError('Original independent Token GPU save/resume gate not accepted')
    for step in (1,2):
        inspection = base.read_json(probe/f'optimizer_inspection_step{step}.json')
        if inspection.get('passed') is not True or [r['rank'] for r in inspection['ranks']] != list(range(4)):
            raise ValueError('Original optimizer inspection failed')
        hashes = {str(probe/f'checkpoints/global_step_{step}/actor/optim_world_size_4_rank_{r["rank"]}.pt'):
                  r['sha256'] for r in inspection['ranks']}
        shared.verify_hashes(hashes)
    q.audit_rollouts(probe, (1,2), probe=True)
    for path in (SOURCE/'queue_state.json', SOURCE/'evaluation_acceptance.json',
                 SOURCE/'block3_mean/rollout_acceptance.json',probe/'resume_gate.json',probe/'run_card.json',
                 probe/'acceptance.json', SOURCE/'token_opd/logs/nohup.log'):
        protected[str(path)] = q.assets.sha256(path)
    jobs.write_json(RUN_ROOT/'reused_resume_gate.json',{'source':str(probe),'acceptance':gate,
        'reason':'Identical frozen runtime, models, training config; only CPU startup warmup/output paths differ'})
    return protected, accepted['models']


def train(runner, protected):
    import run_qwen8_teacher_pair as checkpoint_helpers
    run = RUN_ROOT/'token_opd'
    runner(['bash',run/'command.sh'],run,gpu=True,pid_name='train.pid',log_name='nohup.log')
    runner(q.audit_command(TRAIN_RUNTIME,run,TRAIN_COMMIT),RUN_ROOT/'queue_jobs/token_checkpoint_audit')
    checkpoint_helpers.verify_formal_training_states(run)
    evidence = q.audit_rollouts(run,range(1,201))
    jobs.write_json(run/'rollout_acceptance.json',evidence)
    jobs.write_json(RUN_ROOT/'paired_rollout_acceptance.json',q.validate_paired_rollouts(
        base.read_json(SOURCE/'block3_mean/rollout_acceptance.json'),evidence))
    runner([base.PLOT_PYTHON,TRAIN_RUNTIME/'scripts/analyze_single_opd_diagnostics.py',
            '--run-dir',run,'--output-dir',run/'figures','--label','Qwen1.7 Instruct Token seed21'],
           RUN_ROOT/'queue_jobs/token_figures')
    shared.verify_hashes(protected)
    return {'passed':True,'steps':200,'checkpoint_steps':sorted(q.STEPS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ray-gate-output',type=Path)
    args = parser.parse_args()
    q.validate_host(socket.gethostname())
    if args.ray_gate_output:
        ray_gate(args.ray_gate_output)
        return
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime/'DEPLOYED_COMMIT').read_text().strip()
    if runtime != q.ROOT/'deployments'/commit or (TRAIN_RUNTIME/'DEPLOYED_COMMIT').read_text().strip() != TRAIN_COMMIT:
        raise ValueError('New immutable controller and unchanged frozen training runtime required')
    sys.path.insert(0,str(TRAIN_RUNTIME/'external/revisiting_opd'))
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT/'queue_state.json'
    with attempt_locks(SOURCE,RUN_ROOT):
        if (RUN_ROOT/'queue_manifest.json').exists():
            raise FileExistsError('Existing recovery attempt; no automatic retry or overwrite')
        jobs.write_json(RUN_ROOT/'queue_manifest.json',{'controller_commit':commit,'training_commit':TRAIN_COMMIT,
            'source':str(SOURCE),'created_at':jobs.now(),'variant':'token_opd','seed':21,
            'steps':200,'checkpoint_steps':sorted(q.STEPS),'eval_steps':q.STEPS,
            'eval_tasks':shared.TASK_COUNTS,'reused_evaluations':REUSED_NAMES,'initialization':str(q.STUDENT),
            'retain_all_checkpoints':True,'retain_all_rollouts':True,'full_eval_autostart':True,
            'engineering_change':'Fresh short tmpfs cache plus CPU Ray import warmup and worker gate; no timeout patch'})
        (RUN_ROOT/'queue.pid').write_text(str(os.getpid())+'\n')
        try:
            jobs.wait_for_idle()
            if shutil.disk_usage(q.ROOT).free < 1_000_000_000_000:
                raise ValueError('Less than 1TB free; no checkpoint pruning authorized')
            CACHE.mkdir(exist_ok=False)
            mount = subprocess.run(['findmnt','-T',str(CACHE),'-n','-o','FSTYPE,OPTIONS'],
                check=True,capture_output=True,text=True).stdout.strip().split(maxsplit=1)
            q.validate_cache_mount(*mount,shutil.disk_usage(CACHE).free)
            check_socket_budget(CACHE/'train/tmp')
            # Durably retain the original agent/raylet failure before starting another Ray instance.
            if 'Timed out waiting for file' not in (FAILED_RAY/'logs/raylet.err').read_text():
                raise ValueError('Missing reviewed port-file failure evidence')
            shutil.copytree(FAILED_RAY/'logs',RUN_ROOT/'failure_evidence/ray_logs')
            jobs.write_json(RUN_ROOT/'failure_evidence/source.json',{'session':str(FAILED_RAY)})
            env = dict(os.environ,PATH=str(base.PYTHON.parent)+':'+os.environ.get('PATH',''),
                PYTHONPATH=f'{TRAIN_RUNTIME}:{TRAIN_RUNTIME}/external/revisiting_opd',CUDA_VISIBLE_DEVICES='0,1,2,3',
                PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
                RAY_DEDUP_LOGS='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',ENGINE='vllm',
                VLLM_WORKER_MULTIPROC_METHOD='spawn',EVAL_GRADE_UTILS_PATH=str(base.GRADER))
            if env.get('RAY_TMPDIR') or env.get('RAY_ADDRESS'):
                raise ValueError('Unexpected inherited Ray address/tmp override')
            for key,suffix in {'TMPDIR':'tmp','VLLM_CACHE_ROOT':'vllm','TRITON_CACHE_DIR':'triton',
                              'TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda',
                              'OUTLINES_CACHE_DIR':'outlines'}.items():
                (CACHE/suffix).mkdir()
                env[key] = str(CACHE/suffix)
            def runner(command,job,job_env=None,**kwargs):
                try:
                    jobs.run_job(command,job,TRAIN_RUNTIME,state,{**env,**(job_env or {})},**kwargs)
                except BaseException:
                    pidfile = job/kwargs.get('pid_name','job.pid')
                    if pidfile.exists():
                        q.cleanup_failed_group(int(pidfile.read_text()))
                    raise
            for label,release in (('controller',runtime),('training',TRAIN_RUNTIME)):
                jobs.run_job(['sha256sum','-c','.expected.sha256'],RUN_ROOT/f'queue_jobs/{label}_hashes',release,state,env)
            jobs.write_json(state,{'status':'preflight','updated_at':jobs.now()})
            protected, reused = reuse_preflight()
            runner(['bash',TRAIN_RUNTIME/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                   RUN_ROOT/'queue_jobs/prepare_token_formal',job_env=formal_env())
            run = RUN_ROOT/'token_opd'
            validate_relocated_card(base.read_json(SOURCE/'token_opd/run_card.json'),base.read_json(run/'run_card.json'))
            for name in ('artifact_hashes.sha256','script_hashes.sha256'):
                if base.paired.load_sha256_manifest(run/name) != base.paired.load_sha256_manifest(SOURCE/'token_opd'/name):
                    raise ValueError('Recovery changed original training artifacts/runtime')
            for name in ('run_card.json','command.sh','artifact_hashes.sha256','script_hashes.sha256'):
                protected[str(run/name)] = q.assets.sha256(run/name)
            jobs.write_json(RUN_ROOT/'protected_inputs.json',protected)
            jobs.write_json(RUN_ROOT/'recovery_preflight.json',{'passed':True,'only_card_differences':
                ['diagnostic_output_dir','lossless_rollout_dir'],'same_training_runtime':True})
            results = dict(reused)
            def report():
                jobs.write_json(RUN_ROOT/'evaluation_acceptance.json',{'passed':True,'complete':len(results)==9,'models':results})
                jobs.write_json(RUN_ROOT/'per_benchmark_results.json',{
                    task:{n:r['per_task'][task] for n,r in results.items()} for task in shared.TASK_COUNTS})
            report()
            def evaluate(name):
                result = q.evaluate_model(RUN_ROOT,name,TRAIN_RUNTIME,TRAIN_COMMIT,runner,protected)
                results[name] = result
                report()
                return result
            runner([base.PYTHON,runtime/'scripts/recover_qwen17_instruct_token.py','--ray-gate-output',RUN_ROOT/'ray_gate.json'],
                   RUN_ROOT/'queue_jobs/ray_warmup_gate',job_env={'CUDA_VISIBLE_DEVICES':''},deadline_epoch=time.time()+360)
            if base.read_json(RUN_ROOT/'ray_gate.json').get('passed') is not True:
                raise ValueError('Ray worker gate failed')
            results = execute_remaining(reused,lambda:train(runner,protected),evaluate)
            q.write_comparison(RUN_ROOT,results)
            shared.verify_hashes(protected)
            jobs.wait_for_idle()
            jobs.write_json(state,{'status':'complete','updated_at':jobs.now(),'evaluation_models':list(results),
                                  'protected_inputs_verified':True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state,{**previous,'status':'failed','error':str(error),'updated_at':jobs.now()})
            raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Reviewed recovery of the pre-training Ray socket-path failure on ml2."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_historical17_reeval_llama as historical

llama = historical.llama
base, jobs, shared = historical.base, historical.jobs, historical.shared
OLD_COMMIT = '94be7ea1d659309256c8356681925bb9710895c4'
SOURCE = historical.RUN_ROOT
RUN_ROOT = llama.ROOT / 'runs/20260918v3_llama32_historical17_recovery_seed21_ml2'
TRAIN_RUNTIME = llama.ROOT / 'deployments' / OLD_COMMIT
CACHE = Path('/limx_embap/tos/lh/r1')


def training_tmp():
    return CACHE / 'train/tmp'


def check_socket_budget(tmp):
    # Reserve space for Ray's timestamp, maximum Linux PID, and longest socket name.
    socket = tmp / 'ray/session_2026-09-18_15-04-52_411645_2147483647/sockets/plasma_store'
    length = len(str(socket).encode())
    if length > 107:
        raise ValueError(f'Ray socket exceeds 107 bytes: {length}: {socket}')
    return length


def ray_gate():
    import ray
    tmp = training_tmp()
    length = check_socket_budget(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(tmp)
    try:
        ray.init(num_cpus=1, num_gpus=0, include_dashboard=False,
                 _temp_dir=str(tmp / 'ray'), object_store_memory=100 * 1024 * 1024)
        @ray.remote
        def check():
            return 'ray-worker-ok'
        if ray.get(check.remote(), timeout=60) != 'ray-worker-ok':
            raise ValueError('Ray worker gate failed')
        print(json.dumps({'passed': True, 'tmpdir': str(tmp), 'socket_budget_bytes': length}), flush=True)
    finally:
        ray.shutdown()


def validate_failure(source):
    failed = source / 'llama32/queue_jobs/token_opd_probe1'
    state = base.read_json(source / 'queue_state.json')
    if (state.get('status') != 'failed' or state.get('job') != str(failed)
            or base.read_json(source / 'queue_manifest.json').get('source_commit') != OLD_COMMIT
            or (failed / 'exit_code.txt').read_text().strip() != '1'
            or 'AF_UNIX path length cannot exceed 107 bytes' not in (failed / 'logs/job.log').read_text()):
        raise ValueError('Recovery requires the reviewed pre-training socket failure')
    for parent in ('probes', *llama.VARIANTS):
        folder = source / 'llama32' / parent
        if list(folder.rglob('global_step_*')) or list(folder.rglob('raw.jsonl.gz')):
            raise ValueError('Unexpected training artifacts; do not silently restart trained weights')


def reuse_evaluation(folder, model):
    from opd_ext.eval_rollout_archive import audit_archive
    result = base.read_json(folder / 'acceptance.json')
    llama.predecessor.validate_result(result)
    if (folder / 'exit_code.txt').read_text().strip() != '0' or result.get('model') != str(model):
        raise ValueError('Original evaluation did not finish for the expected model')
    shared.verify_hashes(result['sha256'])
    fresh = shared.audit_evaluation(folder / 'outputs', base.DATA / 'eval_jsonl', model)
    for task, metrics in fresh['per_task'].items():
        if any(result['per_task'][task].get(key) != value for key, value in metrics.items()):
            raise ValueError('Original evaluation metrics differ from raw predictions')
    archive = audit_archive(folder / 'outputs', base.read_json(folder / 'outputs/eval_config.json'))
    if result.get('rollout_archive') != archive:
        raise ValueError('Original rollout archive differs from acceptance')
    return result


def preflight():
    validate_failure(SOURCE)
    check_socket_budget(training_tmp())
    for pid_file in (SOURCE / 'queue.pid', SOURCE / 'llama32/queue_jobs/token_opd_probe1/job.pid'):
        pid = int(pid_file.read_text())
        proc = Path(f'/proc/{pid}/cmdline')
        if proc.exists() and proc.read_bytes():
            raise ValueError(f'Original process still exists: {pid}')
    protected = base.read_json(SOURCE / 'protected_inputs.json')
    shared.verify_hashes(protected)
    accepted = base.read_json(SOURCE / 'qwen17/acceptance.json')
    if accepted.get('passed') is not True or set(accepted.get('models', {})) != set(historical.historical_models()):
        raise ValueError('Six historical evaluations must be complete before recovery')
    for name, model in historical.historical_models().items():
        result = reuse_evaluation(SOURCE / 'qwen17/evaluations' / name, model)
        if result != accepted['models'][name]:
            raise ValueError('Historical phase acceptance differs')
    for role in llama.assets.SPECS:
        gate = base.read_json(SOURCE / f'llama32/gpu_gate_{role}/summary.json')
        if gate.get('passed') is not True or gate.get('protocol') != historical.LEGACY_PROTOCOL:
            raise ValueError('Missing original GPU prompt gate')
    initial = reuse_evaluation(SOURCE / 'llama32/evaluations/student_base', llama.assets.STUDENT)
    # Keep the failed attempt, all completed evaluations, and their provenance read-only.
    for path in SOURCE.rglob('*'):
        if path.is_file():
            protected[str(path)] = llama.assets.sha256(path)
    return protected, initial


def execute_remaining(initial, evaluate, train):
    return llama.execute_ordered(lambda name: initial if name == 'student_base' else evaluate(name), train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ray-gate', action='store_true')
    args = parser.parse_args()
    if args.ray_gate:
        ray_gate()
        return
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime / 'DEPLOYED_COMMIT').read_text().strip()
    if runtime != llama.ROOT / 'deployments' / commit:
        raise ValueError('Immutable recovery runtime required')
    if (TRAIN_RUNTIME / 'DEPLOYED_COMMIT').read_text().strip() != OLD_COMMIT:
        raise ValueError('Training and evaluation runtime must remain unchanged')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (SOURCE / 'queue.lock').open('r') as old_lock, (RUN_ROOT / 'queue.lock').open('a') as lock:
        for handle in (old_lock, lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists():
            raise FileExistsError('Existing recovery attempt; no automatic retry')
        jobs.write_json(RUN_ROOT / 'queue_manifest.json', {'controller_commit': commit,
            'training_commit': OLD_COMMIT, 'source': str(SOURCE), 'created_at': jobs.now(),
            'reason': 'reviewed Ray AF_UNIX path length failure before first training step',
            'training_cache': str(CACHE), 'training_protocol': historical.LEGACY_PROTOCOL,
            'eval_protocol': llama.LLAMA_PROTOCOL, 'seed': 21, 'retain_rollouts': True,
            'reused_evaluation': str(SOURCE / 'llama32/evaluations/student_base'),
            'initialization': str(llama.assets.STUDENT), 'resume_from_failed_attempt': False})
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
            PYTHONPATH=f'{TRAIN_RUNTIME}:{TRAIN_RUNTIME}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
            RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
            VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(base.GRADER))
        if env.get('RAY_TMPDIR'):
            raise ValueError('Unexpected RAY_TMPDIR override; review rather than silently change paths')
        for key, suffix in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                            'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'OUTLINES_CACHE_DIR': 'outlines'}.items():
            (CACHE / suffix).mkdir(parents=True, exist_ok=True)
            env[key] = str(CACHE / suffix)
        def runner(command, job, job_env=None, **kwargs):
            jobs.run_job(command, job, TRAIN_RUNTIME, state, {**env, **(job_env or {})}, **kwargs)
        try:
            jobs.wait_for_idle()
            for label, release in (('controller', runtime), ('training', TRAIN_RUNTIME)):
                jobs.run_job(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / f'queue_jobs/{label}_hashes',
                             release, state, env)
            runner([base.PYTHON, runtime / 'scripts/recover_historical17_llama.py', '--ray-gate'],
                   RUN_ROOT / 'queue_jobs/ray_socket_gate')
            jobs.write_json(state, {'status': 'preflight', 'updated_at': jobs.now()})
            protected, initial = preflight()
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            jobs.write_json(RUN_ROOT / 'reused_initial_evaluation.json', initial)
            options = {'run_root': RUN_ROOT, 'training_protocol': historical.LEGACY_PROTOCOL, 'cache': CACHE}
            llama.prepare_pair(TRAIN_RUNTIME, OLD_COMMIT, runner, **options)
            def evaluate(name):
                result = llama.evaluate_model(name, TRAIN_RUNTIME, OLD_COMMIT, runner,
                                              run_root=RUN_ROOT, retain_rollouts=True)
                return historical.accept_archive(RUN_ROOT / 'evaluations' / name, result)
            results = execute_remaining(initial, evaluate, lambda variant: llama.train_model(
                variant, TRAIN_RUNTIME, OLD_COMMIT, runner, protected, **options))
            llama.compare_prompt_order(RUN_ROOT)
            for result in results.values():
                shared.verify_hashes(result['sha256'])
            shared.verify_hashes(protected)
            llama.write_comparison(RUN_ROOT, results, training_protocol=historical.LEGACY_PROTOCOL)
            jobs.write_json(RUN_ROOT / 'pair_acceptance.json', {'passed': True, 'models': results,
                'training_protocol': historical.LEGACY_PROTOCOL, 'eval_protocol': llama.LLAMA_PROTOCOL,
                'intentional_train_eval_instruction_difference': True})
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                'evaluation_models': list(results), 'protected_inputs_verified': True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'error': str(error), 'updated_at': jobs.now()})
            raise


if __name__ == '__main__':
    main()

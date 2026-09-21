#!/usr/bin/env python3
"""ml2-only, inference-only Base then non-thinking Qwen8-to-1.7 pair diagnostics."""

import fcntl
import os
from pathlib import Path
import shutil
import signal
import sys

ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
RUN_ROOT = ROOT / 'runs/20260921v1_qwen8_to17_diagnostics_ml2'
PYTHON = Path('/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl/bin/python')
CACHE = Path('/dev/shm/q817d')


def job_specs(control):
    assets = control / 'scripts/prepare_qwen17_pair_assets.py'
    qualification = control / 'scripts/qualify_qwen17_pairs.py'
    result = []
    for pair in ('base', 'instruct'):
        for label in ('student', 'teacher'):
            key = f'{pair}_{label}'
            result.append((f'asset_{key}', [PYTHON, assets, '--model', key, '--download'], False))
        common = ['--root', RUN_ROOT / 'qualification', '--pair', pair]
        result.append((f'prepare_{pair}', [PYTHON, qualification, 'prepare', *common], False))
        if pair == 'instruct':
            for label in ('student', 'teacher'):
                result.append((f'{pair}_smoke_{label}', [PYTHON, qualification, 'smoke', *common,
                                                       '--label', label, '--gpu', '0'], True))
        for phase in ('direct', 'continuation'):
            for label in ('student', 'teacher'):
                result.append((f'{pair}_{phase}_{label}', [PYTHON, qualification, 'generate', *common,
                                                         '--phase', phase, '--label', label, '--gpu', '0'], True))
        result.append((f'summarize_{pair}', [PYTHON, qualification, 'summarize', *common], False))
    return result


def queue_manifest(commit):
    return dict(control_commit=commit, pairs=['base', 'instruct'], training_authorized=False,
                full_benchmark_eval=False, expected_diagnostic_rollouts=768,
                expected_smoke_rollouts=8, retain_all_rollouts=True,
                independent_pair_decisions=True, capabilities_only=True)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_window_queue as jobs
    import qualify_qwen17_pairs as qualification
    from run_qwen4_completion_block3 import cleanup_failed_group, validate_cache_mount
    import subprocess

    control = Path(__file__).resolve().parents[1]
    commit = (control / 'DEPLOYED_COMMIT').read_text().strip()
    if control != ROOT / 'analysis_deployments' / commit:
        raise ValueError('Immutable ml2 deployment required')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt: no overwrite, relaunch, or automatic retry')
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        qualification.old.seal_json(RUN_ROOT / 'queue_manifest.json',
                                    {**queue_manifest(commit), 'created_at': jobs.now()})
        try:
            jobs.wait_for_idle()
            if shutil.disk_usage(ROOT).free < 100_000_000_000:
                raise ValueError('Less than 100GB free; no pruning authorized')
            CACHE.mkdir(exist_ok=True)
            mount = subprocess.run(['findmnt', '-T', str(CACHE), '-n', '-o', 'FSTYPE,OPTIONS'],
                                   check=True, capture_output=True, text=True).stdout.strip().split(maxsplit=1)
            validate_cache_mount(*mount, shutil.disk_usage(CACHE).free)
            env = dict(os.environ, PATH=str(PYTHON.parent) + ':' + os.environ.get('PATH', ''),
                       PYTHONPATH=f'{control}:{control}/scripts:{control}/external/revisiting_opd',
                       CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                       VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(qualification.old.GRADER))
            for key, suffix in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TRITON_CACHE_DIR': 'triton',
                                'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'CUDA_CACHE_PATH': 'cuda'}.items():
                (CACHE / suffix).mkdir(exist_ok=True)
                env[key] = str(CACHE / suffix)
            specs = [('control_hashes', ['sha256sum', '-c', '.expected.sha256'], False), *job_specs(control)]
            for name, argv, gpu in specs:
                job = RUN_ROOT / 'queue_jobs' / name
                try:
                    jobs.run_job(argv, job, control, state, env, gpu=gpu)
                except BaseException:
                    pidfile = job / 'job.pid'
                    if gpu and pidfile.exists():
                        cleanup_failed_group(int(pidfile.read_text()))
                    raise
            reports = {pair: qualification.old.read_sealed(RUN_ROOT / 'qualification' / pair / 'gate_acceptance.json')
                       for pair in ('base', 'instruct')}
            if any(r.get('training_authorized') is not False for r in reports.values()):
                raise ValueError('Diagnostics must never authorize training')
            qualification.old.seal_json(RUN_ROOT / 'pair_summary.json', dict(reports=reports, training_started=False,
                                                                           full_benchmark_eval=False, finished_at=jobs.now()))
            jobs.wait_for_idle()
            jobs.write_json(state, dict(status='complete', updated_at=jobs.now(), training_started=False,
                                       pair_statuses={p: r['status'] for p, r in reports.items()},
                                       report=str(RUN_ROOT / 'pair_summary.json')))
        except BaseException as error:
            previous = qualification.old.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'updated_at': jobs.now(), 'error': str(error),
                                   'training_started': False})
            raise


if __name__ == '__main__':
    main()

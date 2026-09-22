#!/usr/bin/env python3
"""Detached ml2-only inference queue; one worker per GPU, no automatic retries."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import qualify_qwen06_teachers as q
import run_window_queue as jobs
from run_qwen4_completion_block3 import cleanup_failed_group, validate_cache_mount

ROOT = q.old.BASE
RUN = ROOT / 'runs/20260923v1_qwen06_teacher_screen_ml2'
PYTHON = Path('/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/envs/verl/bin/python')
CACHE = Path('/dev/shm/q06screen0923')


def idle_gpus():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    rows = [[int(x.strip()) for x in line.split(',')] for line in result.stdout.splitlines()]
    if len(rows) != 4 or {r[0] for r in rows} != set(range(4)):
        raise ValueError('Expected ml2 four-GPU inventory')
    return {i for i, memory, utilization in rows if memory < 500 and utilization == 0}


def environment(control, name, gpu=None):
    cache = CACHE / name
    cache.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PATH=str(PYTHON.parent) + ':' + os.environ.get('PATH', ''),
               PYTHONPATH=f'{control}:{control}/scripts:{control}/external/revisiting_opd',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
               VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='4',
               EVAL_GRADE_UTILS_PATH=str(q.old.GRADER))
    for key, directory in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TRITON_CACHE_DIR': 'triton',
                           'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'CUDA_CACHE_PATH': 'cuda'}.items():
        (cache / directory).mkdir()
        env[key] = str(cache / directory)
    return env


def launch(control, name, argv, gpu=None):
    directory = RUN / 'jobs' / name
    directory.mkdir(parents=True, exist_ok=False)
    env = environment(control, name, gpu)
    q.old.write_json(directory / 'command.json', dict(argv=[str(x) for x in argv],
        cwd=str(control), gpu=gpu, environment={k: env[k] for k in (
            'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'TMPDIR', 'OMP_NUM_THREADS', 'VLLM_WORKER_MULTIPROC_METHOD')}))
    with (directory / 'output.log').open('x') as log:
        process = subprocess.Popen([str(x) for x in argv], cwd=control, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    q.old.write_json(directory / 'started.json', dict(pid=process.pid, gpu=gpu, time=jobs.now()))
    print(json.dumps(dict(event='started', job=name, pid=process.pid, gpu=gpu)), flush=True)
    return process


def complete(name, process, code):
    q.old.write_json(RUN / 'jobs' / name / 'finished.json', dict(pid=process.pid, exit_code=code, time=jobs.now()))
    print(json.dumps(dict(event='finished', job=name, exit_code=code)), flush=True)
    if code != 0:
        cleanup_failed_group(process.pid)


def main():
    control = Path(__file__).resolve().parents[1]
    commit = (control / 'DEPLOYED_COMMIT').read_text().strip()
    if control != ROOT / 'analysis_deployments' / commit:
        raise ValueError('Immutable ml2 deployment required')
    RUN.mkdir(exist_ok=True)
    active, statuses = {}, {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (RUN / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q.old.seal_json(RUN / 'queue_manifest.json', dict(commit=commit, protocol=q.protocol(),
            gpu_slots=[0, 1, 2, 3], pairs=q.PAIRS, training_started=False, created_at=jobs.now(),
            no_automatic_retry=True, source_note='Official ModelScope; existing public GRPO read-only'))
        q.old.write_json(RUN / 'queue_process.json', dict(pid=os.getpid(), ppid=os.getppid(),
                                                       sid=os.getsid(0), argv=sys.argv))
        try:
            subprocess.run(['sha256sum', '--quiet', '-c', '.expected.sha256'], cwd=control, check=True)
            if len(idle_gpus()) != 4 or shutil.disk_usage(ROOT).free < 100_000_000_000:
                raise ValueError('GPUs busy or insufficient disk; not disturbing other tasks')
            CACHE.mkdir(exist_ok=False)
            mount = subprocess.run(['findmnt', '-T', str(CACHE), '-n', '-o', 'FSTYPE,OPTIONS'],
                                   check=True, capture_output=True, text=True).stdout.strip().split(maxsplit=1)
            validate_cache_mount(*mount, shutil.disk_usage(CACHE).free)
            q.old.write_json(RUN / 'resource_inventory.json', dict(
                gpu=subprocess.check_output(['nvidia-smi', '-q'], text=True),
                cpu_count=os.cpu_count(), disk_free=shutil.disk_usage(ROOT).free,
                memory=Path('/proc/meminfo').read_text(), time=jobs.now()))
            cpu_jobs = [(f'asset_{k}', [PYTHON, control / 'scripts/prepare_qwen06_screen_assets.py',
                                      '--model', k, '--download']) for k in ('i06', 'i4')]
            cpu_jobs.append(('prepare', [PYTHON, control / 'scripts/qualify_qwen06_teachers.py',
                                         'prepare', '--root', RUN / 'qualification']))
            for name, argv in cpu_jobs:
                jobs.write_json(RUN / 'queue_state.json', dict(status=name, updated_at=jobs.now()))
                proc = launch(control, name, argv)
                active[name] = (None, proc)
                code = proc.wait()
                complete(name, proc, code)
                active.pop(name)
                if code != 0:
                    raise RuntimeError(f'{name} failed; no GPU work launched')
            m, _ = q.load_preparation(RUN / 'qualification')
            cells = m['cells']
            last_progress = time.monotonic()
            while len(statuses) < len(cells):
                for name, (gpu, proc) in list(active.items()):
                    code = proc.poll()
                    if code is not None:
                        complete(name, proc, code)
                        statuses[name] = 'complete' if code == 0 else 'failed'
                        active.pop(name)
                        last_progress = time.monotonic()
                for name in q.blocked_cells(cells, statuses):
                    statuses[name] = 'blocked'
                ready = q.ready_cells(cells, statuses, set(active))
                available = sorted(idle_gpus() - {gpu for gpu, _ in active.values()})
                for name, gpu in zip(ready, available):
                    proc = launch(control, name, [PYTHON, control / 'scripts/qualify_qwen06_teachers.py',
                        'generate', '--root', RUN / 'qualification', '--cell', name, '--gpu', str(gpu)], gpu)
                    active[name] = (gpu, proc)
                    last_progress = time.monotonic()
                jobs.write_json(RUN / 'queue_state.json', dict(status='running', updated_at=jobs.now(),
                    statuses=statuses, running={k: dict(gpu=g, pid=p.pid) for k, (g, p) in active.items()},
                    pending=[k for k in cells if k not in statuses and k not in active], training_started=False))
                if not active and len(statuses) < len(cells) and time.monotonic() - last_progress > 300:
                    raise RuntimeError('No schedulable free GPU or dependency deadlock; no unrelated processes killed')
                if len(statuses) < len(cells):
                    time.sleep(10)
            proc = launch(control, 'summarize', [PYTHON, control / 'scripts/qualify_qwen06_teachers.py',
                'summarize', '--root', RUN / 'qualification'])
            active['summarize'] = (None, proc)
            code = proc.wait()
            complete('summarize', proc, code)
            active.pop('summarize')
            if code != 0:
                raise RuntimeError('Summary verification failed')
            reports = q.old.read_sealed(RUN / 'qualification/pair_summary.json')
            jobs.write_json(RUN / 'queue_state.json', dict(
                status='complete' if all(v == 'complete' for v in statuses.values()) and not reports['cell_errors'] else 'incomplete',
                updated_at=jobs.now(), statuses=statuses, training_started=False,
                pair_statuses={k: r['status'] for k, r in reports['reports'].items()}))
        except BaseException as error:
            for _, proc in active.values():
                cleanup_failed_group(proc.pid)
                proc.wait(timeout=60)
            jobs.write_json(RUN / 'queue_state.json', dict(status='failed', error=str(error),
                updated_at=jobs.now(), statuses=statuses, training_started=False))
            raise


if __name__ == '__main__':
    main()

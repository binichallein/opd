#!/usr/bin/env python3
"""Launch only the approved 200-step Block3 run using the accepted frozen runtime."""

import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
TRAIN_COMMIT = '0f9161f02f08287fb07f0375ad0a6bda81133ff0'
RUNTIME = ROOT / 'deployments' / TRAIN_COMMIT
RUN_ROOT = ROOT / 'runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2'
RUN = RUN_ROOT / 'block3_mean'
CACHE = Path('/dev/shm/opd-q4-c3')
SAVE_STEPS = '50,100,150,200'

# On ml2 all helper imports come from the already GPU-accepted runtime.
sys.path[:0] = [str(RUNTIME / 'scripts'), str(RUNTIME), str(Path(__file__).resolve().parent)]
import run_qwen_completion_gate as gate

old, jobs, base = gate.old, gate.jobs, gate.base


def formal_card_overrides():
    return {'project_name': 'opd_qwen4_completion', 'experiment_name': 'qwen4-completion-block3-mean',
            'total_training_steps': 200, 'diagnostic_save_steps': SAVE_STEPS, 'stop_after_step': -1,
            'resume_mode': 'disable', 'resume_from_path': '', 'rollout_attempt_id': 'formal',
            'diagnostic_output_dir': str(RUN / 'diagnostics'), 'lossless_rollout_dir': str(RUN / 'rollouts'),
            'baseline_alignment': 'Accepted completion/independent-seed protocol; unchanged historical Block3 joint ratio and reduction'}


def training_env():
    env = gate.training_env(RUNTIME, TRAIN_COMMIT, RUN_ROOT, 'block3_mean', 1)
    card = formal_card_overrides()
    env.update(PROJECT_NAME=card['project_name'], EXP_NAME=card['experiment_name'],
               TOTAL_TRAINING_STEPS='200', DIAGNOSTIC_SAVE_STEPS=SAVE_STEPS, STOP_AFTER_STEP='-1',
               ROLLOUT_ATTEMPT_ID='formal', LOCAL_CACHE_ROOT=str(CACHE / 'train'),
               BASELINE_ALIGNMENT=card['baseline_alignment'])
    return env


def validate_card(card, accepted):
    expected = {**accepted, **formal_card_overrides()}
    if card != expected:
        changed = [k for k in set(card) | set(expected) if card.get(k) != expected.get(k)]
        raise ValueError(f'Unapproved difference from GPU-accepted run: {changed}')


def validate_gate(state, acceptance, commit):
    if (state.get('status') != 'complete' or acceptance.get('passed') is not True
            or set(acceptance.get('arms', {})) != {'block3_mean', 'token_opd'} or commit != TRAIN_COMMIT):
        raise ValueError('Both accepted GPU update/resume arms at the frozen training commit are required')


def checkpoint_audit_command():
    return old.audit_command(RUNTIME, RUN, TRAIN_COMMIT) + [
        '--expected-diag-interval', '1', '--expected-diagnostic-steps', ','.join(map(str, range(1, 201)))]


def validate_cache_mount(fstype, options, free_bytes):
    if fstype != 'tmpfs' or 'noexec' in options.split(',') or free_bytes < 20_000_000_000:
        raise ValueError('Executable local tmpfs with at least 20GB free is required for temporary caches')


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


def main():
    control_root = Path(__file__).resolve().parents[1]
    control_commit = (control_root / 'DEPLOYED_COMMIT').read_text().strip()
    if (control_root != ROOT / 'analysis_deployments' / control_commit
            or Path(gate.__file__).resolve() != RUNTIME / 'scripts/run_qwen_completion_gate.py'
            or (RUNTIME / 'DEPLOYED_COMMIT').read_text().strip() != TRAIN_COMMIT):
        raise ValueError('Immutable ml2 controller and accepted training runtime required')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists() or RUN.exists():
            raise FileExistsError('Never overwrite or automatically retry an existing formal run')
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
                   PYTHONPATH=f'{RUNTIME}:{RUNTIME}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
                   VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(base.GRADER))
        if env.get('RAY_TMPDIR'):
            raise ValueError('Unexpected RAY_TMPDIR')
        CACHE.mkdir(parents=True, exist_ok=True)
        mount = subprocess.run(['findmnt', '-T', str(CACHE), '-n', '-o', 'FSTYPE,OPTIONS'],
                               check=True, capture_output=True, text=True).stdout.strip().split(maxsplit=1)
        validate_cache_mount(*mount, shutil.disk_usage(CACHE).free)
        from recover_historical17_llama import check_socket_budget
        check_socket_budget(CACHE / 'train/tmp')
        for key, suffix in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                            'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'OUTLINES_CACHE_DIR': 'outlines'}.items():
            (CACHE / suffix).mkdir(parents=True, exist_ok=True)
            env[key] = str(CACHE / suffix)
        def runner(command, job, job_env=None, **kwargs):
            try:
                jobs.run_job(command, job, RUNTIME, state, {**env, **(job_env or {})}, **kwargs)
            except BaseException:
                # A failed driver can leave Ray workers in its owned process group.
                if job == RUN and (RUN / 'train.pid').exists():
                    cleanup_failed_group(int((RUN / 'train.pid').read_text()))
                raise
        try:
            jobs.wait_for_idle()
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/runtime_hashes')
            validate_gate(base.read_json(gate.RUN_ROOT / 'queue_state.json'),
                          base.read_json(gate.RUN_ROOT / 'gate_acceptance.json'),
                          base.read_json(gate.RUN_ROOT / 'queue_manifest.json')['source_commit'])
            if not base.read_json(gate.RUN_ROOT / 'input_contract.json')['passed']:
                raise ValueError('Train/eval input contract not accepted')
            for variant in ('block3_mean', 'token_opd'):
                if not base.read_json(gate.RUN_ROOT / variant / 'resume_gate.json')['passed']:
                    raise ValueError('Resume gate failed')
                for step in (1, 2):
                    if not base.read_json(gate.RUN_ROOT / variant / f'optimizer_inspection_step{step}.json')['passed']:
                        raise ValueError('Rank optimizer inspection failed')
            protected = base.read_json(gate.RUN_ROOT / 'protected_inputs.json')
            old.shared.verify_hashes(protected)
            free = shutil.disk_usage(ROOT).free
            if free < 1_000_000_000_000:
                raise ValueError('Less than 1TB free; no checkpoint pruning allowed')
            jobs.write_json(RUN_ROOT / 'queue_manifest.json', {
                'control_commit': control_commit, 'source_commit': TRAIN_COMMIT, 'created_at': jobs.now(),
                'variant': 'block3_mean', 'total_training_steps': 200, 'checkpoint_steps': [50, 100, 150, 200],
                'initialization': str(gate.assets.STUDENT), 'resume_mode': 'disable',
                'accepted_gate': str(gate.RUN_ROOT), 'retain_all_checkpoints': True,
                'full_eval_autostart': False, 'token_training_autostart': False, 'free_bytes_at_start': free,
                'cache_root': str(CACHE), 'cache_mount': mount,
                'previous_failed_attempt': str(ROOT / 'runs/20260920v2_qwen4_completion_blockfirst_seed21_ml2')})
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            runner(['bash', RUNTIME / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
                   RUN_ROOT / 'queue_jobs/prepare_block3', job_env=training_env())
            accepted = base.read_json(gate.RUN_ROOT / 'initial_paired_cards.json')['block3_mean']
            card = base.read_json(RUN / 'run_card.json')
            validate_card(card, accepted)
            jobs.write_json(RUN_ROOT / 'preflight.json', {'passed': True, 'card': card,
                'control_commit': control_commit, 'training_source_commit': TRAIN_COMMIT,
                'allowed_changes_from_probe': formal_card_overrides(),
                'command_sha256': gate.assets.sha256(RUN / 'command.sh')})
            runner(['bash', RUN / 'command.sh'], RUN, gpu=True, pid_name='train.pid', log_name='nohup.log')
            runner(checkpoint_audit_command(), RUN_ROOT / 'queue_jobs/checkpoints')
            jobs.write_json(RUN_ROOT / 'first_batches_acceptance.json',
                            {'passed': True, 'steps': gate.audit_rollouts(RUN, (1, 2))})
            runner([base.PLOT_PYTHON, RUNTIME / 'scripts/analyze_single_opd_diagnostics.py',
                    '--run-dir', RUN, '--output-dir', RUN / 'figures', '--label', 'Qwen4 Block3 completion'],
                   RUN_ROOT / 'queue_jobs/figures')
            old.shared.verify_hashes(protected)
            jobs.wait_for_idle()
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                'checkpoint_steps': [50, 100, 150, 200], 'full_eval_started': False, 'token_training_started': False})
        except BaseException as exc:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'updated_at': jobs.now(), 'error': str(exc)})
            raise


if __name__ == '__main__':
    main()

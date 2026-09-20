#!/usr/bin/env python3
"""Wait for accepted Qwen4 Block3 evaluations, then train the matched Token arm."""

import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_qwen4_completion_block3 as formal

gate, jobs, base = formal.gate, formal.jobs, formal.base
shared = formal.old.shared
ROOT, RUNTIME, TRAIN_COMMIT = formal.ROOT, formal.RUNTIME, formal.TRAIN_COMMIT
EVAL_COMMIT = '645a2f391f96311fe3ca3d85d0af44c0e877acbd'
PREDECESSOR = ROOT / 'runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2'
RUN_ROOT = ROOT / 'runs/20260920v1_qwen4_completion_token_seed21_ml2'
RUN = RUN_ROOT / 'token_opd'
CACHE = Path('/dev/shm/opd-q4-t1')
STEPS = (200, 150, 100, 50)


def training_env():
    env = gate.training_env(RUNTIME, TRAIN_COMMIT, RUN_ROOT, 'token_opd', 1)
    env.update(PROJECT_NAME='opd_qwen4_completion', EXP_NAME='qwen4-completion-token-opd',
               TOTAL_TRAINING_STEPS='200', DIAGNOSTIC_SAVE_STEPS=formal.SAVE_STEPS,
               STOP_AFTER_STEP='-1', ROLLOUT_ATTEMPT_ID='formal', LOCAL_CACHE_ROOT=str(CACHE / 'train'),
               BASELINE_ALIGNMENT=card_changes()['baseline_alignment'])
    return env


def card_changes():
    return {'variant': 'token_opd', 'opd_block_size': 1, 'opd_block_advantage_mode': 'sum',
            'experiment_name': 'qwen4-completion-token-opd',
            'diagnostic_output_dir': str(RUN / 'diagnostics'),
            'lossless_rollout_dir': str(RUN / 'rollouts'),
            'baseline_alignment': 'Matched completion/independent-seed Token control for completed Qwen4 Block3; original token loss'}


def validate_card(card, block3):
    expected = {**block3, **card_changes()}
    if card != expected:
        changed = [k for k in set(card) | set(expected) if card.get(k) != expected.get(k)]
        raise ValueError(f'Unapproved difference from completed Block3: {changed}')


def validate_command(command, block3):
    expected = block3.replace(str(formal.RUN), str(RUN)).replace(str(formal.CACHE), str(CACHE))
    expected = expected.replace("VARIANT='block3_mean'", "VARIANT='token_opd'")
    expected = expected.replace("EXP_NAME='qwen4-completion-block3-mean'", "EXP_NAME='qwen4-completion-token-opd'")
    if command != expected:
        raise ValueError('Token command differs beyond the approved method, experiment name and paths')


def predecessor_ready(root):
    state = base.read_json(root / 'queue_state.json')
    if state.get('status') == 'running':
        return False
    if (state.get('status') != 'complete' or state.get('evaluated_steps') != list(STEPS)
            or state.get('protected_inputs_verified') is not True):
        raise ValueError('Evaluation predecessor failed/interrupted/incomplete; no training allowed')
    manifest = base.read_json(root / 'queue_manifest.json')
    expected = {'eval_steps': list(STEPS), 'runtime_commit': TRAIN_COMMIT, 'control_commit': EVAL_COMMIT,
                'training_root': str(formal.RUN_ROOT), 'prompt_protocol': gate.PROTOCOL, 'retain_rollouts': True}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong predecessor evaluation identity/protocol')
    accepted = base.read_json(root / 'evaluation_acceptance.json')
    if (accepted.get('passed') is not True or accepted.get('complete') is not True
            or accepted.get('completed_steps') != list(STEPS)
            or set(accepted.get('models', {})) != {str(s) for s in STEPS}):
        raise ValueError('All four evaluation acceptances are required')
    for step in STEPS:
        folder = root / 'evaluations' / f'block3_mean_step{step}'
        result = accepted['models'][str(step)]
        archive = result.get('rollout_archive', {})
        if ((folder / 'exit_code.txt').read_text().strip() != '0'
                or base.read_json(folder / 'acceptance.json') != result
                or result.get('passed') is not True or result.get('completion_protocol_verified') is not True
                or result.get('model') != str(root / 'merged' / f'block3_mean_step{step}')
                or result.get('grader_sha256') != shared.grading.HISTORICAL_GRADER_SHA256
                or archive.get('passed') is not True or archive.get('num_examples') != 643
                or archive.get('num_rollouts') != 5144 or archive.get('num_archive_files') != 32
                or set(result.get('per_task', {})) != set(shared.TASK_COUNTS)
                or set(archive.get('per_task', {})) != set(shared.TASK_COUNTS)):
            raise ValueError(f'Incomplete predecessor evidence at Step{step}')
        for task, count in shared.TASK_COUNTS.items():
            for entry in (result['per_task'][task], archive['per_task'][task]):
                if entry.get('num_examples') != count or entry.get('num_rollouts') != count * 8:
                    raise ValueError(f'Incomplete predecessor {task} coverage')
            if archive['per_task'][task].get('num_archive_files') != 8:
                raise ValueError('Missing predecessor raw archives')
    return True


def checkpoint_audit_command():
    return formal.old.audit_command(RUNTIME, RUN, TRAIN_COMMIT) + [
        '--expected-diag-interval', '1', '--expected-diagnostic-steps', ','.join(map(str, range(1, 201)))]


def validate_paired_rows(block3, token, step):
    if len(block3) != 32 or len(token) != 32:
        raise ValueError('Incomplete paired trajectory batch')
    fields = ('source_extra_info', 'prompt_token_ids', 'request_identity', 'sampling', 'protocol',
              'enable_thinking', 'mask_policy', 'eos_token_id', 'request_turn', 'response_tensor_width')
    for left, right in zip(block3, token):
        if left['step'] != step or right['step'] != step or any(left[k] != right[k] for k in fields):
            raise ValueError(f'Actual paired prompt/order/seed/protocol differs at step {step}')


def audit_paired_rollouts():
    evidence = []
    for step in range(1, 201):
        paths = [run / f'rollouts/formal/step_{step:06d}/raw.jsonl.gz' for run in (formal.RUN, RUN)]
        block3, token = [gate.read_archive(path) for path in paths]
        validate_paired_rows(block3, token, step)
        evidence.append({'step': step, 'count_per_arm': 32,
                         'block3_sha256': gate.assets.sha256(paths[0]),
                         'token_sha256': gate.assets.sha256(paths[1])})
    return {'passed': True, 'steps': 200, 'trajectories_per_arm': 6400, 'evidence': evidence}


def main():
    control_root = Path(__file__).resolve().parents[1]
    commit = (control_root / 'DEPLOYED_COMMIT').read_text().strip()
    if (control_root != ROOT / 'analysis_deployments' / commit
            or Path(formal.__file__).resolve() != control_root / 'scripts/run_qwen4_completion_block3.py'
            or Path(gate.__file__).resolve() != RUNTIME / 'scripts/run_qwen_completion_gate.py'
            or (RUNTIME / 'DEPLOYED_COMMIT').read_text().strip() != TRAIN_COMMIT):
        raise ValueError('Immutable control release and accepted frozen training runtime required')

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists() or RUN.exists():
            raise FileExistsError('Existing Token attempt; no automatic overwrite or retry')
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        jobs.write_json(RUN_ROOT / 'queue_manifest.json', {
            'control_commit': commit, 'source_commit': TRAIN_COMMIT, 'created_at': jobs.now(),
            'variant': 'token_opd', 'predecessor': str(PREDECESSOR), 'paired_block3': str(formal.RUN),
            'total_training_steps': 200, 'checkpoint_steps': sorted(STEPS),
            'initialization': str(gate.assets.STUDENT), 'resume_mode': 'disable',
            'prompt_protocol': gate.PROTOCOL, 'request_seed_rule': gate.SEED_RULE,
            'retain_all_checkpoints': True, 'retain_rollouts': True,
            'accepted_gate': str(gate.RUN_ROOT), 'cache_root': str(CACHE),
            'full_eval_autostart': False, 'base_evaluation_autostart': False})
        try:
            while not predecessor_ready(PREDECESSOR):
                jobs.write_json(state, {'status': 'waiting_for_evaluation', 'predecessor': str(PREDECESSOR),
                                       'updated_at': jobs.now()})
                time.sleep(30)
            jobs.write_json(state, {'status': 'preflight', 'updated_at': jobs.now()})
            jobs.wait_for_idle()
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
            formal.validate_cache_mount(*mount, shutil.disk_usage(CACHE).free)
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
                    if job == RUN and (RUN / 'train.pid').exists():
                        formal.cleanup_failed_group(int((RUN / 'train.pid').read_text()))
                    raise

            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/runtime_hashes')
            formal.validate_gate(base.read_json(gate.RUN_ROOT / 'queue_state.json'),
                                 base.read_json(gate.RUN_ROOT / 'gate_acceptance.json'),
                                 base.read_json(gate.RUN_ROOT / 'queue_manifest.json')['source_commit'])
            if not base.read_json(gate.RUN_ROOT / 'input_contract.json')['passed']:
                raise ValueError('Training/evaluation input gate not accepted')
            if not base.read_json(gate.RUN_ROOT / 'token_opd/resume_gate.json')['passed']:
                raise ValueError('Token resume gate failed')
            for step in (1, 2):
                if not base.read_json(gate.RUN_ROOT / f'token_opd/optimizer_inspection_step{step}.json')['passed']:
                    raise ValueError('Token rank optimizer inspection failed')
            block_card = base.read_json(formal.RUN / 'run_card.json')
            accepted_cards = base.read_json(gate.RUN_ROOT / 'initial_paired_cards.json')
            formal.validate_card(block_card, accepted_cards['block3_mean'])
            if (base.read_json(formal.RUN_ROOT / 'queue_state.json').get('status') != 'complete'
                    or base.read_json(formal.RUN / 'acceptance.json').get('passed') is not True
                    or (formal.RUN / 'exit_code.txt').read_text().strip() != '0'):
                raise ValueError('Original Block3 formal training was not accepted')

            protected = base.read_json(formal.RUN_ROOT / 'protected_inputs.json')
            for name in ('artifact_hashes.sha256', 'script_hashes.sha256'):
                protected.update({str(p): digest for digest, p in base.paired.load_sha256_manifest(formal.RUN / name)})
            for name in ('run_card.json', 'command.sh', 'artifact_hashes.sha256', 'script_hashes.sha256'):
                protected[str(formal.RUN / name)] = gate.assets.sha256(formal.RUN / name)
            acceptance = base.read_json(PREDECESSOR / 'evaluation_acceptance.json')
            for result in acceptance['models'].values():
                shared.verify_hashes(result['sha256'])
            protected[str(PREDECESSOR / 'evaluation_acceptance.json')] = gate.assets.sha256(PREDECESSOR / 'evaluation_acceptance.json')
            shared.verify_hashes(protected)
            free = shutil.disk_usage(ROOT).free
            if free < 1_000_000_000_000:
                raise ValueError('Less than 1TB free; no pruning allowed')
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            runner(['bash', RUNTIME / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
                   RUN_ROOT / 'queue_jobs/prepare_token', job_env=training_env())
            card = base.read_json(RUN / 'run_card.json')
            validate_card(card, block_card)
            if card != {**accepted_cards['token_opd'], **formal.formal_card_overrides(), **card_changes()}:
                raise ValueError('Token differs from its accepted GPU/resume probe')
            validate_command((RUN / 'command.sh').read_text(), (formal.RUN / 'command.sh').read_text())
            for name in ('artifact_hashes.sha256', 'script_hashes.sha256'):
                if set(base.paired.load_sha256_manifest(RUN / name)) != set(base.paired.load_sha256_manifest(formal.RUN / name)):
                    raise ValueError(f'Paired data/model/runtime bytes differ: {name}')
            protected.update({str(RUN / name): gate.assets.sha256(RUN / name)
                              for name in ('run_card.json', 'command.sh', 'artifact_hashes.sha256', 'script_hashes.sha256')})
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            jobs.write_json(RUN_ROOT / 'preflight.json', {'passed': True, 'card': card, 'block3_card': block_card,
                'control_commit': commit, 'training_source_commit': TRAIN_COMMIT,
                'allowed_card_changes': card_changes(), 'exact_command_alignment': True,
                'model_data_runtime_hashes_matched': True, 'free_bytes': free, 'cache_mount': mount})
            shared.verify_hashes(protected)
            runner(['bash', RUN / 'command.sh'], RUN, gpu=True, pid_name='train.pid', log_name='nohup.log')
            runner(checkpoint_audit_command(), RUN_ROOT / 'queue_jobs/checkpoints')
            jobs.write_json(RUN_ROOT / 'first_batches_acceptance.json',
                            {'passed': True, 'steps': gate.audit_rollouts(RUN, (1, 2))})
            jobs.write_json(RUN_ROOT / 'paired_rollout_acceptance.json', audit_paired_rollouts())
            runner([base.PLOT_PYTHON, RUNTIME / 'scripts/analyze_single_opd_diagnostics.py',
                    '--run-dir', RUN, '--output-dir', RUN / 'figures', '--label', 'Qwen4 Token completion'],
                   RUN_ROOT / 'queue_jobs/figures')
            shared.verify_hashes(protected)
            jobs.wait_for_idle()
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                'checkpoint_steps': sorted(STEPS), 'paired_rollouts_verified': True,
                'protected_inputs_verified': True, 'full_eval_started': False})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'updated_at': jobs.now(), 'error': str(error)})
            raise


if __name__ == '__main__':
    main()

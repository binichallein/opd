#!/usr/bin/env python3
"""ml2-only: qualify official 8B teacher, then gated original 4B paired training."""

import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
TRAIN_COMMIT = '0f9161f02f08287fb07f0375ad0a6bda81133ff0'
RUNTIME = ROOT / 'deployments' / TRAIN_COMMIT
RUN_ROOT = ROOT / 'runs/20260921v1_qwen4_from8b_completion_seed21_ml2'
CACHE = Path('/dev/shm/q48')
TEACHER = ROOT / 'models/Qwen3-8B-Base'
VARIANTS = ('token_opd', 'block3_mean')
SAVE_STEPS = '50,100,150,200'
HISTORICAL = {
    'token_opd': ROOT / 'runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd',
    'block3_mean': ROOT / 'runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean',
}


def helpers():
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
    import run_qwen_completion_gate as gate
    return gate, gate.base, gate.jobs


def require_capability(result):
    if result.get('passed') is not True:
        raise ValueError('Teacher capability not accepted; no training is authorized')


def card_changes(root, variant, probe_step=None):
    if variant not in VARIANTS or probe_step not in (None, 1, 2):
        raise ValueError('Unapproved method or probe')
    run = root / variant
    return {
        'project_name': 'opd_qwen4_from8b_completion',
        'experiment_name': f'qwen4-from8b-{variant}' + ('-probe' if probe_step else ''),
        'teacher_model': str(TEACHER),
        # Frozen launcher assumes a HF revision for Qwen teachers. Pin the genuine
        # ModelScope revision in protected_inputs/asset_manifest, never forge HF metadata.
        'teacher_model_revision': '',
        'total_training_steps': 2 if probe_step else 200,
        'diagnostic_save_steps': {None: SAVE_STEPS, 1: '1', 2: '1,2'}[probe_step],
        'stop_after_step': 1 if probe_step == 1 else -1,
        'resume_mode': 'resume_path' if probe_step == 2 else 'disable',
        'resume_from_path': str(run / 'checkpoints/global_step_1') if probe_step == 2 else '',
        'rollout_attempt_id': f'probe{probe_step}' if probe_step else 'formal',
        'diagnostic_output_dir': str(run / 'diagnostics'),
        'lossless_rollout_dir': str(run / 'rollouts'),
        'baseline_alignment': 'Official 8B-Base teacher; matched original 4B student/completion/data/seed; historical losses unchanged',
    }


def training_env(root, variant, probe_step=None):
    gate, _, _ = helpers()
    env = gate.training_env(RUNTIME, TRAIN_COMMIT, root, variant, probe_step or 1)
    card = card_changes(root, variant, probe_step)
    mapping = {'PROJECT_NAME': 'project_name', 'EXP_NAME': 'experiment_name',
               'MATH_TEACHER': 'teacher_model', 'TEACHER_MODEL_REVISION': 'teacher_model_revision',
               'TOTAL_TRAINING_STEPS': 'total_training_steps', 'DIAGNOSTIC_SAVE_STEPS': 'diagnostic_save_steps',
               'STOP_AFTER_STEP': 'stop_after_step', 'RESUME_MODE': 'resume_mode',
               'RESUME_FROM_PATH': 'resume_from_path', 'ROLLOUT_ATTEMPT_ID': 'rollout_attempt_id',
               'BASELINE_ALIGNMENT': 'baseline_alignment'}
    env.update({key: str(card[field]) for key, field in mapping.items()})
    env.update(OPD_DIAG_INTERVAL='1', LOCAL_CACHE_ROOT=str(CACHE / variant / 'train'))
    return env


def validate_card(card, original, root, variant, probe_step=None):
    expected = {**original, **card_changes(root, variant, probe_step)}
    changed = [k for k in set(card) | set(expected) if card.get(k) != expected.get(k)]
    if changed:
        raise ValueError(f'Unapproved historical setting changes: {changed}')


def validate_pair(cards):
    if set(cards) != set(VARIANTS):
        raise ValueError('Both original comparison methods are required')
    allowed = {'variant', 'opd_block_size', 'opd_block_advantage_mode', 'experiment_name',
               'diagnostic_output_dir', 'lossless_rollout_dir'}
    left, right = (cards[v] for v in VARIANTS)
    for key in (set(left) | set(right)) - allowed:
        if left.get(key) != right.get(key):
            raise ValueError(f'Unapproved paired difference: {key}')


def audit_command(run, probe_step=None):
    gate, _, _ = helpers()
    command = gate.old.audit_command(RUNTIME, run, TRAIN_COMMIT, probe_step=probe_step)
    command[command.index('--expected-teacher-model-suffix') + 1] = TEACHER.name
    if not probe_step:
        command += ['--expected-diag-interval', '1', '--expected-diagnostic-steps',
                    ','.join(map(str, range(1, 201)))]
    return command


def validate_formal_command(command, historical, variant):
    old_cache = {'token_opd': '/dev/shm/opd-q4-t1', 'block3_mean': '/dev/shm/opd-q4-c3'}[variant]
    old_exp = {'token_opd': 'qwen4-completion-token-opd', 'block3_mean': 'qwen4-completion-block3-mean'}[variant]
    expected = historical.replace(str(HISTORICAL[variant]), str(RUN_ROOT / variant))
    expected = expected.replace(old_cache, str(CACHE / variant))
    expected = expected.replace(str(ROOT / 'models/Qwen3-4B-Base-GRPO'), str(TEACHER))
    expected = expected.replace("PROJECT_NAME='opd_qwen4_completion'", "PROJECT_NAME='opd_qwen4_from8b_completion'")
    expected = expected.replace(f"EXP_NAME='{old_exp}'", f"EXP_NAME='qwen4-from8b-{variant}'")
    if command != expected:
        raise ValueError('Formal command changes beyond teacher identity and output/cache names')


def audit_rollouts(run, steps):
    import torch
    from transformers import AutoTokenizer
    gate, _, _ = helpers()
    sys.path.insert(0, str(RUNTIME / 'external/revisiting_opd'))
    from verl.utils.torch_functional import get_response_mask
    from opd_ext.math_protocol import completion_input_ids
    from opd_ext.request_seeds import request_identities
    from diagnose_token_truncation import analyze_tokens
    tokenizer = AutoTokenizer.from_pretrained(gate.assets.STUDENT, local_files_only=True)
    result = []
    for step in steps:
        paths = list((run / 'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(paths) != 1:
            raise ValueError(f'Incomplete or duplicated raw archives at step {step}')
        rows = gate.read_archive(paths[0])
        historical = gate.read_archive(HISTORICAL['token_opd'] / f'rollouts/formal/step_{step:06d}/raw.jsonl.gz')
        if len(rows) != 32 or len(historical) != 32 or len({r['traj_uid'] for r in rows}) != 32:
            raise ValueError('Incomplete rollout coverage')
        sources = [historical[i]['source_extra_info'] for i in range(0, 32, 8)]
        identities = request_identities(sources, group_size=8, global_seed=21, step=step)
        for row, old, identity in zip(rows, historical, identities):
            if (row['source_extra_info'] != old['source_extra_info']
                    or row['prompt_token_ids'] != completion_input_ids(tokenizer, row['source_extra_info']['question'])):
                raise ValueError('Actual source order or train/eval input changed')
            gate.validate_seed_settings(row, identity, step)
            ids, count = row['training_response_token_ids'], row['response_length']
            mask = get_response_mask(torch.tensor([ids]), 151643, dtype=torch.long)[0].tolist()
            if (len(ids) != 16384 or row['response_tensor_width'] != 16384
                    or count != len(row['response_token_ids']) or not 0 <= count <= 16384
                    or row['padding_length'] != 16384-count or ids[:count] != row['response_token_ids']
                    or mask != row['training_response_mask']
                    or len(row['training_rollout_log_probs']) != 16384
                    or not all(math.isfinite(v) for v in row['training_rollout_log_probs'])
                    or row['finish_reason'] not in ('length', 'stop')
                    or (row['finish_reason'] == 'length' and count != 16384)):
                raise ValueError('Raw response/logprob/mask/termination mismatch')
        result.append({'step': step, 'count': 32, 'sha256': gate.assets.sha256(paths[0]),
                       'historical_inputs_and_seeds_match': True,
                       'length_stops': sum(r['finish_reason'] == 'length' for r in rows),
                       'periodic_tails': sum(analyze_tokens(r['response_token_ids'], 16384)['tail_period'] is not None for r in rows)})
    return {'passed': True, 'steps': result}


def verify_optimizer_state(run, steps):
    import torch
    gate, _, jobs = helpers()
    for step in steps:
        ranks = []
        for rank in range(4):
            path = run / f'checkpoints/global_step_{step}/actor/optim_world_size_4_rank_{rank}.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            states = saved.get('state', {})
            if not states or not saved.get('param_groups'):
                raise ValueError('Missing actual optimizer buffers')
            nonzero = 0
            for state in states.values():
                if not {'step', 'exp_avg', 'exp_avg_sq'}.issubset(state):
                    raise ValueError('Incomplete Adam optimizer state')
                if float(state['step']) != step:
                    raise ValueError('Optimizer step does not match checkpoint')
                for name in ('exp_avg', 'exp_avg_sq'):
                    value = state[name]
                    value = value.to_local() if hasattr(value, 'to_local') else value
                    if not torch.isfinite(value).all().item():
                        raise ValueError('Nonfinite optimizer moments')
                    nonzero += int(torch.count_nonzero(value).item())
            if nonzero == 0:
                raise ValueError('No optimizer update observed')
            ranks.append({'rank': rank, 'states': len(states), 'nonzero_moment_elements': nonzero,
                          'sha256': gate.assets.sha256(path)})
            del saved, states
        jobs.write_json(run / f'optimizer_inspection_step{step}.json', {'passed': True, 'ranks': ranks})


def verify_formal_training_states(run):
    import torch
    _, base, jobs = helpers()
    steps = (50, 100, 150, 200)
    verify_optimizer_state(run, steps)
    records = []
    for step in steps:
        checkpoint = run / f'checkpoints/global_step_{step}'
        data = torch.load(checkpoint / 'data.pt', map_location='cpu', weights_only=False)
        for rank in range(4):
            extra = torch.load(checkpoint / f'actor/extra_state_world_size_4_rank_{rank}.pt',
                               map_location='cpu', weights_only=False)
            base.validate_resume_state(extra, data, step)
            records.append({'step': step, 'rank': rank, 'rng_scheduler_sampler_valid': True})
    jobs.write_json(run / 'training_state_acceptance.json', {'passed': True, 'states': records})


def main():
    gate, base, jobs = helpers()
    import prepare_qwen8_teacher_assets as assets
    from run_qwen4_completion_block3 import cleanup_failed_group, validate_cache_mount
    from recover_historical17_llama import check_socket_budget
    control = Path(__file__).resolve().parents[1]
    commit = (control / 'DEPLOYED_COMMIT').read_text().strip()
    if (control != ROOT / 'analysis_deployments' / commit
            or (RUNTIME / 'DEPLOYED_COMMIT').read_text().strip() != TRAIN_COMMIT):
        raise ValueError('Immutable ml2 control and historical training deployments required')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt; never overwrite or auto-retry')
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        jobs.write_json(RUN_ROOT / 'queue_manifest.json', {
            'created_at': jobs.now(), 'control_commit': commit, 'training_commit': TRAIN_COMMIT,
            'teacher': assets.REPO, 'teacher_revision': assets.REVISION, 'teacher_provider': 'modelscope',
            'student': gate.assets.REPO, 'student_revision': gate.assets.REVISION,
            'variants': VARIANTS, 'total_training_steps': 200, 'checkpoint_steps': [50,100,150,200],
            'retain_all_checkpoints': True, 'retain_all_rollouts': True,
            'conditional_training_autostart': True, 'full_benchmark_eval_autostart': False,
            'qualification_plan_sha256': gate.assets.sha256(control / 'docs/plans/2026-09-21-qwen8-teacher-acceptance.md')})
        try:
            jobs.wait_for_idle()
            if shutil.disk_usage(ROOT).free < 1_000_000_000_000:
                raise ValueError('Less than 1TB free; no pruning allowed')
            CACHE.mkdir(exist_ok=True)
            mount = subprocess.run(['findmnt', '-T', str(CACHE), '-n', '-o', 'FSTYPE,OPTIONS'],
                                   check=True, capture_output=True, text=True).stdout.strip().split(maxsplit=1)
            validate_cache_mount(*mount, shutil.disk_usage(CACHE).free)
            env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
                       PYTHONPATH=f'{control}:{control}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                       PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                       RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
                       VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(base.GRADER))
            if env.get('RAY_TMPDIR'):
                raise ValueError('Unexpected inherited RAY_TMPDIR')
            for key, suffix in {'TMPDIR':'tmp', 'VLLM_CACHE_ROOT':'vllm', 'TORCHINDUCTOR_CACHE_DIR':'inductor',
                                'TRITON_CACHE_DIR':'triton', 'CUDA_CACHE_PATH':'cuda', 'OUTLINES_CACHE_DIR':'outlines'}.items():
                (CACHE / suffix).mkdir(exist_ok=True)
                env[key] = str(CACHE / suffix)
            for variant in VARIANTS:
                check_socket_budget(CACHE / variant / 'train/tmp')
            def runner(command, job, job_env=None, runtime=control, **kwargs):
                try:
                    jobs.run_job(command, job, runtime, state, {**env, **(job_env or {})}, **kwargs)
                except BaseException:
                    if kwargs.get('gpu'):
                        pidfile = job / kwargs.get('pid_name', 'job.pid')
                        if pidfile.exists():
                            cleanup_failed_group(int(pidfile.read_text()))
                    raise
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/control_hashes')
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/runtime_hashes', runtime=RUNTIME)
            runner([base.PYTHON, control / 'scripts/prepare_qwen8_teacher_assets.py', '--download'],
                   RUN_ROOT / 'queue_jobs/teacher_assets')
            protected = assets.verify_assets()
            for path in [base.DATA/'train.parquet', base.DATA/'test.parquet', base.DATA/'manifest.json', base.GRADER,
                         *[base.DATA/'eval_jsonl'/f'{task}.jsonl' for task in jobs.TASKS]]:
                protected[str(path)] = gate.assets.sha256(path)
            if protected[str(base.DATA/'train.parquet')] != jobs.TRAIN_SHA:
                raise ValueError('Changed DAPO training pool')
            gate.old.grading.validate_grader_hash(base.GRADER, gate.old.grading.HISTORICAL_GRADER_SHA256)
            for variant in VARIANTS:
                for name in ('run_card.json', 'command.sh'):
                    path = HISTORICAL[variant] / name
                    protected[str(path)] = gate.assets.sha256(path)
            jobs.write_json(RUN_ROOT/'protected_inputs.json', protected)
            qualification = RUN_ROOT / 'teacher_qualification'
            script = control / 'scripts/qualify_qwen8_teacher.py'
            runner([base.PYTHON, script, 'prepare', '--root', qualification], RUN_ROOT/'queue_jobs/qualify_prepare')
            for phase in ('direct', 'continuation'):
                for label, model in (('student', gate.assets.STUDENT), ('teacher', TEACHER), ('reference', gate.assets.TEACHER)):
                    runner([base.PYTHON, script, 'generate', '--root', qualification, '--model', model,
                            '--label', label, '--phase', phase, '--gpu', '0'],
                           RUN_ROOT/f'queue_jobs/qualify_{phase}_{label}', gpu=True)
            runner([base.PYTHON, script, 'summarize', '--root', qualification], RUN_ROOT/'queue_jobs/qualify_summary')
            acceptance = base.read_json(qualification/'gate_acceptance.json')
            if acceptance.get('passed') is not True:
                jobs.write_json(state, {'status': 'capability_not_accepted', 'updated_at': jobs.now(),
                                       'report': str(qualification/'gate_acceptance.json'), 'training_started': False})
                return
            require_capability(acceptance)
            protected[str(qualification/'gate_acceptance.json')] = gate.assets.sha256(qualification/'gate_acceptance.json')
            cards = {}
            for variant in VARIANTS:
                original = base.read_json(HISTORICAL[variant] / 'run_card.json')
                probe_root, probe = RUN_ROOT/'probes', RUN_ROOT/'probes'/variant
                for step in (1, 2):
                    runner(['bash', RUNTIME/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                           RUN_ROOT/f'queue_jobs/prepare_{variant}_probe{step}',
                           job_env=training_env(probe_root, variant, step), runtime=RUNTIME)
                    validate_card(base.read_json(probe/'run_card.json'), original, probe_root, variant, step)
                    job = RUN_ROOT/f'queue_jobs/{variant}_probe{step}'
                    runner(['bash', probe/'command.sh'], job, runtime=RUNTIME, gpu=True,
                           job_env={'PYTHONPATH': f'{RUNTIME}:{RUNTIME}/external/revisiting_opd'})
                    shutil.copyfile(job/'logs/job.log', probe/'logs/nohup.log')
                    runner(audit_command(probe, step), RUN_ROOT/f'queue_jobs/{variant}_audit{step}', runtime=RUNTIME)
                    jobs.write_json(probe/f'rollout_acceptance_step{step}.json', audit_rollouts(probe, range(1, step+1)))
                    verify_optimizer_state(probe, (step,))
                base.audit_resume(probe)
                runner(['bash', RUNTIME/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                       RUN_ROOT/f'queue_jobs/prepare_{variant}_formal', job_env=training_env(RUN_ROOT, variant), runtime=RUNTIME)
                run = RUN_ROOT / variant
                cards[variant] = base.read_json(run/'run_card.json')
                validate_card(cards[variant], original, RUN_ROOT, variant)
                validate_formal_command((run/'command.sh').read_text(), (HISTORICAL[variant]/'command.sh').read_text(), variant)
                for name in ('run_card.json', 'command.sh', 'artifact_hashes.sha256', 'script_hashes.sha256'):
                    protected[str(run/name)] = gate.assets.sha256(run/name)
            validate_pair(cards)
            for name in ('artifact_hashes.sha256', 'script_hashes.sha256'):
                if (set(base.paired.load_sha256_manifest(RUN_ROOT/VARIANTS[0]/name)) !=
                        set(base.paired.load_sha256_manifest(RUN_ROOT/VARIANTS[1]/name))):
                    raise ValueError('Paired artifact/runtime hashes differ')
            jobs.write_json(RUN_ROOT/'engineering_acceptance.json', {'passed': True, 'cards': cards,
                'teacher_revision_manifest_pinned': assets.REVISION, 'resume_accepted': list(VARIANTS),
                'historical_command_alignment': True})
            jobs.write_json(RUN_ROOT/'protected_inputs.json', protected)
            for variant in VARIANTS:
                gate.old.shared.verify_hashes(protected)
                run = RUN_ROOT / variant
                runner(['bash', run/'command.sh'], run, runtime=RUNTIME, gpu=True,
                       pid_name='train.pid', log_name='nohup.log',
                       job_env={'PYTHONPATH': f'{RUNTIME}:{RUNTIME}/external/revisiting_opd'})
                runner(audit_command(run), RUN_ROOT/f'queue_jobs/{variant}_checkpoints', runtime=RUNTIME)
                verify_formal_training_states(run)
                jobs.write_json(run/'rollout_acceptance.json', audit_rollouts(run, range(1, 201)))
                runner([base.PLOT_PYTHON, RUNTIME/'scripts/analyze_single_opd_diagnostics.py',
                        '--run-dir', run, '--output-dir', run/'figures', '--label', f'Qwen4 from8B {variant}'],
                       RUN_ROOT/f'queue_jobs/{variant}_figures', runtime=RUNTIME)
            gate.old.shared.verify_hashes(protected)
            jobs.wait_for_idle()
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                                   'variants': list(VARIANTS), 'protected_inputs_verified': True,
                                   'full_benchmark_eval_started': False})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'updated_at': jobs.now(), 'error': str(error)})
            raise


if __name__ == '__main__':
    main()

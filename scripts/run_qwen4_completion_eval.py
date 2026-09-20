#!/usr/bin/env python3
"""Evaluate the four completed Qwen4 completion checkpoints, Step200 first."""

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
TRAIN_COMMIT = '0f9161f02f08287fb07f0375ad0a6bda81133ff0'
RUNTIME = ROOT / 'deployments' / TRAIN_COMMIT
TRAIN_ROOT = ROOT / 'runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2'
RUN_ROOT = ROOT / 'runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2'
CACHE = Path('/dev/shm/opd-q4-e1')
STEPS = (200, 150, 100, 50)


@dataclass(frozen=True)
class EvaluationSpec:
    variant: str
    train_root: Path
    run_root: Path
    cache: Path
    block_size: int
    advantage_mode: str


BLOCK_EVAL = EvaluationSpec('block3_mean', TRAIN_ROOT, RUN_ROOT, CACHE, 3, 'mean')
TOKEN_EVAL = EvaluationSpec('token_opd',
    ROOT / 'runs/20260920v1_qwen4_completion_token_seed21_ml2',
    ROOT / 'runs/20260921v1_qwen4_completion_token_eval_seed21_ml2',
    Path('/dev/shm/opd-q4-te1'), 1, 'sum')

# Inference, merger and grader helpers stay on the GPU-accepted training release.
sys.path[:0] = [str(RUNTIME / 'scripts'), str(RUNTIME), str(Path(__file__).resolve().parent)]
import run_qwen_completion_gate as gate
from opd_ext.math_protocol import completion_input_ids, completion_math_prompt

old, jobs, base = gate.old, gate.jobs, gate.base
assets, shared, historical = gate.assets, old.shared, old.historical
grading, PROTOCOL = shared.grading, gate.PROTOCOL


def model_paths(step, spec=BLOCK_EVAL):
    if step not in STEPS:
        raise ValueError('Only the four completed checkpoints are authorized')
    return (spec.train_root / f'{spec.variant}/checkpoints/global_step_{step}/actor',
            spec.run_root / f'merged/{spec.variant}_step{step}')


def evaluation_command(model, output):
    command = jobs.eval_command(RUNTIME, base.PYTHON, model, base.DATA / 'eval_jsonl', output, assets.STUDENT)
    command[command.index('--grader') + 1] = 'external'
    return command + ['--retain-rollouts', '--prompt-protocol', PROTOCOL]


def validate_training(state, acceptance, card, spec=BLOCK_EVAL):
    expected = {'source_commit': TRAIN_COMMIT, 'opd_prompt_protocol': PROTOCOL,
                'variant': spec.variant, 'student_model': str(assets.STUDENT),
                'opd_block_size': spec.block_size, 'opd_block_advantage_mode': spec.advantage_mode}
    if (state.get('status') != 'complete' or state.get('checkpoint_steps') != sorted(STEPS)
            or acceptance.get('passed') is not True
            or any(card.get(k) != v for k, v in expected.items())):
        raise ValueError('Completed, accepted Qwen4 completion training is required')


def validate_paired_training(acceptance):
    if (acceptance.get('passed') is not True or acceptance.get('steps') != 200
            or acceptance.get('trajectories_per_arm') != 6400):
        raise ValueError('All 200 steps and 6400 paired rollout inputs must have passed acceptance')


def validate_completion_row(row, tokenizer):
    if (row.get('prompt_protocol') != PROTOCOL or row.get('enable_thinking') is not False
            or row.get('rendered_prompt') != completion_math_prompt(row['problem'])
            or row.get('prompt_token_ids') != completion_input_ids(tokenizer, row['problem'])
            or row.get('eos_token_id') != 151643
            or row.get('sampling', {}).get('stop_token_ids') != []):
        raise ValueError('Actual completion input, thinking flag or EOS contract changed')


def prompt_contract(model):
    from transformers import AutoTokenizer
    reference = AutoTokenizer.from_pretrained(assets.STUDENT, local_files_only=True)
    target = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if reference.get_vocab() != target.get_vocab() or target.eos_token_id != 151643:
        raise ValueError('Merged tokenizer differs from the original training student')
    digest, count, maximum = hashlib.sha256(), 0, 0
    for task, expected in shared.TASK_COUNTS.items():
        rows = grading.load_jsonl(base.DATA / 'eval_jsonl' / f'{task}.jsonl')
        if len(rows) != expected or len({str(row['id']) for row in rows}) != expected:
            raise ValueError('Incomplete benchmark identities')
        for row in rows:
            ids = completion_input_ids(target, row['problem'])
            if ids != completion_input_ids(reference, row['problem']):
                raise ValueError('Merged model evaluation differs from training input token IDs')
            digest.update(json.dumps([task, row['id'], ids]).encode())
            count += 1
            maximum = max(maximum, len(ids))
    return {'passed': True, 'examples': count, 'max_prompt_tokens': maximum,
            'prompt_protocol': PROTOCOL, 'enable_thinking': False,
            'actual_input_sha256': digest.hexdigest(), 'stop_token_ids': [], 'eos_token_id': 151643}


def per_benchmark_results(results):
    return {task: {f'step{step}': result['per_task'][task] for step, result in results.items()}
            for task in shared.TASK_COUNTS}


def evaluate(step, runner, protected, control_commit, spec=BLOCK_EVAL):
    from transformers import AutoTokenizer
    shared.verify_hashes(protected)
    actor, model = model_paths(step, spec)
    folder = spec.run_root / 'evaluations' / f'{spec.variant}_step{step}'
    folder.mkdir(parents=True, exist_ok=False)
    if model.exists():
        raise FileExistsError(model)
    source_files = [actor / f'model_world_size_4_rank_{rank}.pt' for rank in range(4)]
    source_files += [p for p in actor.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja', '.txt')]
    source_hashes = {str(p): assets.sha256(p) for p in source_files}
    jobs.write_json(folder / 'checkpoint_identity.json', {'checkpoint': str(actor), 'sha256': source_hashes})
    runner([base.PYTHON, RUNTIME / 'external/revisiting_opd/scripts/model_merger.py', 'merge',
            '--backend', 'fsdp', '--local_dir', actor, '--target_dir', model],
           spec.run_root / f'queue_jobs/merge_step{step}', job_env={'CUDA_VISIBLE_DEVICES': ''})
    shared.verify_hashes(source_hashes)
    if not list(model.glob('*.safetensors')):
        raise ValueError('Merged evaluation weights are missing')
    model_hashes = {str(p): assets.sha256(p) for p in model.iterdir() if p.is_file()}
    jobs.write_json(folder / 'model_identity.json', {'model': str(model), 'sha256': model_hashes})
    jobs.write_json(folder / 'prompt_contract.json', prompt_contract(model))
    jobs.write_json(folder / 'eval_card.json', {**shared.EVAL_VALUES, 'model': str(model),
        'checkpoint_step': step, 'variant': spec.variant,
        'training_source_commit': TRAIN_COMMIT, 'control_commit': control_commit,
        'tasks': shared.TASK_COUNTS, 'prompt_protocol': PROTOCOL, 'retain_rollouts': True,
        'grader_path': str(base.GRADER), 'grader_sha256': grading.HISTORICAL_GRADER_SHA256,
        'reporting': 'Per-benchmark only; do not use the legacy generator macro fields as results.'})
    runner(evaluation_command(model, folder / 'outputs'), folder, gpu=True,
           pid_name='eval.pid', log_name='eval.log')
    shared.verify_hashes(model_hashes)
    shared.verify_hashes(protected)
    output = folder / 'outputs'
    config = base.read_json(output / 'eval_config.json')
    if config.get('prompt_protocol') != PROTOCOL or config.get('retain_rollouts') is not True:
        raise ValueError('Evaluation used the wrong prompt or omitted raw trajectory logging')
    accepted = shared.audit_evaluation(output, base.DATA / 'eval_jsonl', model)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    for task in shared.TASK_COUNTS:
        for row in grading.load_jsonl(grading.raw_output_path(output, task, config)):
            validate_completion_row(row, tokenizer)
    accepted['completion_protocol_verified'] = True
    return historical.accept_archive(folder, accepted)


def main(spec=BLOCK_EVAL):
    control_root = Path(__file__).resolve().parents[1]
    commit = (control_root / 'DEPLOYED_COMMIT').read_text().strip()
    if (control_root != ROOT / 'analysis_deployments' / commit
            or (RUNTIME / 'DEPLOYED_COMMIT').read_text().strip() != TRAIN_COMMIT
            or Path(gate.__file__).resolve() != RUNTIME / 'scripts/run_qwen_completion_gate.py'):
        raise ValueError('Immutable controller and frozen inference runtime are required')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    run_root, cache = spec.run_root, spec.cache
    run_root.mkdir(exist_ok=True)
    state = run_root / 'queue_state.json'
    with (run_root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run_root / 'queue_manifest.json').exists() or (run_root / 'evaluations').exists():
            raise FileExistsError('Existing evaluation attempt; never overwrite or automatically retry')
        (run_root / 'queue.pid').write_text(str(os.getpid()) + '\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
                   PYTHONPATH=f'{RUNTIME}:{RUNTIME}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(base.GRADER))
        def runner(command, job, job_env=None, **kwargs):
            jobs.run_job(command, job, RUNTIME, state, {**env, **(job_env or {})}, **kwargs)
        try:
            jobs.wait_for_idle()
            cache.mkdir(parents=True, exist_ok=True)
            mount = subprocess.run(['findmnt', '-T', str(cache), '-n', '-o', 'FSTYPE,OPTIONS'],
                                   capture_output=True, text=True, check=True).stdout.strip().split(maxsplit=1)
            if (mount[0] != 'tmpfs' or 'noexec' in mount[1].split(',')
                    or shutil.disk_usage(cache).free < 20_000_000_000
                    or shutil.disk_usage(ROOT).free < 200_000_000_000):
                raise ValueError('Insufficient persistent or executable temporary storage; never prune checkpoints')
            for key, suffix in {'TMPDIR': 'tmp', 'TRITON_CACHE_DIR': 'triton', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                                'VLLM_CACHE_ROOT': 'vllm', 'CUDA_CACHE_PATH': 'cuda', 'OUTLINES_CACHE_DIR': 'outlines'}.items():
                (cache / suffix).mkdir(parents=True, exist_ok=True)
                env[key] = str(cache / suffix)
            runner(['sha256sum', '-c', '.expected.sha256'], run_root / 'queue_jobs/runtime_hashes')
            training = spec.train_root / spec.variant
            validate_training(base.read_json(spec.train_root / 'queue_state.json'),
                              base.read_json(training / 'acceptance.json'),
                              base.read_json(training / 'run_card.json'), spec)
            if spec.variant == 'token_opd':
                validate_paired_training(base.read_json(spec.train_root / 'paired_rollout_acceptance.json'))
            if (training / 'exit_code.txt').read_text().strip() != '0':
                raise ValueError('Training did not exit successfully')
            manifest = base.read_json(training / 'data_manifest.json')
            protected = {str(base.DATA / 'eval_jsonl' / f'{task}.jsonl'): manifest['sha256'][f'{task}_jsonl']
                         for task in shared.TASK_COUNTS}
            protected[str(base.GRADER)] = grading.HISTORICAL_GRADER_SHA256
            shared.verify_hashes(protected)
            grade = grading.load_grader(base.GRADER)
            if not grade(r'\boxed{2}', '2') or grade(r'\boxed{3}', '2'):
                raise ValueError('Historical grader self-test failed')
            jobs.write_json(run_root / 'protected_inputs.json', protected)
            jobs.write_json(run_root / 'queue_manifest.json', {
                'created_at': jobs.now(), 'control_commit': commit, 'runtime_commit': TRAIN_COMMIT,
                'training_root': str(spec.train_root), 'variant': spec.variant,
                'eval_steps': STEPS, 'tasks': shared.TASK_COUNTS,
                **shared.EVAL_VALUES, 'prompt_protocol': PROTOCOL, 'retain_rollouts': True,
                'cache_root': str(cache), 'cache_mount': mount, 'checkpoint_retention': 'all',
                'token_training_autostart': False, 'base_evaluation_autostart': False})
            results = {}
            for step in STEPS:
                results[step] = evaluate(step, runner, protected, commit, spec)
                jobs.write_json(run_root / 'per_benchmark_results.json', per_benchmark_results(results))
                jobs.write_json(run_root / 'evaluation_acceptance.json', {'passed': True,
                                'complete': len(results) == len(STEPS),
                                'completed_steps': list(results), 'models': results})
            jobs.wait_for_idle()
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                            'evaluated_steps': list(results), 'protected_inputs_verified': True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'updated_at': jobs.now(), 'error': str(error)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('block3_mean', 'token_opd'), default='block3_mean')
    args = parser.parse_args()
    main(TOKEN_EVAL if args.variant == 'token_opd' else BLOCK_EVAL)

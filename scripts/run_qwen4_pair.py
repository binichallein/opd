#!/usr/bin/env python3
"""Fixed ml2 experiment: Block3 first, late initial evaluation, then Token OPD."""

import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import prepare_qwen4_assets as assets
import run_historical17_reeval_llama as historical
from opd_ext.math_protocol import QWEN_HISTORICAL_PROTOCOL as PROTOCOL

base, jobs, shared = historical.base, historical.jobs, historical.shared
grading = shared.grading
ROOT = assets.ROOT
RUN_ROOT = ROOT / 'runs/20260919v1_qwen4_blockfirst_seed21_ml2'
PREDECESSOR = ROOT / 'runs/20260918v4_llama32_historical17_recovery_seed21_ml2'
CACHE = Path('/limx_embap/tos/q4/a1')
VARIANTS = ('block3_mean', 'token_opd')
STEPS = (200, 150, 100, 50)
SAVE_STEPS = '50,100,150,200'


def training_env(runtime, commit, root, variant, probe_step=None):
    env = base.launcher_env(runtime, commit, root, variant, probe_step)
    env.update(PROJECT_NAME='opd_qwen4_pair', EXP_NAME=f'qwen4-{variant}' + ('-probe' if probe_step else ''),
               STUDENT_MODEL=str(assets.STUDENT), STUDENT_MODEL_REVISION=assets.REVISION,
               MATH_TEACHER=str(assets.TEACHER), TEACHER_MODEL_REVISION='', OPD_PROMPT_PROTOCOL=PROTOCOL,
               LOSSLESS_ROLLOUT_DIR=str(root / variant / 'rollouts'),
               ROLLOUT_ATTEMPT_ID=f'probe{probe_step}' if probe_step else 'formal',
               LOCAL_CACHE_ROOT=str(CACHE / 'train'),
               BASELINE_ALIGNMENT='Qwen4 Base capacity transfer; historical Qwen17 prompt, sampling, EOS mask and original loss')
    if probe_step is None:
        env['DIAGNOSTIC_SAVE_STEPS'] = SAVE_STEPS
    return env


def audit_command(runtime, run, commit, probe_step=None):
    command = list(map(str, base.audit_command(runtime, run, commit, probe_step=probe_step)))
    changes = {'--expected-student-model-suffix': assets.STUDENT.name,
               '--expected-teacher-model-suffix': assets.TEACHER.name,
               '--expected-student-model-revision': assets.REVISION}
    if probe_step is None:
        changes['--checkpoint-steps'] = SAVE_STEPS
    for flag, value in changes.items():
        command[command.index(flag) + 1] = value
    return command


def validate_pair(cards, commit):
    if set(cards) != set(VARIANTS):
        raise ValueError('Unexpected comparison arms')
    left, right = (cards[v] for v in VARIANTS)
    allowed = {'variant', 'opd_block_size', 'opd_block_advantage_mode', 'experiment_name',
               'diagnostic_output_dir', 'lossless_rollout_dir'}
    for key in (set(left) | set(right)) - allowed:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f'Unapproved paired difference: {key}')
    common = {**base.paired.EXPECTED_TRAINING_VALUES, 'source_commit': commit,
              'diagnostic_save_steps': SAVE_STEPS, 'student_model': str(assets.STUDENT),
              'student_model_revision': assets.REVISION, 'teacher_model': str(assets.TEACHER),
              'teacher_model_revision': '', 'opd_prompt_protocol': PROTOCOL,
              'opd_window_mode': 'fixed', 'ppo_epochs': 1, 'resume_mode': 'disable',
              'resume_from_path': '', 'rollout_attempt_id': 'formal', 'expected_train_sha256': jobs.TRAIN_SHA}
    for variant, size, mode in (('block3_mean', 3, 'mean'), ('token_opd', 1, 'sum')):
        for key, value in {**common, 'variant': variant, 'opd_block_size': size,
                           'opd_block_advantage_mode': mode}.items():
            if cards[variant].get(key) != value:
                raise ValueError(f'{variant}: {key} must equal {value!r}')


def execute_ordered(evaluate, train):
    results = {}
    def accepted(name):
        result = evaluate(name)
        if result.get('passed') is not True:
            raise ValueError(f'Incomplete evaluation: {name}')
        results[name] = result
    train('block3_mean')
    for step in STEPS:
        accepted(f'block3_mean_step{step}')
    accepted('student_base')
    train('token_opd')
    for step in STEPS:
        accepted(f'token_opd_step{step}')
    return results


def model_paths(root, name):
    if name == 'student_base':
        return None, assets.STUDENT
    if name not in {f'{v}_step{s}' for v in VARIANTS for s in STEPS}:
        raise ValueError('Unapproved model evaluation')
    variant, step = name.rsplit('_step', 1)
    return root / variant / f'checkpoints/global_step_{step}/actor', root / 'merged' / name


def preflight():
    state = base.read_json(PREDECESSOR / 'queue_state.json')
    accepted = base.read_json(PREDECESSOR / 'pair_acceptance.json')
    expected = {'student_base'} | {f'{v}_step{s}' for v in VARIANTS for s in (50, 100, 200)}
    if (state.get('status') != 'complete' or state.get('protected_inputs_verified') is not True
            or set(state.get('evaluation_models', [])) != expected or accepted.get('passed') is not True
            or set(accepted.get('models', {})) != expected):
        raise ValueError('Llama predecessor lacks full completion evidence')
    for result in accepted['models'].values():
        if result.get('passed') is not True:
            raise ValueError('Predecessor has an incomplete evaluation')
        shared.verify_hashes(result['sha256'])
    if shutil.disk_usage(ROOT).free < 400_000_000_000:
        raise ValueError('Less than 400 GB free; preserve all old checkpoints')
    protected = assets.verify_assets()
    old = base.read_json(PREDECESSOR / 'protected_inputs.json')
    for path in [base.DATA / 'train.parquet', base.DATA / 'test.parquet',
                 *[base.DATA / 'eval_jsonl' / f'{task}.jsonl' for task in shared.TASK_COUNTS]]:
        digest = assets.sha256(path)
        if old.get(str(path)) != digest:
            raise ValueError(f'Changed historical data: {path}')
        protected[str(path)] = digest
    if protected[str(base.DATA / 'train.parquet')] != jobs.TRAIN_SHA:
        raise ValueError('Wrong DAPO pool')
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    protected[str(base.GRADER)] = grading.HISTORICAL_GRADER_SHA256
    grader = grading.load_grader(base.GRADER)
    if not grader(r'\boxed{2}', '2') or grader(r'\boxed{3}', '2'):
        raise ValueError('Historical grader self-test failed')
    return protected


def prepare_pair(runtime, commit, runner):
    for variant in VARIANTS:
        runner(['bash', runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
               RUN_ROOT / f'queue_jobs/prepare_{variant}',
               job_env=training_env(runtime, commit, RUN_ROOT, variant))
    cards = {v: base.read_json(RUN_ROOT / v / 'run_card.json') for v in VARIANTS}
    validate_pair(cards, commit)
    manifests = [base.paired.load_sha256_manifest(RUN_ROOT / v / 'artifact_hashes.sha256') for v in VARIANTS]
    if set(manifests[0]) != set(manifests[1]):
        raise ValueError('Paired model/data/runtime hashes differ')
    frozen = {str(RUN_ROOT / v / name): assets.sha256(RUN_ROOT / v / name)
              for v in VARIANTS for name in ('run_card.json', 'command.sh', 'artifact_hashes.sha256')}
    jobs.write_json(RUN_ROOT / 'paired_preflight.json', {'passed': True, 'cards': cards, 'sha256': frozen})
    return frozen


def audit_rollouts(run, steps):
    import torch
    from transformers import AutoTokenizer
    from verl.utils.torch_functional import get_response_mask
    from opd_ext.math_protocol import validate_qwen_historical_prompt
    tokenizer = AutoTokenizer.from_pretrained(str(assets.STUDENT), local_files_only=True)
    evidence = []
    for step in steps:
        files = list((run / 'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(files) != 1:
            raise ValueError(f'Incomplete or duplicate rollout archive: step {step}')
        path = files[0]
        digest = path.with_suffix(path.suffix + '.sha256').read_text().split()[0]
        if assets.sha256(path) != digest:
            raise ValueError('Raw training archive hash mismatch')
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != 32 or len({row['traj_uid'] for row in rows}) != 32:
            raise ValueError('Incomplete training trajectory coverage')
        for row in rows:
            validate_qwen_historical_prompt(tokenizer, row['source_extra_info']['question'],
                                           row['prompt_token_ids'], max_prompt_length=2048)
            expected = {'temperature': 1., 'top_p': .9, 'seed': 21, 'max_tokens': 16384,
                        'n': 1, 'ignore_eos': False, 'stop_token_ids': []}
            if (row['protocol'] != PROTOCOL or row['enable_thinking'] is not None
                    or row['step'] != step or row['mask_policy'] != 'historical_eos_mask'
                    or any(row['sampling'].get(key) != value for key, value in expected.items())
                    or row['finish_reason'] not in ('length', 'stop')):
                raise ValueError('Historical training protocol changed')
            ids = row['training_response_token_ids']
            mask = get_response_mask(torch.tensor([ids]), row['eos_token_id'], dtype=torch.long)[0].tolist()
            count = row['response_length']
            if (mask != row['training_response_mask'] or len(ids) != row['response_tensor_width']
                    or ids[:count] != row['response_token_ids'] or count != len(row['response_token_ids'])
                    or len(row['training_rollout_log_probs']) != len(ids)
                    or not all(math.isfinite(v) for v in row['training_rollout_log_probs'])):
                raise ValueError('Archived full tensors differ from historical EOS-mask contract')
        evidence.append({'step': step, 'n': len(rows), 'sha256': digest,
                         'length_stops': sum(r['finish_reason'] == 'length' for r in rows),
                         'generated_think_tags': sum(r['generated_think_tags'] for r in rows),
                         'unique_response_count': len({tuple(r['response_token_ids']) for r in rows})})
    return evidence


def train_model(variant, runtime, commit, runner, protected):
    shared.verify_hashes(protected)
    probe = RUN_ROOT / 'probes' / variant
    for step in (1, 2):
        runner(['bash', runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
               RUN_ROOT / f'queue_jobs/prepare_{variant}_probe{step}',
               job_env=training_env(runtime, commit, RUN_ROOT / 'probes', variant, step))
        job = RUN_ROOT / f'queue_jobs/{variant}_probe{step}'
        runner(['bash', probe / 'command.sh'], job, gpu=True)
        shutil.copyfile(job / 'logs/job.log', probe / 'logs/nohup.log')
        runner(audit_command(runtime, probe, commit, step), RUN_ROOT / f'queue_jobs/audit_{variant}_probe{step}')
    base.audit_resume(probe)
    jobs.write_json(probe / 'rollout_acceptance.json', {'passed': True, 'evidence': audit_rollouts(probe, [1, 2])})
    shared.verify_hashes(protected)
    validate_pair({v: base.read_json(RUN_ROOT / v / 'run_card.json') for v in VARIANTS}, commit)
    run = RUN_ROOT / variant
    runner(['bash', run / 'command.sh'], run, gpu=True, pid_name='train.pid', log_name='nohup.log')
    runner(audit_command(runtime, run, commit), RUN_ROOT / f'queue_jobs/{variant}_checkpoints')
    jobs.write_json(run / 'rollout_acceptance.json', {'passed': True, 'evidence': audit_rollouts(run, range(1, 201))})
    runner([base.PLOT_PYTHON, runtime / 'scripts/analyze_single_opd_diagnostics.py', '--run-dir', run,
            '--output-dir', run / 'figures', '--label', f'Qwen4 {variant} seed21'], RUN_ROOT / f'queue_jobs/{variant}_figures')


def evaluate_model(name, runtime, commit, runner, protected):
    shared.verify_hashes(protected)
    folder = RUN_ROOT / 'evaluations' / name
    folder.mkdir(parents=True, exist_ok=False)
    actor, model = model_paths(RUN_ROOT, name)
    if actor is not None:
        if model.exists():
            raise FileExistsError(model)
        runner([base.PYTHON, runtime / 'external/revisiting_opd/scripts/model_merger.py', 'merge',
                '--backend', 'fsdp', '--local_dir', actor, '--target_dir', model], RUN_ROOT / f'queue_jobs/merge_{name}')
    if not list(model.glob('*.safetensors')):
        raise ValueError('Missing evaluation model weights')
    hashes = {str(path): assets.sha256(path) for path in model.iterdir() if path.is_file()}
    jobs.write_json(folder / 'model_identity.json', {'model': str(model), 'sha256': hashes})
    jobs.write_json(folder / 'prompt_contract.json', historical.historical_prompt_contract(model))
    jobs.write_json(folder / 'eval_card.json', {**shared.EVAL_VALUES, 'model': str(model), 'role': name,
                    'source_commit': commit, 'tasks': shared.TASK_COUNTS, 'retain_rollouts': True,
                    'training_protocol': PROTOCOL, 'grader_sha256': grading.HISTORICAL_GRADER_SHA256})
    command = jobs.eval_command(runtime, base.PYTHON, model, base.DATA / 'eval_jsonl', folder / 'outputs', assets.STUDENT)
    command[command.index('--grader') + 1] = 'external'
    runner(command + ['--retain-rollouts'], folder, gpu=True, pid_name='eval.pid', log_name='eval.log')
    shared.verify_hashes(hashes)
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    accepted = shared.audit_evaluation(folder / 'outputs', base.DATA / 'eval_jsonl', model)
    accepted = historical.accept_archive(folder, accepted)
    print(json.dumps({'evaluated': name, 'per_task': accepted['per_task']}), flush=True)
    return accepted


def write_comparison(root, results):
    report = {'student': str(assets.STUDENT), 'teacher': str(assets.TEACHER), 'training_protocol': PROTOCOL,
              'intentional_train_eval_instruction_difference': True, 'per_benchmark': {}}
    lines = ['# Qwen3-4B-Base <- Qwen3-4B-Base-GRPO', '',
             '每题n8，历史grader，数值为百分比。初始学生评测在Block3后执行，权重仍为同一份原始Base。', '',
             '历史Block3包含joint PPO ratio和原block reduction；本实验不是只广播advantage的消融。', '']
    for task in shared.TASK_COUNTS:
        values = report['per_benchmark'][task] = {}
        lines += [f'## {task}', '', '| Model | Avg@8 | Pass@8 | 缺boxed (%) | 实际截断 (%) |',
                  '|---|---:|---:|---:|---:|']
        for name, result in results.items():
            value = result['per_task'][task]
            values[name] = value
            lines.append(f"| {name} | {100*value['avg_at_8']:.4f} | {100*value['pass_at_8']:.4f} | "
                         f"{100*value['format_error_rate']:.4f} | {100*value['engine_truncation_ratio']:.4f} |")
        for step in sorted(STEPS):
            left, right = (results[f'{v}_step{step}']['per_task'][task] for v in ('token_opd', 'block3_mean'))
            delta = {key: 100 * (right[key] - left[key]) for key in ('avg_at_8', 'pass_at_8')}
            values[f'delta_step{step}_pp'] = delta
            lines.append(f"| Block3 - Token Step{step} (pp) | {delta['avg_at_8']:+.4f} | {delta['pass_at_8']:+.4f} | | |")
        lines.append('')
    lines += ['单训练seed；固定seed的训练同题rollout可能重复，不能声称为独立采样。',
              '执行成功不代表方法有效。200为主终点，50/100/150用于训练曲线，不挑最佳checkpoint。', '']
    jobs.write_json(root / 'paired_comparison.json', report)
    (root / 'paired_comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime / 'DEPLOYED_COMMIT').read_text().strip()
    if runtime != ROOT / 'deployments' / commit:
        raise ValueError('Immutable ml2 deployment required')
    # Match the reviewed short-path budget that prevents Ray AF_UNIX failures.
    from recover_historical17_llama import check_socket_budget
    check_socket_budget(CACHE / 'train/tmp')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt; no automatic retry')
        jobs.write_json(RUN_ROOT / 'queue_manifest.json', {'source_commit': commit, 'created_at': jobs.now(),
            'student': str(assets.STUDENT), 'teacher': str(assets.TEACHER), 'predecessor': str(PREDECESSOR),
            'training_protocol': PROTOCOL, 'eval_enable_thinking': False, 'seed': 21, 'training_steps': 200,
            'save_steps': sorted(STEPS), 'eval_steps': STEPS, 'variants': VARIANTS,
            'order': 'block3_train_and_eval -> original_student_eval -> token_train_and_eval',
            'retain_rollouts': True, 'checkpoint_retention': 'all; no automatic deletion'})
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
                   PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
                   VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(base.GRADER))
        if env.get('RAY_TMPDIR'):
            raise ValueError('Unexpected RAY_TMPDIR override')
        for key, suffix in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                            'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'OUTLINES_CACHE_DIR': 'outlines'}.items():
            (CACHE / suffix).mkdir(parents=True, exist_ok=True)
            env[key] = str(CACHE / suffix)
        def runner(command, job, job_env=None, **kwargs):
            jobs.run_job(command, job, runtime, state, {**env, **(job_env or {})}, **kwargs)
        try:
            jobs.wait_for_idle()
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/runtime_hashes')
            jobs.write_json(state, {'status': 'preflight', 'updated_at': jobs.now()})
            protected = preflight()
            protected.update(prepare_pair(runtime, commit, runner))
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            results = execute_ordered(lambda name: evaluate_model(name, runtime, commit, runner, protected),
                                      lambda variant: train_model(variant, runtime, commit, runner, protected))
            historical.llama.compare_prompt_order(RUN_ROOT)
            for result in results.values():
                shared.verify_hashes(result['sha256'])
            shared.verify_hashes(protected)
            write_comparison(RUN_ROOT, results)
            jobs.write_json(RUN_ROOT / 'pair_acceptance.json', {'passed': True, 'models': results})
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                'evaluation_models': list(results), 'protected_inputs_verified': True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'error': str(error), 'updated_at': jobs.now()})
            raise


if __name__ == '__main__':
    main()

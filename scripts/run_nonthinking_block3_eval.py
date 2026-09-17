#!/usr/bin/env python3
"""Evaluate all authorized Block3 checkpoints against accepted Token results."""

import fcntl
import math
import os
from pathlib import Path
import signal
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_nonthinking_eval_block3 as shared

base, jobs = shared.base, shared.jobs
ROOT = shared.ROOT
SOURCE = shared.RUN_ROOT
BLOCK = SOURCE / 'block3_mean'
RUN_ROOT = ROOT / 'runs/20260917v2_qwen06_nonthinking_block3_eval_seed21_ml2'
CACHE = Path('/limx_embap/tos/q06/0917v2')
STEPS = (200, 100, 50)


def validate_result(result):
    if (result.get('passed') is not True
            or result.get('grader_sha256') != shared.grading.HISTORICAL_GRADER_SHA256
            or set(result.get('per_task', {})) != set(shared.TASK_COUNTS)):
        raise ValueError('Incomplete evaluation or incorrect grader')
    for task, count in shared.TASK_COUNTS.items():
        item = result['per_task'][task]
        if item.get('num_examples') != count or item.get('num_rollouts') != count * 8:
            raise ValueError(f'{task}: incomplete evaluation coverage')
        for name in ('avg_at_8', 'pass_at_8'):
            value = item.get(name, -1)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{task}: invalid {name}')


def execute_ordered(root, token, evaluate):
    if set(token) != set(STEPS):
        raise ValueError('Missing matched Token checkpoint')
    for result in token.values():
        validate_result(result)
    block = {}
    for step in STEPS:
        block[step] = evaluate(step)
        validate_result(block[step])
    report = {'protocol': shared.PROTOCOL, 'block3_run': str(BLOCK), 'token_run': str(shared.TOKEN),
              'grader_sha256': shared.grading.HISTORICAL_GRADER_SHA256, 'per_benchmark': {}}
    lines = ['# 0.6B Block3 mean与Token完整对照', '',
             '每题8次生成，沿用历史grader。各benchmark独立计分，差值为Block3减Token的百分点。', '']
    for task, count in shared.TASK_COUNTS.items():
        report['per_benchmark'][task] = {}
        lines += [f'## {task} ({count}题)', '',
                  '| Step | Token Avg@8 | Block3 Avg@8 | 差值pp | Token Pass@8 | Block3 Pass@8 | 差值pp |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for step in sorted(STEPS):
            left, right = token[step]['per_task'][task], block[step]['per_task'][task]
            avg = 100 * (right['avg_at_8'] - left['avg_at_8'])
            passed = 100 * (right['pass_at_8'] - left['pass_at_8'])
            report['per_benchmark'][task][str(step)] = {
                'token': left, 'block3': right, 'delta_avg_pp': avg, 'delta_pass_pp': passed}
            lines.append(f"| {step} | {100*left['avg_at_8']:.4f} | {100*right['avg_at_8']:.4f} | {avg:+.4f} | "
                         f"{100*left['pass_at_8']:.4f} | {100*right['pass_at_8']:.4f} | {passed:+.4f} |")
        lines.append('')
    lines += ['单训练seed的点估计，不构成跨seed显著性或因果机制证明。未按分数选择或跳过checkpoint。', '']
    jobs.write_json(root / 'block3_comparison.json', report)
    (root / 'block3_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    jobs.write_json(root / 'block3_eval_acceptance.json', {'passed': True, 'models': block})


def preflight():
    state = base.read_json(SOURCE / 'queue_state.json')
    accepted = base.read_json(BLOCK / 'acceptance.json')
    if (state.get('status') != 'complete' or state.get('block3_training_steps') != 200
            or state.get('protected_inputs_verified') is not True
            or accepted.get('passed') is not True or accepted.get('checkpoint_steps') != [50, 100, 200]
            or accepted.get('source_commit') != shared.TRAIN_COMMIT):
        raise ValueError('Block3 training is not complete and accepted')
    shared.validate_pair(base.read_json(shared.TOKEN / 'run_card.json'), base.read_json(BLOCK / 'run_card.json'))
    raw = base.read_json(BLOCK / 'rollout_acceptance.json')
    if raw.get('passed') is not True or [v['step'] for v in raw.get('evidence', [])] != list(range(1, 201)):
        raise ValueError('Missing full Block3 rollout acceptance')
    protected = base.read_json(SOURCE / 'protected_inputs.json')
    shared.verify_hashes(protected)
    files = [BLOCK / name for name in ('run_card.json', 'acceptance.json', 'rollout_acceptance.json',
                                       'artifact_hashes.sha256', 'logs/nohup.log')]
    files += [p for p in (BLOCK / 'checkpoints').rglob('*') if p.is_file()]
    protected.update({str(p): base.assets.sha256(p) for p in files})
    token = {}
    for step in STEPS:
        folder = SOURCE / 'evaluations' / f'token_step{step}'
        evidence = base.read_json(folder / 'acceptance.json')
        model = SOURCE / 'merged' / f'token_step{step}'
        if shared.audit_evaluation(folder / 'outputs', base.DATA / 'eval_jsonl', model) != evidence:
            raise ValueError('Stored Token acceptance differs from original predictions')
        shared.verify_hashes(evidence['sha256'])
        token[step] = evidence
        protected.update(evidence['sha256'])
        protected[str(folder / 'acceptance.json')] = base.assets.sha256(folder / 'acceptance.json')
    return protected, token


def main():
    release = Path(__file__).resolve().parents[1]
    commit = (release / 'DEPLOYED_COMMIT').read_text().strip()
    if release != ROOT / 'deployments' / commit:
        raise ValueError('An immutable ml2 controller release is required')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt requires review; no automatic restart')
        jobs.write_json(RUN_ROOT / 'queue_manifest.json', {'controller_commit': commit,
                        'eval_runtime': str(shared.TRAIN_RUNTIME), 'block3_run': str(BLOCK),
                        'token_reference': str(SOURCE), 'steps': STEPS, 'training_enabled': False,
                        'created_at': jobs.now()})
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
                   PYTHONPATH=f'{shared.TRAIN_RUNTIME}:{shared.TRAIN_RUNTIME}/external/revisiting_opd',
                   CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   TOKENIZERS_PARALLELISM='false', RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1', ENGINE='vllm', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   EVAL_GRADE_UTILS_PATH=str(base.GRADER))
        for key, suffix in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                            'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'OUTLINES_CACHE_DIR': 'outlines'}.items():
            (CACHE / suffix).mkdir(parents=True, exist_ok=True)
            env[key] = str(CACHE / suffix)
        def runner(command, job, cwd=shared.TRAIN_RUNTIME, **kwargs):
            jobs.run_job(command, job, cwd, state, env, **kwargs)
        try:
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/controller_hashes', cwd=release)
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/eval_runtime_hashes')
            protected, token = preflight()
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            jobs.write_json(RUN_ROOT / 'preflight.json', {'passed': True, 'token_steps': sorted(token),
                            'eval_values': shared.EVAL_VALUES, 'source': str(BLOCK)})
            execute_ordered(RUN_ROOT, token, lambda step: shared.evaluate_model(
                f'block3_step{step}', RUN_ROOT, runner, checkpoint_run=BLOCK, checkpoint_prefix='block3_step'))
            shared.verify_hashes(protected)
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                            'block3_eval_steps': sorted(STEPS), 'protected_inputs_verified': True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'error': str(error), 'updated_at': jobs.now()})
            raise


if __name__ == '__main__':
    main()

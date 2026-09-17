#!/usr/bin/env python3
"""Audit full non-thinking Token evaluations before launching its matched Block3 arm."""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import regrade_opd_eval_external as grading
import run_qwen06_nonthinking as nonthinking
import run_qwen06_pair as base
import run_window_queue as jobs

ROOT = base.ROOT
TRAIN_COMMIT = '0ce73aa42d7f734b9d34c379f474458bb6d48771'
TRAIN_RUNTIME = ROOT / 'deployments' / TRAIN_COMMIT
TOKEN = ROOT / 'runs/20260913v4_qwen06_nonthinking_token_seed21_ml2/token_opd'
RUN_ROOT = ROOT / 'runs/20260917v1_qwen06_nonthinking_eval_block3_seed21_ml2'
CACHE = Path('/limx_embap/tos/q06/0917v1')
PROTOCOL = 'math_eval_nonthinking_v1'
TASK_COUNTS = dict(grading.TASK_COUNTS)
EVAL_ORDER = ('token_step200', 'token_step100', 'token_step50', 'student_base', 'teacher')
EVAL_VALUES = {'n': 8, 'temperature': 1.0, 'top_p': .9, 'max_tokens': 16384,
               'eval_seed': 21, 'rollout_seeds': list(range(21, 29)),
               'grader': 'external', 'enable_thinking': False}


def block_env(root, probe_step=None):
    env = base.launcher_env(TRAIN_RUNTIME, TRAIN_COMMIT, root, 'block3_mean', probe_step)
    env.update({'PROJECT_NAME': 'opd_qwen06_nonthinking',
                'EXP_NAME': 'qwen06-nonthinking-block3-mean' + ('-probe' if probe_step else ''),
                'OPD_PROMPT_PROTOCOL': PROTOCOL,
                'LOSSLESS_ROLLOUT_DIR': str(root / 'block3_mean/rollouts'),
                'ROLLOUT_ATTEMPT_ID': f'probe{probe_step}' if probe_step else 'formal',
                'LOCAL_CACHE_ROOT': str(CACHE / 'train'),
                'BASELINE_ALIGNMENT': f'Matched non-thinking Token reference: {TOKEN}'})
    return env


def validate_pair(left, right):
    allowed = {'variant', 'opd_block_size', 'opd_block_advantage_mode', 'experiment_name',
               'diagnostic_output_dir', 'lossless_rollout_dir', 'baseline_alignment'}
    for key in (set(left) | set(right)) - allowed:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f'Unapproved paired difference: {key}')
    common = {**base.paired.EXPECTED_TRAINING_VALUES, 'source_commit': TRAIN_COMMIT,
              'student_model': str(base.assets.STUDENT), 'student_model_revision': base.assets.REVISION,
              'teacher_model': str(base.assets.TEACHER), 'opd_window_mode': 'fixed',
              'ppo_epochs': 1, 'opd_prompt_protocol': PROTOCOL}
    for card, name, size, mode in ((left, 'token_opd', 1, 'sum'), (right, 'block3_mean', 3, 'mean')):
        expected = {**common, 'variant': name, 'opd_block_size': size, 'opd_block_advantage_mode': mode}
        for key, value in expected.items():
            if card.get(key) != value:
                raise ValueError(f'{name}: {key} must equal {value!r}')


def audit_evaluation(output, data, model):
    config = base.read_json(output / 'eval_config.json')
    for key, value in EVAL_VALUES.items():
        if config.get(key) != value:
            raise ValueError(f'Evaluation {key} differs from fixed protocol')
    if (set(config.get('tasks', [])) != set(TASK_COUNTS)
            or config.get('gpus') != ['0', '1', '2', '3']
            or config.get('model_path') != str(model)
            or config.get('eval_jsonl_dir') != str(data)):
        raise ValueError('Evaluation model/data/task/GPU identity mismatch')
    summary = base.read_json(output / 'summary.json')
    if set(summary.get('tasks', {})) != set(TASK_COUNTS):
        raise ValueError('Incomplete task summary')
    result = {'passed': True, 'model': str(model), 'per_task': {}, 'sha256': {},
              'grader_sha256': grading.HISTORICAL_GRADER_SHA256}
    paths = [output / 'eval_config.json', output / 'summary.json']
    for task, count in TASK_COUNTS.items():
        raw = grading.raw_output_path(output, task, config)
        graded = output / f'{task}_graded.jsonl'
        rows = grading.load_jsonl(raw)
        groups = grading.validate_task_rows(rows, task, count, 0)
        scored = grading.load_jsonl(graded)
        grading.validate_task_rows(scored, task, count, 0)
        expected_rows = grading.load_jsonl(data / f'{task}.jsonl')
        expected = {str(x['id']): x for x in expected_rows}
        if len(expected_rows) != count or len(expected) != count or set(groups) != set(expected):
            raise ValueError(f'{task}: benchmark example identities differ')
        by_key = {(str(x['example_id']), x['rollout_id'], x['seed']): x for x in scored}
        successes = {example_id: [] for example_id in groups}
        for row in rows:
            example = expected[str(row['example_id'])]
            if any(row.get(k) != example[k] for k in ('problem', 'prompt', 'answer')):
                raise ValueError(f'{task}: changed problem/prompt/answer')
            if row.get('source') != example.get('source', task):
                raise ValueError(f'{task}: changed source')
            item = by_key[(str(row['example_id']), row['rollout_id'], row['seed'])]
            if type(item.get('correct')) is not bool or {k: v for k, v in item.items() if k != 'correct'} != row:
                raise ValueError(f'{task}: grading changed raw generation')
            successes[str(row['example_id'])].append(item['correct'])
        avg = sum(sum(scores) / 8 for scores in successes.values()) / count
        pass8 = sum(any(scores) for scores in successes.values()) / count
        item = summary['tasks'][task]
        if (item.get('num_examples') != count or item.get('total_rollouts') != count * 8
                or not math.isclose(item.get('avg_at_n', -1), avg, abs_tol=1e-12)
                or not math.isclose(item.get('pass_at_n', -1), pass8, abs_tol=1e-12)):
            raise ValueError(f'{task}: summary disagrees with complete graded rows')
        result['per_task'][task] = {'num_examples': count, 'num_rollouts': count * 8,
                                  'avg_at_8': avg, 'pass_at_8': pass8}
        paths.extend([raw, graded, data / f'{task}.jsonl'])
    result['sha256'] = {str(path): base.assets.sha256(path) for path in paths}
    return result


def execute_ordered(root, evaluate, train_block):
    results = {}
    for name in EVAL_ORDER:
        results[name] = evaluate(name)
        if results[name].get('passed') is not True:
            raise ValueError(f'Incomplete evaluation: {name}')
    report = {'protocol': PROTOCOL, 'token_run': str(TOKEN), 'per_benchmark': {},
              'grader_sha256': grading.HISTORICAL_GRADER_SHA256, 'eval_order': EVAL_ORDER}
    lines = ['# Qwen3-0.6B Non-Thinking Token OPD 完整评测', '',
             '每题8次生成，固定历史grader。以下按benchmark独立计分，不合并总分。', '']
    for task, count in TASK_COUNTS.items():
        initial = results['student_base']['per_task'][task]
        per_model = {}
        lines += [f'## {task} ({count}题)', '',
                  '| 模型 | Avg@8 (%) | Pass@8 (%) | 相对Base Avg差值 (pp) | 相对Base Pass差值 (pp) |',
                  '|---|---:|---:|---:|---:|']
        for name in ('student_base', 'teacher', 'token_step50', 'token_step100', 'token_step200'):
            item = dict(results[name]['per_task'][task])
            if item['num_examples'] != count:
                raise ValueError(f'{name}/{task}: incomplete result for report')
            item.update(delta_base_avg_pp=100 * (item['avg_at_8'] - initial['avg_at_8']),
                        delta_base_pass_pp=100 * (item['pass_at_8'] - initial['pass_at_8']))
            per_model[name] = item
            lines.append(f"| {name} | {100 * item['avg_at_8']:.4f} | {100 * item['pass_at_8']:.4f} | "
                         f"{item['delta_base_avg_pp']:+.4f} | {item['delta_base_pass_pp']:+.4f} |")
        report['per_benchmark'][task] = per_model
        lines.append('')
    lines += ['分数是单seed设置下的点估计，不代表多seed显著性；Block3尚需同协议对照。', '']
    jobs.write_json(root / 'token_effect.json', report)
    (root / 'token_effect.md').write_text('\n'.join(lines), encoding='utf-8')
    jobs.write_json(root / 'token_eval_acceptance.json', {'passed': True, 'models': results})
    train_block()


def verify_hashes(values):
    for name, sha in values.items():
        if base.assets.sha256(Path(name)) != sha:
            raise ValueError(f'Protected input changed: {name}')


def prompt_contract(model):
    from transformers import AutoTokenizer
    from opd_ext.math_protocol import DISABLED_SUFFIX, math_prompt, render_nonthinking

    student = AutoTokenizer.from_pretrained(str(base.assets.STUDENT), local_files_only=True)
    target = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    lengths, inputs = [], hashlib.sha256()
    for task, count in TASK_COUNTS.items():
        rows = grading.load_jsonl(base.DATA / 'eval_jsonl' / f'{task}.jsonl')
        if len(rows) != count:
            raise ValueError('Incomplete benchmark data')
        for row in rows:
            if math_prompt(row['problem']) != row['prompt']:
                raise ValueError('Train/eval content rule differs')
            text = render_nonthinking(target, row['prompt'])
            ids = target.encode(text, add_special_tokens=False)
            reference = student.encode(render_nonthinking(student, row['prompt']), add_special_tokens=False)
            # Do not normalize away the known >2048 training/evaluation input boundary.
            if not text.endswith(DISABLED_SUFFIX) or ids != reference or len(ids) > 2048:
                raise ValueError('Actual non-thinking evaluation input differs from training protocol')
            inputs.update(json.dumps([task, row['id'], ids]).encode())
            lengths.append(len(ids))
    return {'passed': True, 'enable_thinking': False, 'examples': len(lengths),
            'max_prompt_tokens': max(lengths), 'actual_input_sha256': inputs.hexdigest()}


def evaluate_model(name, root, runner, *, checkpoint_run=TOKEN, checkpoint_prefix='token_step'):
    folder = root / 'evaluations' / name
    folder.mkdir(parents=True, exist_ok=False)
    if name.startswith(checkpoint_prefix):
        step = int(name.removeprefix(checkpoint_prefix))
        actor = checkpoint_run / f'checkpoints/global_step_{step}/actor'
        model = root / 'merged' / name
        if model.exists():
            raise FileExistsError(f'Merged model already exists: {model}')
        runner([base.PYTHON, TRAIN_RUNTIME / 'external/revisiting_opd/scripts/model_merger.py',
                'merge', '--backend', 'fsdp', '--local_dir', actor, '--target_dir', model],
               root / 'queue_jobs' / f'merge_{name}')
    else:
        model = {'student_base': base.assets.STUDENT, 'teacher': base.assets.TEACHER}[name]
    weights = list(model.glob('*.safetensors'))
    if not weights:
        raise ValueError(f'No model weights: {model}')
    model_hashes = {str(path): base.assets.sha256(path) for path in sorted(model.iterdir()) if path.is_file()}
    jobs.write_json(folder / 'model_identity.json', {'model': str(model), 'sha256': model_hashes})
    jobs.write_json(folder / 'prompt_contract.json', prompt_contract(model))
    jobs.write_json(folder / 'eval_card.json', {**EVAL_VALUES, 'model': str(model), 'role': name,
                    'eval_runtime': str(TRAIN_RUNTIME), 'training_source_commit': TRAIN_COMMIT,
                    'tasks': TASK_COUNTS, 'grader_path': str(base.GRADER),
                    'grader_sha256': grading.HISTORICAL_GRADER_SHA256})
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    grade = grading.load_grader(base.GRADER)
    if not bool(grade(r'\boxed{2}', '2')) or bool(grade(r'\boxed{3}', '2')):
        raise ValueError('Pinned historical grader self-test failed')
    command = jobs.eval_command(TRAIN_RUNTIME, base.PYTHON, model, base.DATA / 'eval_jsonl',
                                folder / 'outputs', base.assets.STUDENT)
    command[command.index('--grader') + 1] = 'external'
    runner(command, folder, gpu=True, pid_name='eval.pid', log_name='eval.log')
    verify_hashes(model_hashes)
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    accepted = audit_evaluation(folder / 'outputs', base.DATA / 'eval_jsonl', model)
    jobs.write_json(folder / 'acceptance.json', accepted)
    print(json.dumps({'evaluated': name, 'per_task': accepted['per_task']}), flush=True)
    return accepted


def prepare_block(root, runner):
    runner(['bash', TRAIN_RUNTIME / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
           root / 'queue_jobs/prepare_formal', job_env=block_env(root))
    block = root / 'block3_mean'
    left, right = base.read_json(TOKEN / 'run_card.json'), base.read_json(block / 'run_card.json')
    validate_pair(left, right)
    if right['lossless_rollout_dir'] != str(block / 'rollouts') or right['diagnostic_output_dir'] != str(block / 'diagnostics'):
        raise ValueError('Incorrect new output destination')
    if set(base.paired.load_sha256_manifest(TOKEN / 'artifact_hashes.sha256')) != set(
            base.paired.load_sha256_manifest(block / 'artifact_hashes.sha256')):
        raise ValueError('Paired model/data file identities differ')
    jobs.write_json(root / 'paired_preflight.json', {'passed': True, 'token': left, 'block3': right,
                    'differences': {k: [left.get(k), right.get(k)] for k in set(left) | set(right)
                                    if left.get(k) != right.get(k)}})


def train_block(root, runner, protected):
    verify_hashes(protected)
    accepted = base.read_json(root / 'token_eval_acceptance.json')
    if accepted.get('passed') is not True or set(accepted.get('models', {})) != set(EVAL_ORDER):
        raise ValueError('Cannot train before full Token/reference evaluation')
    for evidence in accepted['models'].values():
        verify_hashes(evidence['sha256'])
    jobs.write_json(root / 'block_launch_gate.json', {'passed': True, 'at': jobs.now(),
                    'evaluation_models': EVAL_ORDER, 'training_source_commit': TRAIN_COMMIT})
    probe = root / 'probes/block3_mean'
    for step in (1, 2):
        runner(['bash', TRAIN_RUNTIME / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
               root / f'queue_jobs/prepare_probe{step}', job_env=block_env(root / 'probes', step))
        job = root / f'queue_jobs/probe{step}'
        runner(['bash', probe / 'command.sh'], job, gpu=True)
        shutil.copyfile(job / 'logs/job.log', probe / 'logs/nohup.log')
        runner(base.audit_command(TRAIN_RUNTIME, probe, TRAIN_COMMIT, probe_step=step),
               root / f'queue_jobs/audit_probe{step}')
    base.audit_resume(probe)
    evidence = nonthinking.audit_rollouts(probe, [1, 2])
    if any(row['generated_think_tags'] for row in evidence):
        raise ValueError('Block3 GPU probe generated think tags; manual review required')
    jobs.write_json(probe / 'rollout_acceptance.json', {'passed': True, 'evidence': evidence})
    block = root / 'block3_mean'
    validate_pair(base.read_json(TOKEN / 'run_card.json'), base.read_json(block / 'run_card.json'))
    runner(['bash', block / 'command.sh'], block, gpu=True, pid_name='train.pid', log_name='nohup.log')
    runner(base.audit_command(TRAIN_RUNTIME, block, TRAIN_COMMIT), root / 'queue_jobs/block_checkpoints')
    jobs.write_json(block / 'rollout_acceptance.json', {'passed': True,
                    'evidence': nonthinking.audit_rollouts(block, range(1, 201))})
    runner([base.PLOT_PYTHON, TRAIN_RUNTIME / 'scripts/analyze_single_opd_diagnostics.py',
            '--run-dir', block, '--output-dir', block / 'figures',
            '--label', 'Qwen06 non-thinking Block3 mean seed21'], root / 'queue_jobs/block_figures')
    verify_hashes(protected)


def main():
    analysis = Path(__file__).resolve().parents[1]
    commit = (analysis / 'DEPLOYED_COMMIT').read_text().strip()
    if analysis != ROOT / 'deployments' / commit:
        raise ValueError('Only an immutable ml2 controller release can launch this queue')
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
                        'training_source_commit': TRAIN_COMMIT, 'token_run': str(TOKEN),
                        'eval_order': EVAL_ORDER, 'block3_training_steps': 200,
                        'block3_eval_autostart': False, 'created_at': jobs.now()})
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
                   PYTHONPATH=f'{TRAIN_RUNTIME}:{TRAIN_RUNTIME}/external/revisiting_opd',
                   CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   TOKENIZERS_PARALLELISM='false', RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1', ENGINE='vllm', VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   EVAL_GRADE_UTILS_PATH=str(base.GRADER))
        for key, suffix in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                            'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'OUTLINES_CACHE_DIR': 'outlines'}.items():
            (CACHE / suffix).mkdir(parents=True, exist_ok=True)
            env[key] = str(CACHE / suffix)
        def runner(command, job, job_env=None, cwd=TRAIN_RUNTIME, **kwargs):
            jobs.run_job(command, job, cwd, state, {**env, **(job_env or {})}, **kwargs)
        try:
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/controller_hashes', cwd=analysis)
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/training_hashes')
            previous = base.read_json(TOKEN.parent / 'queue_state.json')
            if previous.get('status') != 'complete' or previous.get('training_steps') != 200:
                raise ValueError('The non-thinking Token predecessor is not complete')
            acceptance = base.read_json(TOKEN / 'acceptance.json')
            if (acceptance.get('passed') is not True or acceptance.get('checkpoint_steps') != [50, 100, 200]
                    or acceptance.get('source_commit') != TRAIN_COMMIT):
                raise ValueError('Token checkpoint acceptance is missing')
            protected = base.preflight_assets()
            files = [TOKEN / name for name in ('run_card.json', 'artifact_hashes.sha256', 'acceptance.json',
                                               'rollout_acceptance.json', 'logs/nohup.log')]
            files += [p for p in (TOKEN / 'checkpoints').rglob('*') if p.is_file()]
            protected.update({str(p): base.assets.sha256(p) for p in files})
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            prepare_block(RUN_ROOT, runner)
            execute_ordered(RUN_ROOT, lambda name: evaluate_model(name, RUN_ROOT, runner),
                            lambda: train_block(RUN_ROOT, runner, protected))
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                            'token_evaluations_complete': True, 'block3_training_steps': 200,
                            'block3_eval_started': False, 'protected_inputs_verified': True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'error': str(error), 'updated_at': jobs.now()})
            raise


if __name__ == '__main__':
    main()

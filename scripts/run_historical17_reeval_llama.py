#!/usr/bin/env python3
"""Re-evaluate six old Qwen checkpoints, then the explicitly approved Llama protocol."""

import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_llama32_pair as llama

base, jobs, shared, grading = llama.base, llama.jobs, llama.shared, llama.grading
ROOT = llama.ROOT
RUN_ROOT = ROOT / 'runs/20260918v2_historical17_reeval_llama_seed21_ml2'
CACHE = Path('/limx_embap/tos/h17/0918v2')
LEGACY_PROTOCOL = 'llama32_historical17_v1'
HISTORICAL_COMMIT = '9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977'
QWEN_INITIAL = Path('/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base')
HISTORICAL_RUNS = {
    'token_opd': ROOT / 'runs/20260712v1_token_opd_replication_seed21_ml2/token_opd',
    'block3_mean': ROOT / 'runs/20260711v2_block3_replication_seed21_ml2/block3_mean',
}


def historical_models():
    return {f'{variant}_step{step}': run / f'checkpoints/global_step_{step}/actor/huggingface'
            for step in (50, 100, 200) for variant, run in HISTORICAL_RUNS.items()}


def validate_historical_config(config, model):
    expected = {**shared.EVAL_VALUES, 'grader': 'verl', 'model_path': str(model),
                'eval_jsonl_dir': str(base.DATA / 'eval_jsonl'),
                'tasks': list(shared.TASK_COUNTS), 'gpus': ['0', '1', '2', '3']}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Historical evaluation {key} differs: {config.get(key)!r}')


def evaluation_command(runtime, model, output):
    command = jobs.eval_command(runtime, base.PYTHON, model, base.DATA / 'eval_jsonl', output, QWEN_INITIAL)
    command[command.index('--grader') + 1] = 'external'
    return command + ['--retain-rollouts']


def execute_ordered(evaluate, run_llama):
    results = {}
    for name, model in historical_models().items():
        result = evaluate(name, model)
        if result.get('passed') is not True:
            raise ValueError(f'Incomplete historical re-evaluation: {name}')
        results[name] = result
    run_llama()
    return results


def historical_prompt_contract(model):
    from transformers import AutoTokenizer
    from opd_ext.math_protocol import evaluation_inputs, evaluation_stop_ids, render_nonthinking
    reference = AutoTokenizer.from_pretrained(str(QWEN_INITIAL), local_files_only=True)
    target = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    if reference.get_vocab() != target.get_vocab() or reference.chat_template != target.chat_template:
        raise ValueError('Historical checkpoint tokenizer changed')
    digest = hashlib.sha256()
    count = 0
    for task, expected_count in shared.TASK_COUNTS.items():
        rows = grading.load_jsonl(base.DATA / 'eval_jsonl' / f'{task}.jsonl')
        if len(rows) != expected_count:
            raise ValueError('Incomplete historical benchmark')
        for row in rows:
            legacy = reference.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            actual = render_nonthinking(target, row['prompt'])
            if (actual != legacy or evaluation_inputs(target, [actual]) != [legacy]
                    or target.encode(actual) != reference.encode(legacy)):
                raise ValueError('Logging release changed historical Qwen prompt/input type')
            digest.update(json.dumps([task, row['id'], reference.encode(legacy)]).encode())
            count += 1
    if evaluation_stop_ids(target) != [151645, 151643]:
        raise ValueError('Historical Qwen stopping tokens changed')
    return {'passed': True, 'examples': count, 'enable_thinking': False,
            'rendered_input_sha256': digest.hexdigest(), 'generation_input_type': 'string',
            'stop_token_ids': [151645, 151643], 'historical_source_commit': HISTORICAL_COMMIT}


def preflight():
    old = llama.RUN_ROOT
    state = base.read_json(old / 'queue_state.json')
    if state.get('status') != 'failed' or state.get('error') != '143':
        raise ValueError('Old Llama attempt is not the explicitly stopped attempt')
    for file in (old / 'queue.pid', old / 'token_opd/train.pid'):
        pid = int(file.read_text())
        cmd = Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and cmd.read_bytes():
            raise ValueError(f'Old Llama process still exists: {pid}')
    if shutil.disk_usage(ROOT).free < 200_000_000_000:
        raise ValueError('Less than 200 GB free; no artifact pruning is permitted')
    protected = llama.assets.verify_assets()
    for path in QWEN_INITIAL.iterdir():
        if path.is_file() and path.suffix in ('.json', '.jinja', '.txt'):
            protected[str(path)] = llama.assets.sha256(path)
    previous = base.read_json(old / 'protected_inputs.json')
    data = [base.DATA / 'train.parquet', base.DATA / 'test.parquet']
    data += [base.DATA / 'eval_jsonl' / f'{task}.jsonl' for task in shared.TASK_COUNTS]
    for path in data:
        value = llama.assets.sha256(path)
        if previous.get(str(path)) != value:
            raise ValueError(f'Historical dataset changed: {path}')
        protected[str(path)] = value
    if protected[str(base.DATA / 'train.parquet')] != jobs.TRAIN_SHA:
        raise ValueError('Wrong DAPO pool')
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    protected[str(base.GRADER)] = grading.HISTORICAL_GRADER_SHA256
    grader = grading.load_grader(base.GRADER)
    if not grader(r'\boxed{2}', '2') or grader(r'\boxed{3}', '2'):
        raise ValueError('Historical grader self-test failed')
    evidence = {}
    for name, model in historical_models().items():
        variant, step = name.rsplit('_step', 1)
        run = HISTORICAL_RUNS[variant]
        config_path = run / f'eval_step_{step}_n8/outputs/eval_config.json'
        config = base.read_json(config_path)
        validate_historical_config(config, model)
        if base.read_json(run / 'run_card.json')['source_commit'] != HISTORICAL_COMMIT:
            raise ValueError('Unexpected historical training release')
        if not list(model.glob('*.safetensors')):
            raise ValueError(f'Missing historical weights: {model}')
        evidence[name] = {'config': config, 'prompt_contract': historical_prompt_contract(model)}
        paths = list(model.iterdir()) + [config_path, run / 'run_card.json']
        paths += list((run / f'eval_step_{step}_n8').rglob('*.json*'))
        for path in paths:
            if path.is_file() and path.stat().st_size:
                protected[str(path)] = llama.assets.sha256(path)
    # The interrupted attempt is a distinct experiment, not a warm start or disposable cache.
    for path in old.rglob('*'):
        if path.is_file() and (path.suffix in ('.pt', '.safetensors', '.json', '.gz') or path.name.endswith('.sha256')):
            protected[str(path)] = llama.assets.sha256(path)
    jobs.write_json(RUN_ROOT / 'historical_contracts.json', evidence)
    jobs.write_json(RUN_ROOT / 'stopped_llama.json', {'reason': 'user_requested_queue_reorder',
                    'state': state, 'run': str(old), 'resume_from_stopped_weights': False})
    packages = {}
    for package in ('torch', 'transformers', 'vllm', 'ray', 'numpy'):
        packages[package] = importlib.metadata.version(package)
    jobs.write_json(RUN_ROOT / 'environment.json', {'packages': packages, 'python': sys.version,
                    'historical_runtime': HISTORICAL_COMMIT,
                    'note': 'Equal decoding settings do not guarantee identical outputs across package/hardware changes.'})
    return protected


def accept_archive(folder, result):
    from opd_ext.eval_rollout_archive import audit_archive
    config = base.read_json(folder / 'outputs/eval_config.json')
    evidence = audit_archive(folder / 'outputs', config)
    for task in shared.TASK_COUNTS:
        raw = grading.load_jsonl(grading.raw_output_path(folder / 'outputs', task, config))
        result['per_task'][task].update(llama.native_eval_statistics(raw))
    result['rollout_archive'] = evidence
    result['sha256'].update(evidence['sha256'])
    jobs.write_json(folder / 'acceptance.json', result)
    return result


def evaluate_historical(name, model, runtime, commit, runner, protected):
    shared.verify_hashes(protected)
    folder = RUN_ROOT / 'qwen17/evaluations' / name
    folder.mkdir(parents=True, exist_ok=False)
    variant, step = name.rsplit('_step', 1)
    jobs.write_json(folder / 'eval_card.json', {**shared.EVAL_VALUES, 'source_commit': commit,
                    'historical_source_commit': HISTORICAL_COMMIT, 'model': str(model),
                    'historical_eval': str(HISTORICAL_RUNS[variant] / f'eval_step_{step}_n8'),
                    'grader_sha256': grading.HISTORICAL_GRADER_SHA256, 'retain_rollouts': True,
                    'prompt_contract': historical_prompt_contract(model)})
    runner(evaluation_command(runtime, model, folder / 'outputs'), folder,
           gpu=True, pid_name='eval.pid', log_name='eval.log')
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    result = shared.audit_evaluation(folder / 'outputs', base.DATA / 'eval_jsonl', model)
    result = accept_archive(folder, result)
    shared.verify_hashes(protected)
    print(json.dumps({'evaluated': name, 'per_benchmark': result['per_task']}), flush=True)
    return result


def write_qwen_comparison(results):
    report = {'per_benchmark': {}, 'historical_source_commit': HISTORICAL_COMMIT,
              'note': 'New generations, not recovered historical training rollouts.'}
    lines = ['# Qwen3-1.7B 六权重重新评测', '', '各 benchmark 独立计分，历史 grader，完整 n8。', '']
    for task in shared.TASK_COUNTS:
        values = {name: result['per_task'][task] for name, result in results.items()}
        lines += [f'## {task}', '', '| 模型 | Avg@8 (%) | Pass@8 (%) | 缺 boxed (%) | 实际截断 (%) |',
                  '|---|---:|---:|---:|---:|']
        for name, item in values.items():
            lines.append(f"| {name} | {100*item['avg_at_8']:.4f} | {100*item['pass_at_8']:.4f} | "
                         f"{100*item['format_error_rate']:.4f} | {100*item['engine_truncation_ratio']:.4f} |")
        for step in (50, 100, 200):
            left, right = (values[f'{v}_step{step}'] for v in llama.VARIANTS)
            delta = {key: 100 * (right[key] - left[key]) for key in ('avg_at_8', 'pass_at_8')}
            values[f'delta_step{step}_pp'] = delta
            lines.append(f"| Block3 - Token Step{step} (pp) | {delta['avg_at_8']:+.4f} | {delta['pass_at_8']:+.4f} | | |")
        report['per_benchmark'][task] = values
        lines.append('')
    jobs.write_json(RUN_ROOT / 'qwen17/comparison.json', report)
    (RUN_ROOT / 'qwen17/comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def run_llama(runtime, commit, runner, protected):
    root = RUN_ROOT / 'llama32'
    root.mkdir(exist_ok=False)
    shared.verify_hashes(protected)
    for role, spec in llama.assets.SPECS.items():
        gate = root / f'gpu_gate_{role}'
        runner([base.PYTHON, runtime / 'scripts/verify_nonthinking_gpu.py', '--model', spec['path'],
                '--protocol', LEGACY_PROTOCOL, '--cohort', llama.COHORT, '--eval-data', base.DATA / 'eval_jsonl',
                '--output', gate], RUN_ROOT / f'queue_jobs/llama_gpu_gate_{role}', gpu=True)
        if base.read_json(gate / 'summary.json').get('passed') is not True:
            raise ValueError('Llama historical training/evaluation prompt gate failed')
    options = {'run_root': root, 'training_protocol': LEGACY_PROTOCOL, 'cache': CACHE / 'llama'}
    llama.prepare_pair(runtime, commit, runner, **options)
    def evaluate(name):
        result = llama.evaluate_model(name, runtime, commit, runner, run_root=root, retain_rollouts=True)
        return accept_archive(root / 'evaluations' / name, result)
    results = llama.execute_ordered(evaluate, lambda variant: llama.train_model(
        variant, runtime, commit, runner, protected, **options))
    llama.compare_prompt_order(root)
    for result in results.values():
        shared.verify_hashes(result['sha256'])
    llama.write_comparison(root, results, training_protocol=LEGACY_PROTOCOL)
    jobs.write_json(root / 'pair_acceptance.json', {'passed': True, 'models': results,
                    'training_protocol': LEGACY_PROTOCOL, 'eval_protocol': llama.LLAMA_PROTOCOL,
                    'intentional_train_eval_instruction_difference': True})


def main():
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime / 'DEPLOYED_COMMIT').read_text().strip()
    if runtime != ROOT / 'deployments' / commit:
        raise ValueError('New immutable ml2 runtime required')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt; no automatic retry or overwrite')
        jobs.write_json(RUN_ROOT / 'queue_manifest.json', {'source_commit': commit, 'created_at': jobs.now(),
                        'qwen17_models': {k: str(v) for k, v in historical_models().items()},
                        'llama_training_protocol': LEGACY_PROTOCOL, 'llama_eval_protocol': llama.LLAMA_PROTOCOL,
                        'llama_initialization': str(llama.assets.STUDENT), 'teacher': str(llama.assets.TEACHER),
                        'seed': 21, 'stopped_attempt': str(llama.RUN_ROOT), 'retain_rollouts': True})
        (RUN_ROOT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:' + os.environ.get('PATH', ''),
                   PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
                   VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(base.GRADER))
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
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            results = {}
            def evaluate(name, model):
                result = evaluate_historical(name, model, runtime, commit, runner, protected)
                results[name] = result
                jobs.write_json(RUN_ROOT / 'qwen17/evaluation_progress.json', {'models': results})
                return result
            def dispatch_llama():
                write_qwen_comparison(results)
                jobs.write_json(RUN_ROOT / 'qwen17/acceptance.json', {'passed': True, 'models': results})
                run_llama(runtime, commit, runner, protected)
            execute_ordered(evaluate, dispatch_llama)
            for result in results.values():
                shared.verify_hashes(result['sha256'])
            shared.verify_hashes(protected)
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(),
                                    'qwen17_evaluated': list(results), 'llama_pair_complete': True,
                                    'protected_inputs_verified': True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'error': str(error), 'updated_at': jobs.now()})
            raise


if __name__ == '__main__':
    main()

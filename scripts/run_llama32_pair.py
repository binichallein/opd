#!/usr/bin/env python3
"""Wait for Qwen evaluation, then run the authorized matched Llama32 pair on ml2."""

import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import prepare_llama32_assets as assets
import run_nonthinking_block3_eval as predecessor
import run_nonthinking_eval_block3 as shared
from opd_ext.math_protocol import LLAMA_PROTOCOL, LLAMA_STOP_IDS

base, jobs, grading = shared.base, shared.jobs, shared.grading
ROOT = assets.ROOT
RUN_ROOT = ROOT / 'runs/20260918v1_llama32_1b_3b_nonthinking_seed21_ml2'
PREDECESSOR = predecessor.RUN_ROOT
CACHE = Path('/limx_embap/tos/l32/0918v1')
VARIANTS = ('token_opd', 'block3_mean')
STEPS = (200, 100, 50)
COHORT = ROOT / 'analyses/20260913_qwen06_truncation_probe_v2/prompts.jsonl'


def predecessor_ready(root):
    state = base.read_json(root / 'queue_state.json')
    if state.get('status') == 'running':
        return False
    if (state.get('status') != 'complete' or state.get('block3_eval_steps') != [50, 100, 200]
            or state.get('protected_inputs_verified') is not True):
        raise ValueError('Predecessor is failed/interrupted/incomplete; manual review required')
    accepted = base.read_json(root / 'block3_eval_acceptance.json')
    if accepted.get('passed') is not True or set(accepted.get('models', {})) != {'50', '100', '200'}:
        raise ValueError('Predecessor lacks full checkpoint evaluation acceptance')
    for step in STEPS:
        folder = root / 'evaluations' / f'block3_step{step}'
        if (folder / 'exit_code.txt').read_text().strip() != '0':
            raise ValueError('Predecessor evaluation did not exit successfully')
        result = accepted['models'][str(step)]
        predecessor.validate_result(result)
        if base.read_json(folder / 'acceptance.json') != result:
            raise ValueError('Predecessor acceptance records disagree')
    return True


def training_env(runtime, commit, root, variant, probe_step=None, *,
                 training_protocol=LLAMA_PROTOCOL, cache=CACHE):
    env = base.launcher_env(runtime, commit, root, variant, probe_step)
    env.update({'PROJECT_NAME': 'opd_llama32_pair', 'EXP_NAME': f'llama32-{variant}' + ('-probe' if probe_step else ''),
                'STUDENT_MODEL': str(assets.STUDENT), 'MATH_TEACHER': str(assets.TEACHER),
                'STUDENT_MODEL_REVISION': assets.SPECS['student']['revision'],
                'TEACHER_MODEL_REVISION': assets.SPECS['teacher']['revision'],
                'OPD_PROMPT_PROTOCOL': training_protocol, 'LOSSLESS_ROLLOUT_DIR': str(root / variant / 'rollouts'),
                'ROLLOUT_ATTEMPT_ID': f'probe{probe_step}' if probe_step else 'formal',
                'LOCAL_CACHE_ROOT': str(cache / 'train'),
                'BASELINE_ALIGNMENT': 'Matched Llama32 Instruct pair; ModelScope original BF16; seed21 original data and algorithm'})
    return env


def audit_command(runtime, run, commit, probe_step=None):
    command = list(map(str, base.audit_command(runtime, run, commit, probe_step=probe_step)))
    for flag, value in {'--expected-student-model-suffix': assets.STUDENT.name,
                        '--expected-teacher-model-suffix': assets.TEACHER.name,
                        '--expected-student-model-revision': assets.SPECS['student']['revision']}.items():
        command[command.index(flag) + 1] = value
    return command + ['--expected-teacher-model-revision', assets.SPECS['teacher']['revision']]


def validate_pair(cards, commit, *, training_protocol=LLAMA_PROTOCOL):
    if set(cards) != set(VARIANTS):
        raise ValueError('Unapproved training arms')
    left, right = (cards[v] for v in VARIANTS)
    allowed = {'variant', 'opd_block_size', 'opd_block_advantage_mode', 'experiment_name',
               'diagnostic_output_dir', 'lossless_rollout_dir'}
    for key in (set(left) | set(right)) - allowed:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError(f'Unapproved paired difference: {key}')
    common = {**base.paired.EXPECTED_TRAINING_VALUES, 'source_commit': commit,
              'student_model': str(assets.STUDENT), 'teacher_model': str(assets.TEACHER),
              'student_model_revision': assets.SPECS['student']['revision'],
              'teacher_model_revision': assets.SPECS['teacher']['revision'],
              'opd_prompt_protocol': training_protocol, 'opd_window_mode': 'fixed', 'ppo_epochs': 1,
              'resume_mode': 'disable', 'resume_from_path': '', 'rollout_attempt_id': 'formal',
              'expected_train_sha256': jobs.TRAIN_SHA}
    for variant, size, mode in (('token_opd', 1, 'sum'), ('block3_mean', 3, 'mean')):
        for key, value in {**common, 'variant': variant, 'opd_block_size': size,
                           'opd_block_advantage_mode': mode}.items():
            if cards[variant].get(key) != value:
                raise ValueError(f'{variant}: {key} must be {value!r}')


def execute_ordered(evaluate, train):
    results = {}
    def accepted(name):
        result = evaluate(name)
        if result.get('passed') is not True:
            raise ValueError(f'Incomplete evaluation: {name}')
        results[name] = result
    accepted('student_base')
    for variant in VARIANTS:
        train(variant)
        for step in STEPS:
            accepted(f'{variant}_step{step}')
    return results


def prompt_contract(model):
    from transformers import AutoTokenizer
    from opd_ext.math_protocol import math_prompt, render_nonthinking, valid_control_prefix, evaluation_inputs
    reference = AutoTokenizer.from_pretrained(str(assets.STUDENT), local_files_only=True)
    candidate = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    if candidate.get_vocab() != reference.get_vocab() or candidate.chat_template != reference.chat_template:
        raise ValueError('Changed model token mapping/chat template')
    digest = hashlib.sha256()
    maximum = 0
    for task, count in shared.TASK_COUNTS.items():
        rows = grading.load_jsonl(base.DATA / 'eval_jsonl' / f'{task}.jsonl')
        if len(rows) != count:
            raise ValueError('Incomplete benchmark data')
        for row in rows:
            if row['prompt'] != math_prompt(row['problem']):
                raise ValueError('Math instruction differs between training/evaluation')
            text = render_nonthinking(candidate, row['prompt'])
            ids = candidate.encode(text, add_special_tokens=False)
            ref = reference.encode(render_nonthinking(reference, row['prompt']), add_special_tokens=False)
            if (not valid_control_prefix(text, candidate, LLAMA_PROTOCOL) or ids != ref
                    or evaluation_inputs(candidate, [text])[0]['prompt_token_ids'] != ids or len(ids) > 2048):
                raise ValueError('Actual Llama evaluation prompt does not match training')
            digest.update(json.dumps([task, row['id'], ids]).encode())
            maximum = max(maximum, len(ids))
    return {'passed': True, 'protocol': LLAMA_PROTOCOL, 'enable_thinking': False, 'examples': 643,
            'max_prompt_tokens': maximum, 'actual_input_sha256': digest.hexdigest(), 'stop_token_ids': LLAMA_STOP_IDS}


def preflight():
    if shutil.disk_usage(ROOT).free < 150_000_000_000:
        raise ValueError('Less than 150 GB free; preserve old artifacts and stop')
    protected = assets.verify_assets()
    previous = base.read_json(PREDECESSOR / 'protected_inputs.json')
    paths = [base.DATA / 'train.parquet', base.DATA / 'test.parquet']
    paths += [base.DATA / 'eval_jsonl' / f'{task}.jsonl' for task in shared.TASK_COUNTS]
    for path in paths:
        digest = assets.sha256(path)
        if previous.get(str(path)) != digest:
            raise ValueError(f'Changed historical data bytes: {path}')
        protected[str(path)] = digest
    if protected[str(base.DATA / 'train.parquet')] != jobs.TRAIN_SHA:
        raise ValueError('Wrong frozen DAPO data')
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    protected[str(base.GRADER)] = grading.HISTORICAL_GRADER_SHA256
    grade = grading.load_grader(base.GRADER)
    if not grade(r'\boxed{2}', '2') or grade(r'\boxed{3}', '2'):
        raise ValueError('Historical grader self-test failed')
    for step in STEPS:
        folder = PREDECESSOR / 'evaluations' / f'block3_step{step}'
        accepted = base.read_json(folder / 'acceptance.json')
        if shared.audit_evaluation(folder / 'outputs', base.DATA / 'eval_jsonl',
                                   PREDECESSOR / 'merged' / f'block3_step{step}') != accepted:
            raise ValueError('Predecessor raw/graded evaluation differs from acceptance')
        shared.verify_hashes(accepted['sha256'])
        protected.update(accepted['sha256'])
        protected[str(folder / 'acceptance.json')] = assets.sha256(folder / 'acceptance.json')
    return protected


def prepare_pair(runtime, commit, runner, *, run_root=RUN_ROOT,
                 training_protocol=LLAMA_PROTOCOL, cache=CACHE):
    for variant in VARIANTS:
        runner(['bash', runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
               run_root / f'queue_jobs/prepare_{variant}',
               job_env=training_env(runtime, commit, run_root, variant,
                                    training_protocol=training_protocol, cache=cache))
    cards = {v: base.read_json(run_root / v / 'run_card.json') for v in VARIANTS}
    validate_pair(cards, commit, training_protocol=training_protocol)
    manifests = [base.paired.load_sha256_manifest(run_root / v / 'artifact_hashes.sha256') for v in VARIANTS]
    if set(manifests[0]) != set(manifests[1]):
        raise ValueError('Paired data/model hash manifests differ')
    jobs.write_json(run_root / 'paired_preflight.json', {'passed': True, 'cards': cards})


def audit_rollouts(run, steps, *, training_protocol=LLAMA_PROTOCOL):
    return shared.nonthinking.audit_rollouts(run, steps, student=assets.STUDENT,
                                            protocol=training_protocol, stop_ids=LLAMA_STOP_IDS)


def train_model(variant, runtime, commit, runner, protected, *, run_root=RUN_ROOT,
                training_protocol=LLAMA_PROTOCOL, cache=CACHE):
    shared.verify_hashes(protected)
    probe = run_root / 'probes' / variant
    for step in (1, 2):
        runner(['bash', runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
               run_root / f'queue_jobs/prepare_{variant}_probe{step}',
               job_env=training_env(runtime, commit, run_root / 'probes', variant, step,
                                    training_protocol=training_protocol, cache=cache))
        job = run_root / f'queue_jobs/{variant}_probe{step}'
        runner(['bash', probe / 'command.sh'], job, gpu=True)
        shutil.copyfile(job / 'logs/job.log', probe / 'logs/nohup.log')
        runner(audit_command(runtime, probe, commit, step), run_root / f'queue_jobs/audit_{variant}_probe{step}')
    base.audit_resume(probe)
    evidence = audit_rollouts(probe, [1, 2], training_protocol=training_protocol)
    if training_protocol == LLAMA_PROTOCOL and any(item['generated_think_tags'] for item in evidence):
        raise ValueError('Llama training probe generated think tags; manual review required')
    jobs.write_json(probe / 'rollout_acceptance.json', {'passed': True, 'evidence': evidence})
    validate_pair({v: base.read_json(run_root / v / 'run_card.json') for v in VARIANTS}, commit,
                  training_protocol=training_protocol)
    run = run_root / variant
    runner(['bash', run / 'command.sh'], run, gpu=True, pid_name='train.pid', log_name='nohup.log')
    runner(audit_command(runtime, run, commit), run_root / f'queue_jobs/{variant}_checkpoints')
    jobs.write_json(run / 'rollout_acceptance.json', {
        'passed': True, 'evidence': audit_rollouts(run, range(1, 201), training_protocol=training_protocol)})
    runner([base.PLOT_PYTHON, runtime / 'scripts/analyze_single_opd_diagnostics.py', '--run-dir', run,
            '--output-dir', run / 'figures', '--label', f'Llama32 {variant} seed21'],
           run_root / f'queue_jobs/{variant}_figures')


def evaluate_model(name, runtime, commit, runner, *, run_root=RUN_ROOT, retain_rollouts=False):
    folder = run_root / 'evaluations' / name
    folder.mkdir(parents=True, exist_ok=False)
    if name == 'student_base':
        model = assets.STUDENT
    else:
        variant, step = name.rsplit('_step', 1)
        if variant not in VARIANTS or int(step) not in STEPS:
            raise ValueError('Unapproved model evaluation')
        actor = run_root / variant / f'checkpoints/global_step_{step}/actor'
        model = run_root / 'merged' / name
        if model.exists():
            raise FileExistsError(model)
        runner([base.PYTHON, runtime / 'external/revisiting_opd/scripts/model_merger.py', 'merge',
                '--backend', 'fsdp', '--local_dir', actor, '--target_dir', model],
               run_root / f'queue_jobs/merge_{name}')
    if not list(model.glob('*.safetensors')):
        raise ValueError('No model weights')
    hashes = {str(path): assets.sha256(path) for path in model.iterdir() if path.is_file()}
    jobs.write_json(folder / 'model_identity.json', {'model': str(model), 'sha256': hashes})
    jobs.write_json(folder / 'prompt_contract.json', prompt_contract(model))
    jobs.write_json(folder / 'eval_card.json', {**shared.EVAL_VALUES, 'model': str(model), 'role': name,
                    'source_commit': commit, 'protocol': LLAMA_PROTOCOL, 'stop_token_ids': LLAMA_STOP_IDS,
                    'tasks': shared.TASK_COUNTS, 'grader_sha256': grading.HISTORICAL_GRADER_SHA256})
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    command = jobs.eval_command(runtime, base.PYTHON, model, base.DATA / 'eval_jsonl', folder / 'outputs', assets.STUDENT)
    command[command.index('--grader') + 1] = 'external'
    if retain_rollouts:
        command.append('--retain-rollouts')
    runner(command, folder, gpu=True, pid_name='eval.pid', log_name='eval.log')
    shared.verify_hashes(hashes)
    grading.validate_grader_hash(base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    accepted = shared.audit_evaluation(folder / 'outputs', base.DATA / 'eval_jsonl', model)
    config = base.read_json(folder / 'outputs/eval_config.json')
    summary = base.read_json(folder / 'outputs/summary.json')
    for task in shared.TASK_COUNTS:
        rows = grading.load_jsonl(grading.raw_output_path(folder / 'outputs', task, config))
        statistics = native_eval_statistics(rows)
        if (summary['tasks'][task]['format_error_rollouts'] != statistics['format_error_rollouts']
                or summary['tasks'][task]['engine_length_stop_rollouts'] != statistics['engine_length_stop_rollouts']):
            raise ValueError('Evaluation format/truncation summary differs from raw generation')
        accepted['per_task'][task].update(statistics)
    jobs.write_json(folder / 'acceptance.json', accepted)
    print(json.dumps({'evaluated': name, 'per_task': accepted['per_task']}), flush=True)
    return accepted


def native_eval_statistics(rows):
    if not rows or any(r.get('num_generated_tokens') != len(r.get('response_token_ids', []))
                       or r.get('finish_reason') not in ('length', 'stop')
                       or not 0 <= r['num_generated_tokens'] <= 16384 for r in rows):
        raise ValueError('Incomplete native evaluation generation metadata')
    formatting = sum('\\boxed' not in r['response'] for r in rows)
    truncated = sum(r['finish_reason'] == 'length' for r in rows)
    return {'format_error_rollouts': formatting, 'format_error_rate': formatting / len(rows),
            'engine_length_stop_rollouts': truncated, 'engine_truncation_ratio': truncated / len(rows),
            'mean_generated_tokens': sum(r['num_generated_tokens'] for r in rows) / len(rows)}


def compare_prompt_order(root):
    for step in range(1, 201):
        values = []
        for variant in VARIANTS:
            with gzip.open(root / variant / f'rollouts/formal/step_{step:06d}/raw.jsonl.gz', 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            values.append([(r['source_extra_info'], r['prompt_token_ids'], r['sampling']) for r in rows])
        if values[0] != values[1]:
            raise ValueError(f'Paired actual prompt/order/sampling mismatch at step {step}')
    jobs.write_json(root / 'prompt_order_acceptance.json', {'passed': True, 'steps': 200, 'trajectories_per_arm': 6400})


def write_comparison(root, results, *, training_protocol=LLAMA_PROTOCOL):
    lines = ['# Llama 3.2 1B-Instruct <- 3B-Instruct', '',
             '完整 n8、历史 grader，各 benchmark 独立计分。Base为零步Instruct学生。数值为百分比。', '']
    report = {'protocol': LLAMA_PROTOCOL, 'training_protocol': training_protocol,
              'eval_protocol': LLAMA_PROTOCOL,
              'intentional_train_eval_instruction_difference': training_protocol != LLAMA_PROTOCOL,
              'per_benchmark': {}}
    if training_protocol != LLAMA_PROTOCOL:
        lines += ['本次按用户要求复刻历史提示差异：训练要求 think 标签，评测不要求；'
                  '不是训推提示一致的非 thinking 实验。', '']
    for task in shared.TASK_COUNTS:
        report['per_benchmark'][task] = {}
        lines += [f'## {task}', '', '| Model | Avg@8 | Pass@8 | 缺boxed (%) | 实际截断 (%) |', '|---|---:|---:|---:|---:|']
        for name, result in results.items():
            value = result['per_task'][task]
            report['per_benchmark'][task][name] = value
            lines.append(f"| {name} | {100*value['avg_at_8']:.4f} | {100*value['pass_at_8']:.4f} | "
                         f"{100*value['format_error_rate']:.4f} | {100*value['engine_truncation_ratio']:.4f} |")
        for step in sorted(STEPS):
            left, right = (results[f'{v}_step{step}']['per_task'][task] for v in VARIANTS)
            delta = {metric: 100 * (right[metric] - left[metric]) for metric in ('avg_at_8', 'pass_at_8')}
            report['per_benchmark'][task][f'delta_step{step}_pp'] = delta
            lines.append(f"| Block3 - Token Step{step} (pp) | {delta['avg_at_8']:+.4f} | {delta['pass_at_8']:+.4f} | | |")
        lines.append('')
    lines += ['单训练seed的对照，不构成跨seed显著性结论；不得将执行验收当作方法有效性证明。', '']
    jobs.write_json(root / 'paired_comparison.json', report)
    (root / 'paired_comparison.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime / 'DEPLOYED_COMMIT').read_text().strip()
    if runtime != ROOT / 'deployments' / commit:
        raise ValueError('Immutable ml2 runtime required')
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT / 'queue_state.json'
    with (RUN_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT / 'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt requires manual review; never auto-retry')
        jobs.write_json(RUN_ROOT / 'queue_manifest.json', {'source_commit': commit, 'predecessor': str(PREDECESSOR),
                        'protocol': LLAMA_PROTOCOL, 'variants': VARIANTS, 'steps': STEPS, 'seed': 21,
                        'student': str(assets.STUDENT), 'teacher': str(assets.TEACHER), 'created_at': jobs.now()})
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
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT / 'queue_jobs/runtime_hashes')
            while not predecessor_ready(PREDECESSOR):
                pid = int((PREDECESSOR / 'queue.pid').read_text())
                os.kill(pid, 0)
                if 'run_nonthinking_block3_eval.py' not in Path(f'/proc/{pid}/cmdline').read_text():
                    raise ValueError('Predecessor PID is no longer its controller')
                jobs.write_json(state, {'status': 'waiting_predecessor', 'predecessor': str(PREDECESSOR),
                                       'predecessor_pid': pid, 'updated_at': jobs.now()})
                print(f'{jobs.now()} waiting for all predecessor evaluations; no GPU work', flush=True)
                time.sleep(120)
            jobs.wait_for_idle()
            protected = preflight()
            jobs.write_json(RUN_ROOT / 'protected_inputs.json', protected)
            for role, spec in assets.SPECS.items():
                jobs.write_json(RUN_ROOT / f'{role}_prompt_contract.json', prompt_contract(spec['path']))
                runner([base.PYTHON, runtime / 'scripts/verify_nonthinking_gpu.py', '--model', spec['path'],
                        '--protocol', LLAMA_PROTOCOL, '--cohort', COHORT, '--eval-data', base.DATA / 'eval_jsonl',
                        '--output', RUN_ROOT / f'gpu_gate_{role}'], RUN_ROOT / f'queue_jobs/gpu_gate_{role}', gpu=True)
                if base.read_json(RUN_ROOT / f'gpu_gate_{role}/summary.json').get('passed') is not True:
                    raise ValueError(f'{role} GPU prompt/output gate failed')
            prepare_pair(runtime, commit, runner)
            results = execute_ordered(lambda name: evaluate_model(name, runtime, commit, runner),
                lambda variant: train_model(variant, runtime, commit, runner, protected))
            compare_prompt_order(RUN_ROOT)
            for result in results.values():
                shared.verify_hashes(result['sha256'])
            write_comparison(RUN_ROOT, results)
            shared.verify_hashes(protected)
            jobs.write_json(RUN_ROOT / 'pair_acceptance.json', {'passed': True, 'models': results})
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(), 'variants': VARIANTS,
                                   'evaluation_models': list(results), 'protected_inputs_verified': True})
        except BaseException as error:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'error': str(error), 'updated_at': jobs.now()})
            raise


if __name__ == '__main__':
    main()

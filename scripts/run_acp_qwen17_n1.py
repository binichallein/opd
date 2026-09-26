#!/usr/bin/env python3
"""AFS-backed ACP adapter for the approved Qwen8 -> Qwen1.7 100-step pair."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_qwen17_instruct_pair as q
import run_historical_pair_n1 as n1

ROOT = Path('/mnt/afs/202609/tyf-qwen-opd')
HOST = 'pt-5a50d1567d77437c94035024288a327f-worker-0'
RUN_ROOT = ROOT / 'runs/20260926v1_qwen8_to17_instruct_n1_step100_seed21_acp'
CACHE = Path('/dev/shm/a17')
VENV = ROOT / 'envs/verl-cu128-v1'
PROTOCOL = 'qwen3_native_chat_no_thinking_boxed_v1'
VARIANTS = ('block3_mean', 'token_opd')
STEPS = (100, 75, 50, 25)
SAVE_STEPS = '25,50,75,100'
OLD_ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
REFERENCE = ROOT / 'deployments/7bf5420d69d5e3ce23cb79882a0ceb6c391b1cf5'
execute_ordered = n1.execute_ordered
ENTRYPOINT = Path(__file__).resolve()
LOSS_OVERRIDES = {}
ORDERING = 'Block3 probe/train/eval100,75,50,25 -> Token probe/train/eval100,75,50,25'
write_comparison = n1.write_comparison
HARDWARE = '4xH100-80GB; 32 cgroup CPUs'
LIFETIME = 'Detached from SSH; not immune to ACP scheduling/runtime expiry'
DISPLAY_PREFIX = 'ACP Qwen1.7 Instruct n1'


def validate_location(host, root, fstype):
    if host != HOST or root != ROOT or fstype != 'fuse.quarkfs_client':
        raise ValueError('Only the verified ACP worker and persistent AFS root are authorized')


def configure():
    q.ROOT = q.assets.ROOT = ROOT
    for key in ('instruct_student', 'instruct_teacher'):
        spec = q.assets.MODELS[key]
        spec['path'] = ROOT / 'models' / spec['repo'].split('/')[1]
    q.STUDENT = q.assets.MODELS['instruct_student']['path']
    q.TEACHER = q.assets.MODELS['instruct_teacher']['path']
    q.STEPS, q.SAVE_STEPS = STEPS, SAVE_STEPS
    q.base.VENV, q.base.PYTHON = VENV, VENV / 'bin/python'
    q.base.PLOT_PYTHON = q.base.PYTHON
    q.base.DATA = ROOT / 'data/math_opd_dapo17k_hf_full_eval4'
    q.base.GRADER = ROOT / 'runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/grading/historical_utils_sha04f7.py'
    q.QUALIFICATION = ROOT / 'runs/20260921v1_qwen8_to17_diagnostics_ml2/qualification/instruct'
    q.LOSS_REFERENCE = REFERENCE


def training_env(runtime, commit, root, variant, probe=None):
    if variant not in VARIANTS or probe not in (None, 1, 2):
        raise ValueError('Only the approved paired methods and save1/resume2 probes are allowed')
    env = q.base.launcher_env(runtime, commit, root, variant, probe)
    env.update(REMOTE='acp', VENV=str(VENV), PROJECT_NAME='opd_acp_qwen17_n1',
        EXP_NAME=f'acp-qwen17-{variant}-n1', STUDENT_MODEL=str(ROOT / 'models/Qwen3-1.7B'),
        MATH_TEACHER=str(ROOT / 'models/Qwen3-8B'), STUDENT_MODEL_REVISION=q.STUDENT_REVISION,
        TEACHER_MODEL_REVISION=q.TEACHER_REVISION, HF_HOME_DIR=str(ROOT / 'cache/hf'),
        DATA_DIR=str(ROOT / 'data/math_opd_dapo17k_hf_full_eval4'),
        TRAIN_DATA=str(ROOT / 'data/math_opd_dapo17k_hf_full_eval4/train.parquet'),
        VAL_DATA=str(ROOT / 'data/math_opd_dapo17k_hf_full_eval4/test.parquet'),
        OPD_PROMPT_PROTOCOL=PROTOCOL, OPD_REQUEST_SEED_RULE='legacy', RAY_NUM_CPUS='32',
        TRAIN_BATCH_SIZE='32', ROLLOUT_GROUP_SIZE='1', TOTAL_TRAINING_STEPS='100',
        STOP_AFTER_STEP=str(probe or -1), OPD_DIAG_INTERVAL='1' if probe else '5',
        DIAGNOSTIC_SAVE_STEPS={None:SAVE_STEPS, 1:'1', 2:'1,2'}[probe],
        LOSSLESS_ROLLOUT_DIR=str(root / variant / 'rollouts'),
        ROLLOUT_ATTEMPT_ID=f'probe{probe}' if probe else 'formal',
        LOCAL_CACHE_ROOT=str(CACHE / variant / 'train'),
        BASELINE_ALIGNMENT='ACP H100 matched 100-step 32x1 pair; native nonthinking instruct prompt; legacy losses and request seeds')
    return env


def expected_card(runtime, commit, root, variant, probe=None):
    card = q.expected_card(root, variant, commit, probe)
    card.update(venv=str(VENV), project_name='opd_acp_qwen17_n1',
        experiment_name=f'acp-qwen17-{variant}-n1', ray_num_cpus=32, train_batch_size=32,
        rollout_group_size=1, total_training_steps=100, request_seed_rule='legacy',
        opd_diag_interval=1 if probe else 5, diagnostic_save_steps={None:SAVE_STEPS, 1:'1', 2:'1,2'}[probe],
        stop_after_step=probe or -1, opd_block_ablation='legacy')
    return card


def require_equal_card(expected, actual):
    differences = {k: [v, actual.get(k)] for k, v in expected.items() if actual.get(k) != v}
    if differences:
        raise ValueError(f'Unapproved training configuration drift: {differences}')


def validate_rollout(row, ids, step):
    count = row['response_length']
    sampling = dict(temperature=1., top_p=.9, top_k=-1, max_tokens=16384, n=1,
                    ignore_eos=False, stop_token_ids=[151645, 151643], seed=21)
    if (row['protocol'] != PROTOCOL or row['enable_thinking'] is not False
            or row['step'] != step or row['prompt_token_ids'] != ids or row['eos_token_id'] != 151645
            or any(row['sampling'].get(k) != v for k, v in sampling.items())
            or count != len(row['response_token_ids']) or not 0 < count <= 16384
            or row['response_tensor_width'] != 16384 or row['padding_length'] != 16384-count
            or row['response_mask'] != [1]*count or len(row['rollout_log_probs']) != count
            or not all(math.isfinite(v) for v in row['rollout_log_probs'])):
        raise ValueError('Actual nonthinking prompt, sampling, mask or logprob differs')
    q.qualify.validate_stops([row])


def audit_rollouts(folder, plan, steps, probe=False):
    from transformers import AutoTokenizer
    from diagnose_token_truncation import analyze_tokens
    tokenizer = AutoTokenizer.from_pretrained(q.STUDENT, local_files_only=True)
    records = []
    for step in steps:
        files = list((folder / 'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(files) != 1:
            raise ValueError(f'Missing or duplicate trajectory archive at {step}')
        rows = q.read_archive(files[0])
        fingerprint = n1.batch_fingerprint(rows, plan['sources'][32*(step-1):32*step], step)
        for row in rows:
            validate_rollout(row, q.prompt_input_ids(tokenizer, row['source_extra_info']['question']), step)
            if probe and row['generated_think_tags']:
                raise ValueError('Generated thinking tag in GPU probe; stop before formal training')
        records.append(dict(step=step, count=32, paired_input_sha256=fingerprint,
            sha256=q.assets.sha256(files[0]), length_stops=sum(r['finish_reason']=='length' for r in rows),
            generated_think_tags=sum(r['generated_think_tags'] for r in rows),
            periodic_tails=sum(analyze_tokens(r['response_token_ids'], 16384)['tail_period'] is not None for r in rows)))
    return dict(passed=True, steps=records, total_rollouts=32*len(records))


def preflight(runtime):
    protected = q.common_preflight(runtime, LOSS_OVERRIDES)
    for key in ('instruct_student', 'instruct_teacher'):
        protected.update(q.assets.ensure_asset(key))
    gate = q.qualify.old.read_sealed(q.QUALIFICATION / 'gate_acceptance.json')
    q.require_capability(gate, q.assets.sha256(q.QUALIFICATION / 'gate_acceptance.json'))
    evidence = {str(ROOT / Path(path).relative_to(OLD_ROOT)): digest for path, digest in gate['evidence'].items()}
    q.shared.verify_hashes(evidence)
    protected.update(evidence)
    protected[str(q.QUALIFICATION / 'gate_acceptance.json')] = q.CAPABILITY_SHA
    return protected


def grading_gate():
    grade, extract = q.qualify.old.load_historical_grader(q.base.GRADER)
    cells = {}
    for cell in ('smoke', 'direct', 'continuation'):
        for role in ('student', 'teacher'):
            folder = q.QUALIFICATION / cell / role
            rows = q.shared.grading.load_jsonl(folder / 'raw.jsonl')
            expected = q.shared.grading.load_jsonl(folder / 'results.jsonl')
            if not rows or q.qualify.old.score_records(rows, grade, extract) != expected:
                raise ValueError(f'Historical grading differs on {cell}/{role}')
            cells[f'{cell}/{role}'] = len(rows)
    return dict(passed=True, cells=cells, num_records=sum(cells.values()))


def gpu_smoke(root, label):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model = q.STUDENT if label == 'student' else q.TEACHER
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    selected = q.base.read_json(q.QUALIFICATION / 'selected.json')[:4]
    prompts = [q.prompt_input_ids(tokenizer, r['question']) for r in selected]
    folder = root / 'gpu_smoke' / label
    folder.mkdir(parents=True, exist_ok=False)
    llm = LLM(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
              gpu_memory_utilization=.6, max_model_len=18432, seed=21)
    sampling = SamplingParams(n=1, temperature=1., top_p=.9, top_k=-1, max_tokens=256,
                              seed=21, stop_token_ids=[151645,151643])
    outputs = llm.generate([{'prompt_token_ids': ids} for ids in prompts], sampling, use_tqdm=False)
    rows = [dict(question=row['question'], prompt_token_ids=ids, response_token_ids=list(out.outputs[0].token_ids),
                 response_text=tokenizer.decode(out.outputs[0].token_ids, skip_special_tokens=False),
                 finish_reason=out.outputs[0].finish_reason, stop_reason=out.outputs[0].stop_reason)
            for row, ids, out in zip(selected, prompts, outputs)]
    q.jobs.write_json(folder / 'raw.json', rows)
    if len(rows) != 4 or any(out.prompt_token_ids != ids for out, ids in zip(outputs, prompts)):
        raise ValueError('GPU smoke coverage/input mismatch')
    q.jobs.write_json(folder / 'acceptance.json', q.qualify.validate_non_thinking(rows))


def train_variant(root, variant, runtime, commit, runner, protected, plan):
    probe = root / 'probes' / variant
    runner([q.base.PYTHON, ENTRYPOINT, '--ray-gate', variant],
           root / f'queue_jobs/{variant}_ray_gate', job_env={'CUDA_VISIBLE_DEVICES':''},
           deadline_epoch=time.time()+600)
    if not q.base.read_json(root / f'{variant}_ray_gate.json').get('passed'):
        raise ValueError('Ray worker gate failed')
    old_hashes = {}
    for step in (1, 2):
        runner(['bash', runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
               root / f'queue_jobs/prepare_{variant}_probe{step}',
               job_env=training_env(runtime, commit, root / 'probes', variant, step))
        require_equal_card(expected_card(runtime, commit, root / 'probes', variant, step),
                           q.base.read_json(probe / 'run_card.json'))
        job = root / f'queue_jobs/{variant}_probe{step}'
        runner(['bash', probe / 'command.sh'], job, gpu=True)
        shutil.copyfile(job / 'logs/job.log', probe / 'logs/nohup.log')
        n1.audit_training(q, probe, range(1, step+1), range(1, step+1))
        q.jobs.write_json(probe / f'rollout_acceptance_step{step}.json',
                          audit_rollouts(probe, plan, range(1, step+1), probe=True))
        if step == 1:
            old_hashes = {str(p):q.assets.sha256(p) for p in (probe / 'checkpoints/global_step_1').rglob('*') if p.is_file()}
        else:
            q.shared.verify_hashes(old_hashes)
    q.base.audit_resume(probe)
    q.shared.verify_hashes(protected)
    folder = root / variant
    require_equal_card(expected_card(runtime, commit, root, variant), q.base.read_json(folder / 'run_card.json'))
    runner(['bash', folder / 'command.sh'], folder, gpu=True, pid_name='train.pid', log_name='nohup.log')
    n1.audit_training(q, folder, sorted(STEPS), (1, *range(5, 101, 5)))
    result = audit_rollouts(folder, plan, range(1, 101))
    q.jobs.write_json(folder / 'rollout_acceptance.json', result)
    if variant == VARIANTS[-1]:
        q.jobs.write_json(root / 'paired_rollout_acceptance.json', n1.validate_paired_rollouts(
            q.base.read_json(root / VARIANTS[0] / 'rollout_acceptance.json'), result))
    runner([q.base.PYTHON, runtime / 'scripts/analyze_single_opd_diagnostics.py', '--run-dir', folder,
            '--output-dir', folder / 'figures', '--label', f'{DISPLAY_PREFIX} {variant} seed21'],
           root / f'queue_jobs/{variant}_figures', job_env={'CUDA_VISIBLE_DEVICES':''})
    return dict(passed=True, steps=100, variant=variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray-gate', choices=VARIANTS)
    parser.add_argument('--gpu-smoke', choices=('student', 'teacher'))
    args = parser.parse_args()
    configure()
    mount = subprocess.run(['findmnt', '-T', str(ROOT), '-n', '-o', 'FSTYPE'],
                           check=True, capture_output=True, text=True).stdout.strip()
    validate_location(socket.gethostname(), ROOT, mount)
    runtime = ENTRYPOINT.parents[1]
    commit = (runtime / 'DEPLOYED_COMMIT').read_text().strip()
    if runtime != ROOT / 'deployments' / commit:
        raise ValueError('An immutable deployment is required')
    sys.path.insert(0, str(runtime / 'external/revisiting_opd'))
    root = RUN_ROOT
    if args.ray_gate:
        import recover_qwen17_instruct_token as warmup
        warmup.CACHE = CACHE / ('b' if args.ray_gate == VARIANTS[0] else 't')
        warmup.ray_gate(root / f'{args.ray_gate}_ray_gate.json')
        return
    if args.gpu_smoke:
        gpu_smoke(root, args.gpu_smoke)
        return
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    root.mkdir(parents=True, exist_ok=True)
    state = root / 'queue_state.json'
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'queue_manifest.json').exists():
            raise FileExistsError('Existing attempt; recovery needs explicit verified state, never restart blindly')
        (root / 'queue.pid').write_text(str(os.getpid())+'\n')
        q.jobs.write_json(root / 'queue_manifest.json', dict(source_commit=commit, created_at=q.jobs.now(),
            host=HOST, hardware=HARDWARE, total_training_steps=100,
            train_batch_size=32, rollout_group_size=1, seed=21, request_seed_rule='legacy',
            student=str(q.STUDENT), teacher=str(q.TEACHER), prompt_protocol=PROTOCOL,
            checkpoint_steps=sorted(STEPS), evaluation_steps=STEPS, initial_student_reevaluated=False,
            ordering=ORDERING,
            independent_original_initialization=True, retain_all_checkpoints=True, retain_all_rollouts=True,
            no_automatic_retries=True, full_eval_autostart=True, environment=str(VENV),
            lifetime=LIFETIME))
        try:
            q.jobs.wait_for_idle()
            if shutil.disk_usage(ROOT).free < 1_000_000_000_000:
                raise ValueError('Less than 1TB free; never prune checkpoints')
            from recover_historical17_llama import check_socket_budget
            CACHE.mkdir(exist_ok=True)
            local = subprocess.run(['findmnt','-T',str(CACHE),'-n','-o','FSTYPE,OPTIONS'],
                                   check=True,capture_output=True,text=True).stdout.strip().split(maxsplit=1)
            q.validate_cache_mount(*local, shutil.disk_usage(CACHE).free)
            for v in VARIANTS:
                check_socket_budget(CACHE / v / 'train/tmp')
            env = dict(os.environ, PATH=str(VENV / 'bin')+':'+os.environ.get('PATH',''),
                PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
                VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(q.base.GRADER),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            for key, suffix in dict(TMPDIR='tmp', VLLM_CACHE_ROOT='vllm', TRITON_CACHE_DIR='triton',
                    TORCHINDUCTOR_CACHE_DIR='inductor', CUDA_CACHE_PATH='cuda', OUTLINES_CACHE_DIR='outlines').items():
                (CACHE / suffix).mkdir(exist_ok=True)
                env[key] = str(CACHE / suffix)
            def runner(command, job, job_env=None, **kwargs):
                try:
                    q.jobs.run_job(command, job, runtime, state, {**env, **(job_env or {})}, **kwargs)
                except BaseException:
                    pidfile = job / kwargs.get('pid_name', 'job.pid')
                    if kwargs.get('gpu') and pidfile.exists():
                        q.cleanup_failed_group(int(pidfile.read_text()))
                    raise
            runner(['sha256sum', '-c', '.expected.sha256'], root / 'queue_jobs/runtime_hashes')
            protected = preflight(runtime)
            freeze = subprocess.run([str(q.base.PYTHON), '-m', 'pip', 'freeze'],
                                    check=True, capture_output=True, text=True).stdout
            q.jobs.write_json(root / 'environment.json', dict(packages=q.qualify.runtime_versions(),
                              python=sys.version, pip_freeze=freeze, host=HOST))
            q.jobs.write_json(root / 'historical_grading_acceptance.json', grading_gate())
            q.jobs.write_json(root / 'input_contract.json', q.input_contract())
            plan = n1.input_plan(q)
            q.jobs.write_json(root / 'input_plan.json', plan)
            protected[str(root / 'input_plan.json')] = q.assets.sha256(root / 'input_plan.json')
            for label in ('student', 'teacher'):
                runner([q.base.PYTHON, ENTRYPOINT, '--gpu-smoke', label],
                       root / f'queue_jobs/gpu_smoke_{label}', gpu=True, job_env={'CUDA_VISIBLE_DEVICES':'0'})
                if q.base.read_json(root / f'gpu_smoke/{label}/acceptance.json').get('passed') is not True:
                    raise ValueError('GPU nonthinking gate failed')
            cards = {}
            for v in VARIANTS:
                runner(['bash', runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
                       root / f'queue_jobs/prepare_{v}_formal', job_env=training_env(runtime, commit, root, v))
                cards[v] = q.base.read_json(root / v / 'run_card.json')
                require_equal_card(expected_card(runtime, commit, root, v), cards[v])
                for name in ('run_card.json','command.sh','artifact_hashes.sha256','script_hashes.sha256'):
                    protected[str(root / v / name)] = q.assets.sha256(root / v / name)
            allowed = {'variant','opd_block_size','opd_block_advantage_mode','experiment_name',
                       'diagnostic_output_dir','lossless_rollout_dir','opd_block_ablation'}
            left, right = cards.values()
            require_equal_card({k:v for k,v in left.items() if k not in allowed}, right)
            q.jobs.write_json(root / 'paired_preflight.json', dict(passed=True, cards=cards))
            q.jobs.write_json(root / 'protected_inputs.json', protected)
            results = {}
            def evaluate(name):
                result = q.evaluate_model(root, name, runtime, commit, runner, protected)
                results[name] = result
                q.jobs.write_json(root / 'evaluation_acceptance.json', dict(passed=True, complete=len(results)==8, models=results))
                q.jobs.write_json(root / 'per_benchmark_results.json', {
                    t:{n:r['per_task'][t] for n,r in results.items()} for t in q.shared.TASK_COUNTS})
                return result
            execute_ordered(lambda v: train_variant(root, v, runtime, commit, runner, protected, plan), evaluate)
            write_comparison(q, root, results)
            report = q.base.read_json(root / 'paired_comparison.json')
            report['evaluation_protocol'] = PROTOCOL
            q.jobs.write_json(root / 'paired_comparison.json', report)
            q.shared.verify_hashes(protected)
            q.jobs.write_json(state, dict(status='complete', updated_at=q.jobs.now(), models=list(results)))
        except BaseException as error:
            previous = q.base.read_json(state) if state.exists() else {}
            q.jobs.write_json(state, dict(previous, status='failed', error=repr(error), updated_at=q.jobs.now()))
            raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Run the approved 100-step, 32x1 historical pair block-first on ml2."""

import argparse
from collections import defaultdict
import fcntl
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import prepare_historical_pair_n1 as preparation

VARIANTS = ('block3_mean', 'token_opd')
STEPS = tuple(reversed(preparation.SAVE_STEPS))
CACHE = Path('/dev/shm/hp1')


def execute_ordered(train, evaluate):
    results = {}
    for variant in VARIANTS:
        if train(variant).get('passed') is not True:
            raise ValueError(f'Training acceptance failed: {variant}')
        names = [f'{variant}_step{s}' for s in STEPS]
        for name in names:
            result = evaluate(name)
            if result.get('passed') is not True:
                raise ValueError(f'Evaluation acceptance failed: {name}')
            results[name] = result
    return results


def probe_env(run, runtime, commit, root, variant, step):
    if step not in (1, 2):
        raise ValueError('Only save1/resume2 probes are allowed')
    env = preparation.training_env(run, runtime, commit, root/'probes', variant)
    env.update(STOP_AFTER_STEP=str(step), OPD_DIAG_INTERVAL='1',
        DIAGNOSTIC_SAVE_STEPS='1' if step == 1 else '1,2',
        ROLLOUT_ATTEMPT_ID=f'probe{step}', RESUME_MODE='disable' if step == 1 else 'resume_path',
        RESUME_FROM_PATH='' if step == 1 else str(root/f'probes/{variant}/checkpoints/global_step_1'))
    return env


def validate_probe_card(source, actual, root, variant, step):
    expected = preparation.expected_card(source, root/'probes', variant)
    expected.update(stop_after_step=step, opd_diag_interval=1,
        diagnostic_save_steps='1' if step == 1 else '1,2', rollout_attempt_id=f'probe{step}',
        resume_mode='disable' if step == 1 else 'resume_path',
        resume_from_path='' if step == 1 else str(root/f'probes/{variant}/checkpoints/global_step_1'))
    if expected != actual:
        raise ValueError('Probe differs from the approved 100-step recipe')


def batch_fingerprint(rows, sources, step):
    if (len(rows) != 32 or len(sources) != 32 or len({r['traj_uid'] for r in rows}) != 32
            or any(r['step'] != step or r['sample_index'] != i or r['source_extra_info'] != source
                   or r['sampling'].get('n') != 1 for i, (r, source) in enumerate(zip(rows, sources)))):
        raise ValueError(f'Incorrect 32x1 sampler order or identities at step {step}')
    values = [[r[k] for k in ('source_extra_info', 'prompt_token_ids', 'sampling')] for r in rows]
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_paired_rollouts(block, token):
    for record in (block, token):
        if record.get('passed') is not True or record.get('total_rollouts') != 3200 or len(record['steps']) != 100:
            raise ValueError('Both complete 100-step trajectory audits are required')
    for step, (left, right) in enumerate(zip(block['steps'], token['steps']), 1):
        if (left['step'] != step or right['step'] != step or left['count'] != 32 or right['count'] != 32
                or left['paired_input_sha256'] != right['paired_input_sha256']):
            raise ValueError(f'Paired prompt/seed mismatch at step {step}')
    return dict(passed=True, steps=100, trajectories_per_arm=3200)


def input_plan(run):
    import datasets
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo_multitask import create_rl_sampler
    from verl.utils.dataset.multitask_rl_dataset import MultiTaskRLHFDataset

    path = run.base.DATA/'train.parquet'
    parquet = pq.ParquetFile(path)
    # Use the actual frozen sampler; only its task-index construction omits unused text columns.
    dataset = MultiTaskRLHFDataset.__new__(MultiTaskRLHFDataset)
    dataset.dataframe = range(parquet.metadata.num_rows)
    dataset.task_indices = defaultdict(list)
    if 'task_type' in parquet.schema_arrow.names:
        for index, task in enumerate(parquet.read(columns=['task_type'])['task_type'].to_pylist()):
            dataset.task_indices[task].append(index)
    else:
        dataset.task_indices['unknown'] = list(range(len(dataset)))
    config = OmegaConf.create(dict(batching_mode='sequential', train_batch_size=32, shuffle=True, seed=21))
    indices = list(islice(iter(create_rl_sampler(config, dataset)), 3200))
    if len(indices) != 3200 or len(set(indices)) != 3200:
        raise ValueError('Incomplete or repeated physical-row sampler plan')
    rows = datasets.load_dataset('parquet', data_files=str(path))['train'].select(indices)['extra_info']
    return dict(sampler='frozen create_rl_sampler / SequentialTaskSampler', physical_rows=len(dataset),
                seed=21, prompts_per_step=32, indices=list(map(int, indices)), sources=list(rows))


def audit_rollouts(recipe, run, folder, plan, steps):
    from transformers import AutoTokenizer
    from diagnose_token_truncation import analyze_tokens
    tokenizer = AutoTokenizer.from_pretrained(run.STUDENT, local_files_only=True)
    records = []
    for step in steps:
        paths = list((folder/'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(paths) != 1:
            raise ValueError(f'Missing/duplicate raw trajectory step {step}')
        rows = run.read_archive(paths[0])
        fingerprint = batch_fingerprint(rows, plan['sources'][32*(step-1):32*step], step)
        for row in rows:
            recipe.validate_legacy_rollout(row, step)
            recipe.validate_qwen_historical_prompt(tokenizer, row['source_extra_info']['question'],
                                                  row['prompt_token_ids'], max_prompt_length=2048)
        records.append(dict(step=step, count=32, paired_input_sha256=fingerprint,
            sha256=run.assets.sha256(paths[0]), length_stops=sum(r['finish_reason']=='length' for r in rows),
            generated_think_tags=sum(r['generated_think_tags'] for r in rows),
            periodic_tails=sum(analyze_tokens(r['response_token_ids'], 16384)['tail_period'] is not None for r in rows),
            unique_response_count=len({tuple(r['response_token_ids']) for r in rows})))
    return dict(passed=True, steps=records, total_rollouts=32*len(records))


def audit_training(run, folder, checkpoint_steps, diagnostic_steps):
    import torch
    import audit_block10_run as audit
    import run_qwen8_teacher_pair as checkpoints
    torch.set_num_threads(2)
    if (folder/'exit_code.txt').read_text().strip() != '0':
        raise ValueError('Training did not exit successfully')
    scalar_records = [json.loads(s) for s in (folder/'diagnostics/scalars.jsonl').read_text().splitlines() if s.strip()]
    if [r['step'] for r in scalar_records] != list(diagnostic_steps):
        raise ValueError('Scalar diagnostic steps differ from the required schedule')
    paths = sorted((folder/'diagnostics').glob('step_*.npz'))
    if [int(p.stem.split('_')[-1]) for p in paths] != list(diagnostic_steps):
        raise ValueError('Position diagnostic steps differ from the required schedule')
    issues = audit.scalar_record_issues(scalar_records) + audit.numerical_scalar_issues(scalar_records)
    for path, step in zip(paths, diagnostic_steps):
        issues += audit.diagnostic_snapshot_issues(path, step)
    for step in checkpoint_steps:
        issues += audit.checkpoint_issues(folder, step)
    if issues:
        raise ValueError(issues)
    states = []
    for step in checkpoint_steps:
        checkpoint = folder/f'checkpoints/global_step_{step}'
        data = torch.load(checkpoint/'data.pt', map_location='cpu', weights_only=False)
        for rank in range(4):
            extra = torch.load(checkpoint/f'actor/extra_state_world_size_4_rank_{rank}.pt',
                               map_location='cpu', weights_only=False)
            run.base.validate_resume_state(extra, data, step)
            states.append(dict(step=step, rank=rank, rng_scheduler_sampler_valid=True))
    checkpoints.verify_optimizer_state(folder, checkpoint_steps)
    result = dict(passed=True, checkpoint_steps=list(checkpoint_steps),
                  diagnostic_steps=list(diagnostic_steps), states=states)
    run.jobs.write_json(folder/'training_state_acceptance.json', result)
    return result


def preflight(recipe, run, runtime, root, source):
    accepted = run.base.read_json(root/'preparation_acceptance.json')
    prepared = run.base.read_json(root/'preparation.json')
    if (not accepted.get('passed') or not prepared.get('prepared')
            or run.assets.sha256(root/'preparation.json') != accepted['prepared_sha256']):
        raise ValueError('Missing/changed reviewed preparation')
    if (prepared['total_steps'] != 100 or set(prepared['cards']) != set(VARIANTS)
            or prepared['evaluation']['checkpoint_steps'] != list(STEPS)
            or prepared['evaluation']['responses_per_problem'] != 8):
        raise ValueError('Wrong prepared scope')
    protected = {str(root/name): digest for name, digest in prepared['files'].items()}
    protected[str(root/'preparation.json')] = accepted['prepared_sha256']
    for variant in VARIANTS:
        folder = root/variant
        if (folder/'started_at.txt').exists() or list((folder/'checkpoints').glob('global_step_*')):
            raise ValueError('Existing training attempt; never restart blindly')
        preparation.validate_card(source, run.base.read_json(folder/'run_card.json'), root, variant)
        for name in ('artifact_hashes.sha256', 'script_hashes.sha256'):
            protected.update(recipe.manifest_hashes(folder/name))
    protected[str(run.base.GRADER)] = run.shared.grading.HISTORICAL_GRADER_SHA256
    run.shared.verify_hashes(protected)
    run.validate_benchmark_hashes({t: run.assets.sha256(run.base.DATA/'eval_jsonl'/f'{t}.jsonl')
                                  for t in run.shared.TASK_COUNTS})
    run.jobs.write_json(root/'environment.json', dict(packages=run.qualify.runtime_versions(), python=sys.version))
    return protected


def train_variant(recipe, run, root, variant, runtime, commit, source, plan, runner, protected):
    probe = root/'probes'/variant
    runner([run.base.PYTHON, Path(__file__).resolve(), '--runtime', runtime, '--run-root', root,
            '--ray-gate', variant], root/f'queue_jobs/{variant}_ray_gate',
           job_env={'CUDA_VISIBLE_DEVICES': ''}, deadline_epoch=time.time()+360)
    if not run.base.read_json(root/f'{variant}_ray_gate.json').get('passed'):
        raise ValueError('Ray CPU worker gate failed')
    checkpoint1_hashes = {}
    for step in (1, 2):
        runner(['bash', runtime/'scripts/launch_revisiting_block_opd_formal_train.sh'],
               root/f'queue_jobs/prepare_{variant}_probe{step}',
               job_env=probe_env(run, runtime, commit, root, variant, step))
        validate_probe_card(source, run.base.read_json(probe/'run_card.json'), root, variant, step)
        job = root/f'queue_jobs/{variant}_probe{step}'
        runner(['bash', probe/'command.sh'], job, gpu=True)
        shutil.copyfile(job/'logs/job.log', probe/'logs/nohup.log')
        audit_training(run, probe, range(1,step+1), range(1,step+1))
        run.jobs.write_json(probe/f'rollout_acceptance_step{step}.json',
                           audit_rollouts(recipe, run, probe, plan, range(1,step+1)))
        if step == 1:
            checkpoint1_hashes = {str(p):run.assets.sha256(p)
                for p in (probe/'checkpoints/global_step_1').rglob('*') if p.is_file()}
        else:
            run.shared.verify_hashes(checkpoint1_hashes)
    run.base.audit_resume(probe)
    run.shared.verify_hashes(protected)
    folder = root/variant
    preparation.validate_card(source, run.base.read_json(folder/'run_card.json'), root, variant)
    runner(['bash', folder/'command.sh'], folder, gpu=True, pid_name='train.pid', log_name='nohup.log')
    audit_training(run, folder, preparation.SAVE_STEPS, (1,*range(5,101,5)))
    evidence = audit_rollouts(recipe, run, folder, plan, range(1,101))
    run.jobs.write_json(folder/'rollout_acceptance.json', evidence)
    if variant == 'token_opd':
        run.jobs.write_json(root/'paired_rollout_acceptance.json', validate_paired_rollouts(
            run.base.read_json(root/'block3_mean/rollout_acceptance.json'), evidence))
    runner([run.base.PLOT_PYTHON, runtime/'scripts/analyze_single_opd_diagnostics.py',
            '--run-dir', folder, '--output-dir', folder/'figures', '--label', f'Historical1.7B n1 {variant} seed21'],
           root/f'queue_jobs/{variant}_figures', job_env={'CUDA_VISIBLE_DEVICES': ''})
    run.shared.verify_hashes(protected)
    return dict(passed=True, steps=100, variant=variant, checkpoint_steps=list(preparation.SAVE_STEPS))


def write_comparison(run, root, results):
    expected = {f'{v}_step{s}' for v in VARIANTS for s in STEPS}
    if set(results) != expected or any(not r.get('passed') for r in results.values()):
        raise ValueError('All eight checkpoint evaluations required')
    report = dict(training_protocol=run.PROTOCOL, evaluation_protocol='legacy', training_seed=21,
                  primary_checkpoint=100, total_steps=100, prompts_per_step=32, responses_per_prompt=1,
                  initial_student_reevaluated=False,
                  per_benchmark={})
    for task in run.shared.TASK_COUNTS:
        values = {name: result['per_task'][task] for name, result in results.items()}
        for step in STEPS:
            block, token = (values[f'{v}_step{step}'] for v in VARIANTS)
            values[f'block_minus_token_step{step}_pp'] = {
                key:100*(block[key]-token[key]) for key in ('avg_at_8','pass_at_8')}
        report['per_benchmark'][task] = values
    run.jobs.write_json(root/'paired_comparison.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--ray-gate', choices=VARIANTS)
    args = parser.parse_args()
    runtime, root = args.runtime, args.run_root
    recipe, run, commit = preparation.load_recipe(runtime)
    run.validate_host(socket.gethostname())
    if root != recipe.ROOT/'runs/20260925v3_historical17_pair_n1_step100_save25_seed21_ml2':
        raise ValueError('Only the reviewed preparation is authorized')
    if args.ray_gate:
        import recover_qwen17_instruct_token as warmup
        warmup.CACHE = CACHE/args.ray_gate/'warm'
        warmup.ray_gate(root/f'{args.ray_gate}_ray_gate.json')
        return
    source = run.base.read_json(recipe.RUN_ROOT/'adv3/run_card.json')
    run.validate_card(source, recipe.RUN_ROOT, 'adv3', commit)
    run.VARIANTS, run.STEPS, run.SAVE_STEPS = VARIANTS, STEPS, ','.join(map(str, preparation.SAVE_STEPS))
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    state = root/'queue_state.json'
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'queue_manifest.json').exists():
            raise FileExistsError('Existing queue attempt: do not overwrite or retry automatically')
        (root/'queue.pid').write_text(str(os.getpid())+'\n')
        control_hashes = {str(p):run.assets.sha256(p) for p in Path(__file__).resolve().parent.glob('*.py')}
        run.jobs.write_json(root/'queue_manifest.json', dict(created_at=run.jobs.now(), runtime=str(runtime),
            source_commit=commit, control_files=control_hashes, variants=VARIANTS, total_training_steps=100,
            checkpoint_steps=list(preparation.SAVE_STEPS), rollout_group_size=1, train_batch_size=32,
            initialization='original student independently after own probe',
            authorization='Start Block3 training and full eval, only then Token training and full eval',
            ordering='Block3 probe/train/eval100,75,50,25 -> Token probe/train/eval100,75,50,25',
            preparation_amendment='User cancels initial-student re-evaluation; existing result retained',
            initial_student_reevaluated=False, expected_evaluation_count=8,
            no_automatic_retries=True, preserve_all_checkpoints=True, retain_all_rollouts=True,
            full_eval_autostart=True))
        try:
            run.jobs.wait_for_idle()
            if shutil.disk_usage(recipe.ROOT).free < 1_000_000_000_000:
                raise ValueError('Less than 1TB free; checkpoint pruning not authorized')
            CACHE.mkdir(exist_ok=True)
            from recover_historical17_llama import check_socket_budget
            mount = subprocess.run(['findmnt','-T',str(CACHE),'-n','-o','FSTYPE,OPTIONS'],
                                   check=True,capture_output=True,text=True).stdout.strip().split(maxsplit=1)
            run.validate_cache_mount(*mount, shutil.disk_usage(CACHE).free)
            for variant in VARIANTS:
                check_socket_budget(CACHE/variant/'train/tmp')
            env = dict(os.environ, PATH=str(run.base.PYTHON.parent)+':'+os.environ.get('PATH',''),
                PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
                VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(run.base.GRADER))
            if env.get('RAY_TMPDIR'):
                raise ValueError('Unexpected inherited Ray directory')
            for key, suffix in dict(TMPDIR='tmp', VLLM_CACHE_ROOT='vllm', TRITON_CACHE_DIR='triton',
                    TORCHINDUCTOR_CACHE_DIR='inductor', CUDA_CACHE_PATH='cuda', OUTLINES_CACHE_DIR='outlines').items():
                (CACHE/suffix).mkdir(exist_ok=True)
                env[key] = str(CACHE/suffix)
            def runner(command, job, job_env=None, **kwargs):
                try:
                    run.jobs.run_job(command, job, runtime, state, {**env, **(job_env or {})}, **kwargs)
                except BaseException:
                    pidfile = job/kwargs.get('pid_name','job.pid')
                    if kwargs.get('gpu') and pidfile.exists():
                        run.cleanup_failed_group(int(pidfile.read_text()))
                    raise
            runner(['sha256sum','-c','.expected.sha256'], root/'queue_jobs/runtime_hashes')
            protected = preflight(recipe, run, runtime, root, source)
            protected.update(control_hashes)
            run.jobs.write_json(root/'input_contract.json', run.input_contract())
            plan = input_plan(run)
            run.jobs.write_json(root/'input_plan.json', plan)
            protected[str(root/'input_plan.json')] = run.assets.sha256(root/'input_plan.json')
            run.jobs.write_json(root/'protected_inputs.json', protected)
            results = {}
            def evaluate(name):
                result = run.evaluate_model(root, name, runtime, commit, runner, protected)
                results[name] = result
                run.jobs.write_json(root/'evaluation_acceptance.json',
                    dict(passed=True, complete=len(results)==8, models=results))
                run.jobs.write_json(root/'per_benchmark_results.json', {
                    task:{n:r['per_task'][task] for n,r in results.items()} for task in run.shared.TASK_COUNTS})
                print(json.dumps(dict(evaluated=name, per_task=result['per_task'])), flush=True)
                return result
            execute_ordered(lambda v: train_variant(recipe, run, root, v, runtime, commit, source,
                                                    plan, runner, protected), evaluate)
            write_comparison(run, root, results)
            run.shared.verify_hashes(protected)
            run.jobs.wait_for_idle()
            run.jobs.write_json(state, dict(status='complete', updated_at=run.jobs.now(),
                variants=VARIANTS, checkpoint_steps=list(preparation.SAVE_STEPS), evaluations=list(results),
                protected_inputs_verified=True))
        except BaseException as error:
            previous = run.base.read_json(state) if state.exists() else {}
            run.jobs.write_json(state, dict(previous, status='failed', error=repr(error), updated_at=run.jobs.now()))
            raise


if __name__ == '__main__':
    main()

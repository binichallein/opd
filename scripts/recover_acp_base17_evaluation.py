#!/usr/bin/env python3
"""One-shot, evaluation-only continuation after the September 27 ACP restart."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys

ROOT = Path('/mnt/afs/202609/tyf-qwen-opd')
SOURCE = ROOT / 'runs/20260927v2_base17_grpo_migrated_n1_seed21_acp'
COMMIT = '781df3021f8b84936921e068aa2251e38e686bfd'
RUNTIME = ROOT / 'deployments' / COMMIT
RECOVERY = SOURCE / 'recoveries/20260928v2_token25'
HOST = 'pt-2562f77e00cd4f92b6880471e90635f3-worker-0'
TASKS = ('math500', 'aime24', 'aime25', 'amc23')
REUSE, MISSING = TASKS[:2], TASKS[2:]
ACCEPTED = {f'{v}_step{s}' for v in ('block3_mean', 'token_opd')
            for s in (25, 50, 75, 100)} - {'token_opd_step25'}
ENTRY = Path(__file__).resolve()
NAME = 'token_opd_step25'


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def accepted_results(report):
    models = report.get('models', {})
    if (report.get('passed') is not True or report.get('complete') is not False
            or set(models) != ACCEPTED or any(r.get('passed') is not True for r in models.values())):
        raise ValueError('Exactly seven accepted evaluations are required')
    return models


def missing_command(original):
    command = list(original)
    start = command.index('--tasks') + 1
    end = start
    while end < len(command) and not str(command[end]).startswith('--'):
        end += 1
    if tuple(command[start:end]) != TASKS:
        raise ValueError('Unexpected source evaluation task list')
    command[start:end] = MISSING
    return command


def validate_partial_config(old, new):
    if tuple(old.get('tasks', [])) != TASKS or tuple(new.get('tasks', [])) != MISSING:
        raise ValueError('Wrong reused or missing tasks')
    ignored = {'tasks', 'rollout_archive_dir'}
    if {k: v for k, v in old.items() if k not in ignored} != {
            k: v for k, v in new.items() if k not in ignored}:
        raise ValueError('Recovery changed the evaluation protocol')


def has_native_think(row):
    return bool({151667, 151668}.intersection(row['response_token_ids']))


def copy_verified(source, destination, expected):
    if source.is_symlink() or sha256(source) != expected:
        raise ValueError(f'Source file changed or is a symlink: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    if sha256(destination) != expected:
        raise ValueError(f'Recovery copy mismatch: {destination}')


def frozen_runner():
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(RUNTIME), str(RUNTIME / 'scripts'), str(RUNTIME / 'external/revisiting_opd')]
    import run_acp_base17_migration as migration
    if Path(migration.__file__).resolve() != RUNTIME / 'scripts/run_acp_base17_migration.py':
        raise ValueError('Recovery must import the original frozen controller')
    migration.validate_interpreter(sys.prefix, sys.executable)
    if socket.gethostname() != HOST:
        raise ValueError('Recovery is pinned to the verified replacement worker')
    run = migration.configured_runner()
    run.configure()
    return migration, run


def copy_tasks(source, destination, tasks, config, protected):
    import regrade_opd_eval_external as grading
    for task in tasks:
        paths = [grading.raw_output_path(source, task, config)]
        paths += [source / 'rollout_archive' / task / f'rollout_{i}.jsonl' for i in range(8)]
        for path in paths:
            digest = sha256(path)
            protected[str(path)] = digest
            copy_verified(path, destination / path.relative_to(source), digest)


def audit_rows(q, output, config, tasks):
    from transformers import AutoTokenizer
    import regrade_opd_eval_external as grading
    from opd_ext.eval_rollout_archive import audit_archive
    evidence = audit_archive(output, dict(config, tasks=list(tasks)))
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    for task in tasks:
        rows = grading.load_jsonl(grading.raw_output_path(output, task, config))
        groups = grading.validate_task_rows(rows, task, q.shared.TASK_COUNTS[task], 0)
        expected = {str(r['id']): r for r in grading.load_jsonl(q.base.DATA / 'eval_jsonl' / f'{task}.jsonl')}
        if set(groups) != set(expected):
            raise ValueError(f'Wrong benchmark identities: {task}')
        for row in rows:
            example = expected[str(row['example_id'])]
            if any(row.get(k) != example[k] for k in ('problem', 'prompt', 'answer')):
                raise ValueError(f'Changed benchmark inputs: {task}')
            q.validate_eval_row(row, tokenizer)
    return evidence


def preflight(migration, run):
    q = run.q
    for runtime in (RUNTIME, ENTRY.parents[1]):
        print(f'{q.jobs.now()} verifying deployment {runtime}', flush=True)
        with (RECOVERY / f'verify_{runtime.name}.log').open('x') as log:
            subprocess.run(['sha256sum', '-c', '.expected.sha256'], cwd=runtime, stdout=log, check=True)
    if (RUNTIME / 'DEPLOYED_COMMIT').read_text().strip() != COMMIT:
        raise ValueError('Wrong source runtime')
    protected = {str(ENTRY): sha256(ENTRY)}
    for path in (SOURCE / 'queue_state.json', SOURCE / 'evaluation_acceptance.json',
                 SOURCE / 'token_opd/training_state_acceptance.json', SOURCE / 'paired_rollout_acceptance.json',
                 SOURCE / 'evaluations' / NAME / 'model_identity.json'):
        protected[str(path)] = sha256(path)
    training = read_json(SOURCE / 'token_opd/training_state_acceptance.json')
    paired = read_json(SOURCE / 'paired_rollout_acceptance.json')
    if (training.get('passed') is not True or training.get('checkpoint_steps') != [25, 50, 75, 100]
            or paired.get('passed') is not True or paired.get('trajectories_per_arm') != 3200):
        raise ValueError('Completed training/paired rollouts not accepted; never retrain here')
    results = accepted_results(read_json(SOURCE / 'evaluation_acceptance.json'))
    for name, result in results.items():
        folder = SOURCE / 'evaluations' / name
        if (folder / 'exit_code.txt').read_text().strip() != '0' or read_json(folder / 'acceptance.json') != result:
            raise ValueError(f'Accepted evaluation differs: {name}')
        protected.update(result['sha256'])
    model = SOURCE / 'merged' / NAME
    identity = read_json(SOURCE / 'evaluations' / NAME / 'model_identity.json')
    if identity.get('model') != str(model) or not identity.get('sha256'):
        raise ValueError('Missing model identity')
    protected.update(identity['sha256'])
    protected.update(q.common_preflight(RUNTIME, {}))
    current = dict(python=sys.version, packages=migration.runtime_versions(), host=HOST,
                   pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    alignment = migration.validate_environment(read_json(migration.IMPORT_ROOT / 'environment.json'), current)
    gpu_query = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total',
                                        '--format=csv,noheader,nounits'], text=True)
    gpus = [dict(name=line.rsplit(',', 1)[0], memory_mib=int(line.rsplit(',', 1)[1]))
            for line in gpu_query.splitlines()]
    cpu = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    migration.validate_hardware(gpus, cpu)
    current.update(gpus=gpus, cpu_max=cpu,
                   nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True))
    q.jobs.write_json(RECOVERY / 'environment.json', current)
    q.jobs.write_json(RECOVERY / 'environment_alignment.json', alignment)
    old = SOURCE / 'evaluations' / NAME / 'outputs'
    config = read_json(old / 'eval_config.json')
    required = dict(q.shared.EVAL_VALUES, tasks=list(TASKS), model_path=str(model),
                    eval_jsonl_dir=str(q.base.DATA / 'eval_jsonl'), gpus=['0', '1', '2', '3'],
                    prompt_protocol=run.PROTOCOL, retain_rollouts=True)
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Interrupted evaluation configuration drift')
    if {p.name for p in (old / 'rollout_archive').iterdir()} != set(REUSE):
        raise ValueError('Unexpected partial archives: manual inspection required')
    if any((old / f'{task}_t1.0_p0.9_n8-MNT16384.jsonl').exists() for task in MISSING):
        raise ValueError('Missing-task outputs already exist; refusing regeneration')
    protected[str(old / 'eval_config.json')] = sha256(old / 'eval_config.json')
    print(f'{q.jobs.now()} verifying {len(protected)} protected model/data/result files', flush=True)
    q.shared.verify_hashes(protected)
    output = RECOVERY / 'evaluations' / NAME / 'outputs'
    print(f'{q.jobs.now()} copying and auditing 4240 existing rollouts', flush=True)
    copy_tasks(old, output, REUSE, config, protected)
    evidence = audit_rows(q, output, config, REUSE)
    q.jobs.write_json(RECOVERY / 'reused_rollouts_acceptance.json', evidence)
    q.jobs.write_json(RECOVERY / 'protected_inputs.json', protected)
    full_config = dict(config, rollout_archive_dir=str(output / 'rollout_archive'))
    q.jobs.write_json(output / 'eval_config.json', full_config)
    q.jobs.write_json(RECOVERY / 'recovery_manifest.json', dict(
        source_run=str(SOURCE), source_runtime=str(RUNTIME), source_commit=COMMIT,
        recovery_runtime=str(ENTRY.parents[1]), host=HOST, created_at=q.jobs.now(),
        reused_tasks=list(REUSE), regenerated_tasks=list(MISSING), training=False,
        source_outputs_unchanged=True, no_automatic_retries=True,
        sampling_unchanged=True, hardware=migration.HARDWARE, limitation=migration.LIMITATION,
        note='A replacement H100 worker may change floating-point generation; this is not bitwise replay.'))
    return model


def finalize(migration, run):
    import eval_qwen3_math_vllm as evaluator
    import regrade_opd_eval_external as grading
    from transformers import AutoTokenizer
    q = run.q
    protected = read_json(RECOVERY / 'protected_inputs.json')
    q.shared.verify_hashes(protected)
    folder = RECOVERY / 'evaluations' / NAME
    output, missing = folder / 'outputs', RECOVERY / 'missing_tasks/outputs'
    config = read_json(output / 'eval_config.json')
    partial = read_json(missing / 'eval_config.json')
    validate_partial_config(config, partial)
    audit_rows(q, missing, partial, MISSING)
    copy_tasks(missing, output, MISSING, config, protected)
    audit_rows(q, output, config, TASKS)
    grading.validate_grader_hash(q.base.GRADER, grading.HISTORICAL_GRADER_SHA256)
    evaluator.grade_outputs([grading.raw_output_path(output, task, config) for task in TASKS],
                            output / 'summary.json', str(q.STUDENT), 8, 'external', 21, False)
    accepted = q.shared.audit_evaluation(output, q.base.DATA / 'eval_jsonl', Path(config['model_path']))
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    for task in TASKS:
        rows = grading.load_jsonl(grading.raw_output_path(output, task, config))
        tags = 0
        for row in rows:
            q.validate_eval_row(row, tokenizer)
            tags += int(has_native_think(row))
        accepted['per_task'][task]['generated_think_tag_count'] = tags
    accepted.update(train_eval_prompt_verified=True, evaluation_protocol_verified=True)
    accepted = q.historical.accept_archive(folder, accepted)
    results = accepted_results(read_json(SOURCE / 'evaluation_acceptance.json'))
    results[NAME] = accepted
    q.shared.verify_hashes(protected)
    q.jobs.write_json(RECOVERY / 'protected_inputs_final.json', protected)
    run.write_comparison(q, RECOVERY, results)
    report = read_json(RECOVERY / 'paired_comparison.json')
    report.update(evaluation_protocol=run.PROTOCOL, same_hardware=False,
                  hardware=migration.HARDWARE, limitation=migration.LIMITATION,
                  recovery_manifest=str(RECOVERY / 'recovery_manifest.json'))
    q.jobs.write_json(RECOVERY / 'paired_comparison.json', report)
    q.jobs.write_json(RECOVERY / 'per_benchmark_results.json', {
        task: {name: result['per_task'][task] for name, result in results.items()} for task in TASKS})
    q.jobs.write_json(RECOVERY / 'evaluation_acceptance.json', dict(passed=True, complete=True, models=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    migration, run = frozen_runner()
    q = run.q
    if args.finalize:
        finalize(migration, run)
        return

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with (SOURCE / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        RECOVERY.mkdir(parents=True, exist_ok=False)
        state = RECOVERY / 'queue_state.json'
        (RECOVERY / 'queue.pid').write_text(str(os.getpid()) + '\n')
        q.jobs.write_json(state, dict(status='preflight', host=HOST, pid=os.getpid(), updated_at=q.jobs.now()))
        try:
            migration.CACHE = Path('/dev/shm/am17r')
            env = migration.controller_env(run, RUNTIME)
            q.jobs.wait_for_idle()
            model = preflight(migration, run)
            command = missing_command(q.evaluation_command(RUNTIME, model, RECOVERY / 'missing_tasks/outputs'))
            q.jobs.run_job(command, RECOVERY / 'missing_tasks', RUNTIME, state, env, gpu=True)
            q.jobs.run_job([q.base.PYTHON, ENTRY, '--finalize'], RECOVERY / 'queue_jobs/finalize',
                           RUNTIME, state, dict(env, CUDA_VISIBLE_DEVICES=''))
            q.jobs.run_job([q.base.PYTHON, RUNTIME / 'scripts/analyze_single_opd_diagnostics.py',
                           '--run-dir', migration.IMPORT_ROOT / 'block3_mean',
                           '--output-dir', RECOVERY / 'figures/imported_block3',
                           '--label', 'Imported ml2 Base1.7 Block3 seed21'],
                          RECOVERY / 'queue_jobs/imported_block3_figures', RUNTIME, state,
                          dict(env, CUDA_VISIBLE_DEVICES=''))
            q.shared.verify_hashes(read_json(RECOVERY / 'protected_inputs_final.json'))
            q.jobs.write_json(state, dict(status='completed', host=HOST, updated_at=q.jobs.now(),
                                         evaluations_complete=8, training_restarted=False))
        except BaseException as error:
            q.jobs.write_json(state, dict(status='failed', error=repr(error), updated_at=q.jobs.now(),
                                         no_automatic_retry=True))
            raise


if __name__ == '__main__':
    main()

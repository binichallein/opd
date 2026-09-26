#!/usr/bin/env python3
"""One-shot ACP continuation: imported Block3 evaluations, then independent Token.

The import manifest v1 contains source_host, source_run, source_commit, created_at,
hardware, source_success and files[{source, destination, sha256, size_bytes}].
transfer_acceptance.json binds its SHA256, files_verified and total_bytes.
Neither source cards nor imported artifacts are rewritten by this controller.
"""

import argparse
from contextlib import contextmanager
from functools import partial
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_acp_qwen17_n1 as acp
import run_ml2_base17_protocol_n1 as ml2

ROOT = acp.ROOT
HOST = acp.HOST
VENV = ROOT / 'envs/verl-cu128-base17-migration-v1'
RUN_ROOT = ROOT / 'runs/20260927v1_base17_grpo_migrated_n1_seed21_acp'
IMPORT_ROOT = ROOT / 'imports/20260927_ml2_base17_protocol'
SOURCE_RUN = ml2.RUN_ROOT
SOURCE_HOST = 'di-20260407234928-vrvxk'
SOURCE_COMMIT = '3316918c4ef5488d4ca4b5100c62b831eede979d'
REFERENCE = ROOT / 'deployments' / SOURCE_COMMIT
ENTRYPOINT = Path(__file__).resolve()
CACHE = Path('/dev/shm/am17')
STEPS = (100, 75, 50, 25)
HARDWARE = dict(block3_training='4xA100-80GB', token_training='4xH100-80GB', evaluation='4xH100-80GB')
LIMITATION = 'Block3 trained on ml2 A100/64 CPUs; Token on ACP H100/32 CPUs. No same-hardware or bitwise equivalence claim.'
FROZEN_SCRIPTS = (
    'scripts/launch_revisiting_block_opd_formal_train.sh',
    'scripts/run_revisiting_sampled_block_opd_math.sh',
    'scripts/eval_math_batched.py', 'scripts/eval_qwen3_math_vllm.py',
    'scripts/run_window_queue.py', 'scripts/run_qwen17_instruct_pair.py',
    'scripts/run_qwen17_base_grpo_pair.py', 'scripts/run_ml2_base17_protocol_n1.py',
    'scripts/run_acp_qwen17_n1.py', 'scripts/run_historical_pair_n1.py',
    'scripts/run_qwen06_pair.py', 'scripts/run_llama32_pair.py',
    'scripts/run_historical17_reeval_llama.py', 'scripts/compare_paired_opd_evals.py',
    'scripts/regrade_opd_eval_external.py', 'scripts/regrade_predictions.py',
    'scripts/audit_block10_run.py', 'scripts/run_qwen8_teacher_pair.py',
    'scripts/analyze_single_opd_diagnostics.py', 'scripts/diagnose_token_truncation.py',
)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def execute_ordered(train, evaluate):
    results = {}
    for variant in ('block3_mean', 'token_opd'):
        if variant == 'token_opd' and train(variant).get('passed') is not True:
            raise ValueError('Token training acceptance failed')
        for step in STEPS:
            name = f'{variant}_step{step}'
            result = evaluate(name)
            if result.get('passed') is not True:
                raise ValueError(f'Full evaluation acceptance failed: {name}')
            results[name] = result
    return results


def destination_for(source):
    mappings = ((SOURCE_RUN, IMPORT_ROOT),
                (ml2.STUDENT, ROOT / 'models/Qwen3-1.7B-Base'),
                (ml2.TEACHER, ROOT / 'models/Qwen3-4B-Base-GRPO'),
                (ml2.ROOT / 'data/math_opd_dapo17k_hf_full_eval4', ROOT / 'data/math_opd_dapo17k_hf_full_eval4'),
                (ml2.base_pair.QUALIFICATION, IMPORT_ROOT / 'qualification'),
                (ml2.ROOT / 'deployments' / SOURCE_COMMIT, REFERENCE),
                (ml2.ROOT / 'runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/grading',
                 ROOT / 'runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/grading'))
    for old, new in mappings:
        if source.is_relative_to(old):
            return new / source.relative_to(old)
    raise ValueError(f'Unapproved source path: {source}')


def verify_transfer():
    path = IMPORT_ROOT / 'migration_manifest.json'
    manifest = read_json(path)
    expected = dict(schema_version=1, source_host=SOURCE_HOST, source_run=str(SOURCE_RUN),
                    source_commit=SOURCE_COMMIT, hardware=HARDWARE,
                    source_success=read_json(IMPORT_ROOT / 'block3_mean/training_state_acceptance.json'))
    if any(manifest.get(k) != v for k, v in expected.items()) or not manifest.get('created_at'):
        raise ValueError('Wrong migration source, hardware or success metadata')
    records = manifest.get('files')
    if not isinstance(records, list) or not records:
        raise ValueError('Empty migration manifest')
    receipt = read_json(IMPORT_ROOT / 'transfer_acceptance.json')
    if (receipt.get('passed') is not True or receipt.get('manifest_sha256') != sha256(path)
            or receipt.get('files_verified') != len(records)):
        raise ValueError('Transfer receipt does not bind the manifest')
    protected, mapping, total = {}, {}, 0
    for record in records:
        source, destination = Path(record['source']), Path(record['destination'])
        digest, size = record['sha256'], record['size_bytes']
        if (not source.is_absolute() or '..' in source.parts or '..' in destination.parts
                or destination != destination_for(source) or not destination.is_relative_to(ROOT)
                or destination.resolve() != destination or not destination.is_file()
                or str(source) in mapping or str(destination) in protected
                or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
                or type(size) is not int or size < 0):
            raise ValueError(f'Invalid or duplicate transfer record: {source}')
        if destination.stat().st_size != size or sha256(destination) != digest:
            raise ValueError(f'Hash/size mismatch: {destination}')
        protected[str(destination)] = digest
        mapping[str(source)] = destination
        total += size
    if receipt.get('total_bytes') != total:
        raise ValueError('Transfer receipt byte count differs')
    for name in ('migration_manifest.json', 'transfer_acceptance.json'):
        protected[str(IMPORT_ROOT / name)] = sha256(IMPORT_ROOT / name)
    return protected, mapping


def link_import(root):
    (root / 'block3_mean').symlink_to(IMPORT_ROOT / 'block3_mean', target_is_directory=True)


def require_coverage(run, protected):
    required = {IMPORT_ROOT / name for name in ('input_plan.json', 'environment.json',
        'baseline_alignment.json', 'queue_state.json', 'qualification/selected.json',
        'qualification/pair_summary.json')}
    for step in STEPS:
        checkpoint = IMPORT_ROOT / 'block3_mean/checkpoints' / f'global_step_{step}'
        required.add(checkpoint / 'data.pt')
        required.update(checkpoint / 'actor' / name for name in (
            'config.json', 'generation_config.json', 'tokenizer.json',
            'tokenizer_config.json', 'merges.txt', 'vocab.json'))
        required.update(checkpoint / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt'
                        for kind in ('model', 'optim', 'extra_state') for rank in range(4))
    for folder in (IMPORT_ROOT / 'block3_mean', run.q.STUDENT, run.q.TEACHER):
        if not folder.is_dir():
            raise ValueError(f'Missing imported directory: {folder}')
        required.update(p for p in folder.rglob('*') if p.is_file())
    required.update(run.q.base.DATA / name for name in ('train.parquet', 'test.parquet', 'manifest.json'))
    required.update(run.q.base.DATA / 'eval_jsonl' / f'{task}.jsonl' for task in run.q.shared.TASK_COUNTS)
    required.add(run.q.base.GRADER)
    missing = sorted(str(p) for p in required if str(p) not in protected or not p.is_file())
    if missing:
        raise ValueError(f'Missing manifest coverage: {missing}')


def validate_source_evidence():
    folder = IMPORT_ROOT / 'block3_mean'
    state = read_json(IMPORT_ROOT / 'queue_state.json')
    if (state.get('status') != 'failed'
            or state.get('job') != str(SOURCE_RUN / 'queue_jobs/block3_mean_figures')
            or (folder / 'exit_code.txt').read_text().strip() != '0'):
        raise ValueError('Source must have completed training and failed only at figures')
    accepted = read_json(folder / 'training_state_acceptance.json')
    expected_states = [dict(step=s, rank=r, rng_scheduler_sampler_valid=True)
                       for s in sorted(STEPS) for r in range(4)]
    if (accepted.get('passed') is not True or accepted.get('checkpoint_steps') != sorted(STEPS)
            or accepted.get('diagnostic_steps') != [1, *range(5, 101, 5)]
            or accepted.get('states') != expected_states):
        raise ValueError('Source four-checkpoint training-state acceptance incomplete')
    for step in STEPS:
        inspected = read_json(folder / f'optimizer_inspection_step{step}.json')
        ranks = inspected.get('ranks', [])
        if (inspected.get('passed') is not True or [r.get('rank') for r in ranks] != list(range(4))
                or any(r.get('states', 0) <= 0 or r.get('nonzero_moment_elements', 0) <= 0 for r in ranks)):
            raise ValueError(f'Source optimizer acceptance incomplete: {step}')
    rollouts = read_json(folder / 'rollout_acceptance.json')
    if not isinstance(rollouts.get('steps'), list):
        raise ValueError('Source rollout acceptance incomplete')
    acp.n1.validate_paired_rollouts(rollouts, rollouts)


def verify_frozen_source(runtime):
    if (REFERENCE / 'DEPLOYED_COMMIT').read_text().strip() != SOURCE_COMMIT:
        raise ValueError('Wrong frozen source deployment')
    names = set(FROZEN_SCRIPTS)
    for tree in ('opd_ext', 'external/revisiting_opd'):
        def files(root):
            return {p.relative_to(root).as_posix() for p in (root / tree).rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.sh', '.yaml', '.yml', '.json')
                    and '.git' not in p.parts}
        reference, actual = files(REFERENCE), files(runtime)
        if not reference or reference != actual:
            raise ValueError(f'Frozen algorithm source inventory changed: {tree}')
        names.update(reference)
    protected = {}
    for name in sorted(names):
        old, new = REFERENCE / name, runtime / name
        if not old.is_file() or not new.is_file() or sha256(old) != sha256(new):
            raise ValueError(f'Frozen algorithm/evaluation source differs: {name}')
        protected[str(old)] = protected[str(new)] = sha256(old)
    return protected


def runtime_versions():
    expected = ml2.base_pair.previous.qualify.EXPECTED_VERSIONS
    versions = {name: importlib.metadata.version(name) for name in expected}
    normalized = dict(versions)
    if normalized.get('torch') == '2.8.0+cu128':
        normalized['torch'] = '2.8.0'
    if normalized != expected:
        raise ValueError(f'Environment primary package drift: {versions}')
    return versions


def validate_environment(source, current):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    old, new = dict(source['packages']), dict(current['packages'])
    if new.get('torch') == '2.8.0+cu128':
        new['torch'] = '2.8.0'
    if old != new or old != ml2.base_pair.previous.qualify.EXPECTED_VERSIONS:
        raise ValueError('Environment primary packages differ')
    old_python, new_python = source['python'].split()[0], current['python'].split()[0]
    if old_python != '3.12.13' or new_python not in ('3.12.13', '3.12.3'):
        raise ValueError('Environment Python change is not approved')

    def versions(freeze):
        result = {}
        for line in freeze.splitlines():
            if not line or line.startswith(('#', '-')):
                continue
            requirement = Requirement(line)
            result[canonicalize_name(requirement.name)] = str(requirement.specifier) or requirement.url
        return result

    old_freeze, new_freeze = versions(source['pip_freeze']), versions(current['pip_freeze'])
    math_packages = {'numpy', 'scipy', 'sympy', 'mpmath', 'antlr4-python3-runtime',
                     'latex2sympy2-extended', 'math-verify', 'mathruler', 'pylatexenc',
                     'datasets', 'pyarrow', 'pandas', 'omegaconf', 'hydra-core', 'ray', 'torchdata'}
    changes = {k: [old_freeze.get(k), new_freeze.get(k)] for k in math_packages
               if old_freeze.get(k) != new_freeze.get(k)}
    if changes:
        raise ValueError(f'Environment math/data/runtime dependencies differ: {changes}')
    return dict(passed=True, same_hardware=False, source=source, current=current,
                python_patch_difference_recorded=old_python != new_python,
                torch_cuda_build_difference_recorded=source['packages']['torch'] != current['packages']['torch'],
                matched_math_data_packages=sorted(math_packages & old_freeze.keys()), limitation=LIMITATION)


def audit_import(run, root, plan):
    folder = IMPORT_ROOT / 'block3_mean'
    view = root / 'import_audit/block3_mean'
    view.mkdir(parents=True, exist_ok=False)
    # The shared audit writes optimizer reports; only its inputs point into the import.
    for name in ('checkpoints', 'diagnostics', 'exit_code.txt'):
        (view / name).symlink_to(folder / name, target_is_directory=(folder / name).is_dir())
    states = run.n1.audit_training(run.q, view, sorted(STEPS), (1, *range(5, 101, 5)))
    rollouts = run.audit_rollouts(folder, plan, range(1, 101))
    if states != read_json(folder / 'training_state_acceptance.json'):
        raise ValueError('Imported training-state audit differs from source')
    if rollouts != read_json(folder / 'rollout_acceptance.json'):
        raise ValueError('Imported rollouts differ from source/data schedule')
    run.q.jobs.write_json(root / 'imported_block3_acceptance.json', dict(passed=True,
        training=states, rollouts=rollouts, imported_from=str(folder), no_source_writes=True))


@contextmanager
def claim_attempt(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allowed = {'queue.lock', 'logs', 'wait_state.json', 'wait-state.json',
                   'migration_wait.lock', 'migration_wait_state.json',
                   'transfer_wait_state.json', 'queue_wait_state.json'}
        unexpected = [p.name for p in root.iterdir()
                      if p.name not in allowed and not (p.is_file() and p.suffix in ('.log', '.pid'))]
        if unexpected:
            raise FileExistsError(f'Existing attempt/output; no automatic restart: {unexpected}')
        yield


def plotting_gate(run, root, runner):
    runner([run.q.base.PYTHON, '-c',
            "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot; import numpy"],
           root / 'queue_jobs/plot_imports', job_env={'CUDA_VISIBLE_DEVICES': ''})


def validate_hardware(gpus, cpu_max):
    quota, period = cpu_max.split()
    if (len(gpus) != 4 or any('H100' not in g['name'] or g['memory_mib'] < 75000 for g in gpus)
            or quota == 'max' or int(period) <= 0 or int(quota) != 32 * int(period)):
        raise ValueError('Only four H100-80GB GPUs and a 32-CPU quota are authorized')


def preflight(run, runtime):
    q = run.q
    if os.environ.get('RAY_ADDRESS') or os.environ.get('RAY_TMPDIR'):
        raise ValueError('Unexpected inherited Ray environment')
    protected, mapping = verify_transfer()
    require_coverage(run, protected)
    validate_source_evidence()
    source_card = read_json(IMPORT_ROOT / 'block3_mean/run_card.json')
    source_runner = ml2.configured_runner()
    source_runner.configure()
    run.require_equal_card(source_runner.expected_card(
        REFERENCE, SOURCE_COMMIT, SOURCE_RUN, 'block3_mean'), source_card)
    if source_card.get('val_n') != 1 or source_card.get('resume_mode') != 'disable':
        raise ValueError('Source validation or initialization configuration drift')
    alignment = read_json(IMPORT_ROOT / 'baseline_alignment.json')
    if (alignment.get('passed') is not True or alignment.get('same_input_plan') is not True
            or alignment.get('same_model_runtime_bytes') is not True
            or alignment.get('new_training_and_evaluation_protocol') != run.PROTOCOL
            or alignment.get('request_seed_rule') != 'legacy' or alignment.get('seed') != 21):
        raise ValueError('Source baseline alignment is not accepted')
    artifacts = q.base.paired.load_sha256_manifest(IMPORT_ROOT / 'block3_mean/artifact_hashes.sha256')
    if not artifacts:
        raise ValueError('Source artifact manifest is empty')
    for digest, source in artifacts:
        destination = mapping.get(str(source))
        if destination is None or protected.get(str(destination)) != digest:
            raise ValueError(f'Source model/data identity differs: {source}')
    qualifier = ml2.base_pair.qualification.configured_qualifier()
    for key, path in (('b17', q.STUDENT), ('g4', q.TEACHER)):
        files = ml2.model_runtime_files(qualifier.assets.specifications()[key])
        protected.update(qualifier.assets.verify_files(path, files))
    selected = read_json(IMPORT_ROOT / 'qualification/selected.json')
    if qualifier.old.object_hash(selected) != qualifier.prior.SOURCE_SELECTION_SHA:
        raise ValueError('Pinned qualification selection changed')
    summary = read_json(IMPORT_ROOT / 'qualification/pair_summary.json')
    ml2.base_pair.validate_screen(summary, sha256(IMPORT_ROOT / 'qualification/pair_summary.json'))
    protected.update(verify_frozen_source(runtime))
    protected.update(q.common_preflight(runtime, {}))
    freeze = subprocess.run([str(q.base.PYTHON), '-m', 'pip', 'freeze'],
                            check=True, capture_output=True, text=True).stdout
    current = dict(python=sys.version, packages=runtime_versions(), pip_freeze=freeze,
                   host=HOST, hardware=HARDWARE, ray_num_cpus=32, venv=str(VENV))
    environment = validate_environment(read_json(IMPORT_ROOT / 'environment.json'), current)
    gpu_query = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                '--format=csv,noheader,nounits'],
                               check=True, capture_output=True, text=True).stdout
    gpus = [dict(name=row.rsplit(',', 1)[0].strip(), memory_mib=int(row.rsplit(',', 1)[1]))
            for row in gpu_query.splitlines()]
    cpu_max = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    validate_hardware(gpus, cpu_max)
    environment.update(gpus=gpus, cpu_max=cpu_max)
    q.jobs.write_json(RUN_ROOT / 'environment.json', current)
    q.jobs.write_json(RUN_ROOT / 'migration_alignment.json', environment)
    # Full transfer hashes and the state audit cover optimizers once; never prune them.
    return {path: digest for path, digest in protected.items()
            if not Path(path).name.startswith('optim_world_size_4_rank_')}


def cpu_contract(run, root, protected):
    q = run.q
    plan = run.n1.input_plan(q)
    if plan != read_json(IMPORT_ROOT / 'input_plan.json'):
        raise ValueError('Source physical-row data schedule differs')
    inputs = q.input_contract()
    required = dict(passed=True, actual_collector_exercised=True, protocol=run.PROTOCOL,
                    chat_template_used=False, enable_thinking=False, eos_token_id=151643,
                    stop_token_ids=[], questions_checked=dict(qualification=64, **q.shared.TASK_COUNTS))
    if any(inputs.get(key) != value for key, value in required.items()):
        raise ValueError('Actual 707-input completion contract failed')
    grading = run.grading_gate()
    if grading.get('passed') is not True:
        raise ValueError('Historical grader gate failed')
    q.jobs.write_json(root / 'input_contract.json', inputs)
    q.jobs.write_json(root / 'historical_grading_acceptance.json', grading)
    q.jobs.write_json(root / 'input_plan.json', plan)
    protected[str(root / 'input_plan.json')] = sha256(root / 'input_plan.json')
    audit_import(run, root, plan)
    link_import(root)
    return plan


def run_queue(run, root, runtime, commit, runner, protected, plan):
    q, results = run.q, {}

    def evaluate(name):
        result = q.evaluate_model(root, name, runtime, commit, runner, protected)
        if result.get('passed') is not True:
            raise ValueError(f'Full evaluation acceptance failed: {name}')
        results[name] = result
        q.jobs.write_json(root / 'evaluation_acceptance.json', dict(passed=True, complete=len(results) == 8, models=results))
        q.jobs.write_json(root / 'per_benchmark_results.json', {
            task: {n: r['per_task'][task] for n, r in results.items()} for task in q.shared.TASK_COUNTS})
        return result

    def train(variant):
        for label in ('student', 'teacher'):
            runner([q.base.PYTHON, ENTRYPOINT, '--gpu-smoke', label],
                   root / f'queue_jobs/gpu_smoke_{label}', gpu=True, job_env={'CUDA_VISIBLE_DEVICES': '0'})
            if read_json(root / f'gpu_smoke/{label}/acceptance.json').get('passed') is not True:
                raise ValueError('Base completion GPU smoke failed')
        runner(['bash', runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
               root / 'queue_jobs/prepare_token_opd_formal',
               job_env=run.training_env(runtime, commit, root, variant))
        card = read_json(root / variant / 'run_card.json')
        run.require_equal_card(run.expected_card(runtime, commit, root, variant), card)
        validate_migrated_card(read_json(IMPORT_ROOT / 'block3_mean/run_card.json'), card)
        for name in ('run_card.json', 'command.sh', 'artifact_hashes.sha256', 'script_hashes.sha256'):
            path = root / variant / name
            protected[str(path)] = sha256(path)
        q.jobs.write_json(root / 'protected_inputs.json', protected)
        return run.train_variant(root, variant, runtime, commit, runner, protected, plan)

    execute_ordered(train, evaluate)
    runner([q.base.PYTHON, runtime / 'scripts/analyze_single_opd_diagnostics.py',
            '--run-dir', IMPORT_ROOT / 'block3_mean', '--output-dir', root / 'figures/imported_block3',
            '--label', 'Imported ml2 Base1.7 Block3 seed21'],
           root / 'queue_jobs/imported_block3_figures', job_env={'CUDA_VISIBLE_DEVICES': ''})
    run.write_comparison(q, root, results)
    q.shared.verify_hashes(protected)
    return results


def model_paths(root, name):
    if name not in {f'{v}_step{s}' for v in ('block3_mean', 'token_opd') for s in STEPS}:
        raise ValueError('Only the eight approved milestone evaluations are allowed')
    variant, step = name.rsplit('_step', 1)
    parent = IMPORT_ROOT if variant == 'block3_mean' else root
    return parent / variant / f'checkpoints/global_step_{step}/actor', root / 'merged' / name


def validate_migrated_card(source, target):
    allowed = {'source_commit', 'variant', 'opd_block_size', 'opd_block_advantage_mode',
               'student_model', 'teacher_model', 'train_data', 'val_data', 'venv', 'ray_num_cpus',
               'project_name', 'experiment_name', 'diagnostic_output_dir', 'lossless_rollout_dir',
               'baseline_alignment'}
    differences = {k: [source.get(k), target.get(k)] for k in (source.keys() | target.keys()) - allowed
                   if k not in source or k not in target or source[k] != target[k]}
    if differences:
        raise ValueError(f'Unapproved training configuration drift: {differences}')


def training_env(previous, runtime, commit, root, variant, probe=None):
    if variant != 'token_opd':
        raise ValueError('Imported Block3 must never be prepared or trained')
    env = previous(runtime, commit, root, variant, probe)
    env.update(REMOTE='acp', RAY_NUM_CPUS='32', VENV=str(VENV), HF_HOME_DIR=str(ROOT / 'cache/hf'))
    return env


def expected_card(previous, runtime, commit, root, variant, probe=None):
    if variant != 'token_opd':
        raise ValueError('Only independent Token training is authorized')
    card = previous(runtime, commit, root, variant, probe)
    card.update(ray_num_cpus=32, venv=str(VENV), val_n=1)
    return card


def train_variant(previous, root, variant, runtime, commit, runner, protected, plan):
    if variant != 'token_opd':
        raise ValueError('Imported Block3 must never enter the training helper')
    return previous(root, variant, runtime, commit, runner, protected, plan)


def configure(run, previous):
    previous()
    q = run.q
    q.base.VENV, q.base.PYTHON = VENV, VENV / 'bin/python'
    q.base.PLOT_PYTHON = q.base.PYTHON
    q.base.DATA = ROOT / 'data/math_opd_dapo17k_hf_full_eval4'
    q.base.GRADER = ROOT / 'runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/grading/historical_utils_sha04f7.py'
    q.LOSS_REFERENCE = REFERENCE
    q.QUALIFICATION = IMPORT_ROOT / 'qualification'
    q.BASELINE_ALIGNMENT = LIMITATION
    q.model_paths = model_paths


def configured_runner():
    private = ml2.base_pair.qualification.private_module
    protocol = private('_acp_migration_protocol', ml2.__file__)
    protocol.base_pair = private('_acp_migration_base_pair', ml2.base_pair.__file__)
    protocol.base_pair.QUALIFICATION = IMPORT_ROOT / 'qualification'
    protocol.ROOT, protocol.RUN_ROOT = ROOT, RUN_ROOT
    protocol.STUDENT = ROOT / 'models/Qwen3-1.7B-Base'
    protocol.TEACHER = ROOT / 'models/Qwen3-4B-Base-GRPO'
    protocol.PROJECT = 'opd_acp_base17_migration'
    run = protocol.configured_runner()
    run.q.qualify = private('_acp_migration_qualify', run.q.qualify.__file__)
    run.q.qualify.runtime_versions = runtime_versions
    run.ROOT, run.RUN_ROOT, run.HOST = ROOT, RUN_ROOT, HOST
    run.VENV, run.CACHE, run.ENTRYPOINT = VENV, CACHE, ENTRYPOINT
    run.HARDWARE, run.LIFETIME = acp.HARDWARE, acp.LIFETIME
    run.DISPLAY_PREFIX = 'ACP migrated Base1.7 GRPO4 n1'
    run.LOSS_OVERRIDES = {}
    run.configure = partial(configure, run, run.configure)
    run.validate_location = acp.validate_location
    run.training_env = partial(training_env, run.training_env)
    run.expected_card = partial(expected_card, run.expected_card)
    run.train_variant = partial(train_variant, run.train_variant)
    run.execute_ordered = execute_ordered
    run.write_comparison = acp.n1.write_comparison
    run.preflight = partial(preflight, run)
    run.main = partial(main, run)
    return run


def validate_interpreter(prefix, executable):
    if Path(prefix) != VENV or Path(executable).parent != VENV / 'bin':
        raise ValueError('Controller interpreter must use the isolated migration venv')


def deployment(run):
    validate_interpreter(sys.prefix, sys.executable)
    mount = subprocess.run(['findmnt', '-T', str(ROOT), '-n', '-o', 'FSTYPE'],
                           check=True, capture_output=True, text=True).stdout.strip()
    run.validate_location(socket.gethostname(), ROOT, mount)
    runtime = ENTRYPOINT.parents[1]
    commit = (runtime / 'DEPLOYED_COMMIT').read_text().strip()
    if (re.fullmatch('[0-9a-f]{40}', commit) is None
            or runtime != ROOT / 'deployments' / commit or runtime.resolve() != runtime):
        raise ValueError('An immutable ACP deployment is required')
    sys.path.insert(0, str(runtime / 'external/revisiting_opd'))
    return runtime, commit


def controller_env(run, runtime):
    if shutil.disk_usage(ROOT).free < 1_000_000_000_000:
        raise ValueError('Less than 1TB free; never prune checkpoints')
    from recover_historical17_llama import check_socket_budget
    CACHE.mkdir(exist_ok=True)
    local = subprocess.run(['findmnt', '-T', str(CACHE), '-n', '-o', 'FSTYPE,OPTIONS'],
                           check=True, capture_output=True, text=True).stdout.strip().split(maxsplit=1)
    run.q.validate_cache_mount(*local, shutil.disk_usage(CACHE).free)
    for suffix in ('token_opd/train/tmp', 't/gate/tmp'):
        check_socket_budget(CACHE / suffix)
    env = dict(os.environ, PATH=f'{VENV}/bin:' + os.environ.get('PATH', ''),
               PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(run.q.base.GRADER),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MPLBACKEND='Agg')
    for key, suffix in dict(TMPDIR='tmp', VLLM_CACHE_ROOT='vllm', TRITON_CACHE_DIR='triton',
                            TORCHINDUCTOR_CACHE_DIR='inductor', CUDA_CACHE_PATH='cuda',
                            OUTLINES_CACHE_DIR='outlines', MPLCONFIGDIR='matplotlib', HF_HOME='hf').items():
        (CACHE / suffix).mkdir(exist_ok=True)
        env[key] = str(CACHE / suffix)
    return env


def main(run=None, argv=None):
    run = run or configured_runner()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray-gate', choices=('token_opd',))
    parser.add_argument('--gpu-smoke', choices=('student', 'teacher'))
    args = parser.parse_args(argv)
    run.configure()
    runtime, commit = deployment(run)
    root, q = RUN_ROOT, run.q
    if args.ray_gate or args.gpu_smoke:
        manifest = read_json(root / 'queue_manifest.json')
        if manifest.get('source_commit') != commit or manifest.get('imported_source_commit') != SOURCE_COMMIT:
            raise ValueError('GPU/Ray helpers require this migration attempt')
        if args.ray_gate:
            import recover_qwen17_instruct_token as warmup
            warmup.CACHE = CACHE / 't'
            warmup.ray_gate(root / 'token_opd_ray_gate.json')
        else:
            run.gpu_smoke(root, args.gpu_smoke)
        return

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    old_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        with claim_attempt(root):
            state = root / 'queue_state.json'
            q.jobs.write_json(root / 'queue_manifest.json', dict(
                source_commit=commit, imported_source_commit=SOURCE_COMMIT, source_run=str(SOURCE_RUN),
                import_root=str(IMPORT_ROOT), host=HOST, hardware=HARDWARE, limitation=LIMITATION,
                created_at=q.jobs.now(), total_training_steps=100, train_batch_size=32, rollout_group_size=1,
                seed=21, request_seed_rule='legacy', prompt_protocol=run.PROTOCOL, ray_num_cpus=32,
                checkpoint_steps=sorted(STEPS), evaluation_steps=STEPS, initial_student_reevaluated=False,
                block3_training=False, independent_original_initialization=True,
                retain_all_checkpoints=True, retain_all_rollouts=True, no_automatic_retries=True,
                ordering='Imported Block3 eval100,75,50,25 -> Token save1/resume2/train100/eval100,75,50,25',
                environment=str(VENV), lifetime=run.LIFETIME))
            (root / 'queue.pid').write_text(str(os.getpid()) + '\n')
            try:
                env = controller_env(run, runtime)
                q.jobs.wait_for_idle()

                def runner(command, job, job_env=None, **kwargs):
                    if job.resolve().is_relative_to(IMPORT_ROOT.resolve()):
                        raise ValueError('No jobs may write into the immutable import')
                    try:
                        q.jobs.run_job(command, job, runtime, state, {**env, **(job_env or {})}, **kwargs)
                    except BaseException:
                        pidfile = job / kwargs.get('pid_name', 'job.pid')
                        if kwargs.get('gpu') and pidfile.exists():
                            q.cleanup_failed_group(int(pidfile.read_text()))
                        raise

                runner(['sha256sum', '-c', '.expected.sha256'], root / 'queue_jobs/runtime_hashes')
                plotting_gate(run, root, runner)
                protected = run.preflight(runtime)
                plan = cpu_contract(run, root, protected)
                q.jobs.write_json(root / 'protected_inputs.json', protected)
                results = run_queue(run, root, runtime, commit, runner, protected, plan)
                report = read_json(root / 'paired_comparison.json')
                report.update(evaluation_protocol=run.PROTOCOL, same_hardware=False,
                              hardware=HARDWARE, limitation=LIMITATION)
                q.jobs.write_json(root / 'paired_comparison.json', report)
                q.jobs.write_json(state, dict(status='complete', updated_at=q.jobs.now(), models=list(results)))
            except BaseException as error:
                previous = read_json(state) if state.exists() else {}
                q.jobs.write_json(state, dict(previous, status='failed', error=repr(error), updated_at=q.jobs.now()))
                raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == '__main__':
    main()

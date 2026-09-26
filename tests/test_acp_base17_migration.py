import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    path = ROOT / 'scripts/run_acp_base17_migration.py'
    assert path.exists(), 'Missing bounded ACP migration controller'
    spec = importlib.util.spec_from_file_location('acp_migration_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_imported_evaluations_precede_token_and_never_train_block3(module):
    calls = []
    results = module.execute_ordered(
        lambda v: calls.append(('train', v)) or {'passed': True},
        lambda n: calls.append(('eval', n)) or {'passed': True})
    expected = [('eval', f'block3_mean_step{s}') for s in (100, 75, 50, 25)]
    expected += [('train', 'token_opd')]
    expected += [('eval', f'token_opd_step{s}') for s in (100, 75, 50, 25)]
    assert calls == expected
    assert len(results) == 8 and 'student_base' not in results


@pytest.mark.parametrize('failure', range(9))
def test_every_failure_stops_without_retry(module, failure):
    calls = []

    def action(name):
        calls.append(name)
        return {'passed': len(calls) != failure + 1}

    with pytest.raises(ValueError, match='acceptance'):
        module.execute_ordered(action, action)
    assert len(calls) == failure + 1


@pytest.mark.parametrize('probe', [None, 1, 2])
def test_only_token_can_be_prepared_or_trained(module, probe):
    run = module.configured_runner()
    run.configure()
    runtime, commit = Path('/runtime'), 'a' * 40
    env = run.training_env(runtime, commit, module.RUN_ROOT, 'token_opd', probe)
    card = run.expected_card(runtime, commit, module.RUN_ROOT, 'token_opd', probe)
    assert env['REMOTE'] == 'acp' and env['RAY_NUM_CPUS'] == '32'
    assert env['VENV'] == str(module.VENV)
    assert module.VENV == module.ROOT / 'envs/verl-cu128-base17-migration-v1'
    assert env['STUDENT_MODEL'] == str(module.ROOT / 'models/Qwen3-1.7B-Base')
    assert env['MATH_TEACHER'] == str(module.ROOT / 'models/Qwen3-4B-Base-GRPO')
    assert env['OPD_PROMPT_PROTOCOL'] == 'qwen3_completion_boxed_v1'
    assert env['OPD_REQUEST_SEED_RULE'] == 'legacy'
    assert env['TRAIN_BATCH_SIZE'] == '32' and env['ROLLOUT_GROUP_SIZE'] == '1'
    assert env['PPO_MINI_BATCH_SIZE'] == '32' and env['LEARNING_RATE'] == '2e-6'
    assert env['TOTAL_TRAINING_STEPS'] == '100' and env['ENV_SEED'] == '21'
    assert env['STOP_AFTER_STEP'] == str(probe or -1)
    assert env['RESUME_MODE'] == ('resume_path' if probe == 2 else 'disable')
    assert env['RESUME_FROM_PATH'] == (str(module.RUN_ROOT / 'token_opd/checkpoints/global_step_1') if probe == 2 else '')
    assert card['ray_num_cpus'] == 32 and card['opd_block_ablation'] == 'legacy'
    assert card['venv'] == str(module.VENV) and card['val_n'] == 1
    assert card['opd_block_size'] == 1 and card['opd_block_advantage_mode'] == 'sum'
    assert card['diagnostic_save_steps'] == {None: '25,50,75,100', 1: '1', 2: '1,2'}[probe]
    for variant in ('block3_mean', 'student_base', 'adv3'):
        with pytest.raises(ValueError):
            run.training_env(runtime, commit, module.RUN_ROOT, variant, probe)
        with pytest.raises(ValueError):
            run.train_variant(module.RUN_ROOT, variant, runtime, commit, None, {}, {})


def test_model_paths_override_import_not_source_card(module):
    run = module.configured_runner()
    run.configure()
    assert run.q.VARIANTS == ('block3_mean', 'token_opd')
    assert run.q.input_contract.func.__globals__['QUALIFICATION'] == module.IMPORT_ROOT / 'qualification'
    for variant in run.q.VARIANTS:
        for step in (100, 75, 50, 25):
            actor, merged = run.q.model_paths(module.RUN_ROOT, f'{variant}_step{step}')
            parent = module.IMPORT_ROOT if variant == 'block3_mean' else module.RUN_ROOT
            assert actor == parent / variant / f'checkpoints/global_step_{step}/actor'
            assert merged == module.RUN_ROOT / 'merged' / f'{variant}_step{step}'
            cmd = run.q.evaluation_command(Path('/runtime'), merged, Path('/out'))
            for flag, value in [('--n', '8'), ('--eval-seed', '21'), ('--grader', 'external'),
                                ('--prompt-protocol', 'qwen3_completion_boxed_v1')]:
                assert cmd[cmd.index(flag) + 1] == value
            assert '--retain-rollouts' in cmd
    with pytest.raises(ValueError):
        run.q.model_paths(module.RUN_ROOT, 'student_base')


def test_host_mount_guard(module):
    run = module.configured_runner()
    run.validate_location(module.HOST, module.ROOT, 'fuse.quarkfs_client')
    for args in [('ml2', module.ROOT, 'fuse.quarkfs_client'),
                 (module.HOST, Path('/workspace'), 'fuse.quarkfs_client'),
                 (module.HOST, module.ROOT, 'nfs4')]:
        with pytest.raises(ValueError):
            run.validate_location(*args)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def transfer(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path / 'imports/test')
    destination = module.IMPORT_ROOT / 'block3_mean/run_card.json'
    write_json(destination, {'source_commit': module.SOURCE_COMMIT})
    record = dict(source=str(module.SOURCE_RUN / 'block3_mean/run_card.json'),
                  destination=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  size_bytes=destination.stat().st_size)
    source_evidence(module, module.IMPORT_ROOT)
    source_success = json.loads((module.IMPORT_ROOT / 'block3_mean/training_state_acceptance.json').read_text())
    manifest = dict(schema_version=1, source_host=module.SOURCE_HOST,
                    source_run=str(module.SOURCE_RUN), source_commit=module.SOURCE_COMMIT,
                    created_at='2026-09-27T00:00:00+00:00', files=[record],
                    hardware=module.HARDWARE, source_success=source_success)

    def seal(value=manifest):
        path = module.IMPORT_ROOT / 'migration_manifest.json'
        write_json(path, value)
        write_json(module.IMPORT_ROOT / 'transfer_acceptance.json', dict(passed=True,
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            files_verified=len(value['files']), total_bytes=sum(r['size_bytes'] for r in value['files'])))

    seal()
    return manifest, destination, seal


def test_manifest_rehashes_destination_and_binds_receipt(module, transfer):
    manifest, destination, _ = transfer
    protected, mapping = module.verify_transfer()
    assert protected[str(destination)] == manifest['files'][0]['sha256']
    assert mapping[manifest['files'][0]['source']] == destination
    destination.write_text('changed')
    with pytest.raises(ValueError, match='[Hh]ash|[Ss]ize'):
        module.verify_transfer()


@pytest.mark.parametrize('change', ['commit', 'host', 'source', 'escape', 'duplicate', 'empty', 'size', 'digest', 'receipt', 'source_success'])
def test_invalid_transfer_fails_closed(module, transfer, change):
    manifest, destination, seal = transfer
    value = copy.deepcopy(manifest)
    if change == 'commit': value['source_commit'] = 'a' * 40
    if change == 'host': value['source_host'] = 'other'
    if change == 'source': value['files'][0]['source'] = '/unrelated/file'
    if change == 'escape': value['files'][0]['destination'] = '/tmp/escape'
    if change == 'duplicate': value['files'] *= 2
    if change == 'empty': value['files'] = []
    if change == 'size': value['files'][0]['size_bytes'] += 1
    if change == 'digest': value['files'][0]['sha256'] = 'z' * 64
    if change == 'source_success': value['source_success'] = {'passed': True}
    seal(value)
    if change == 'receipt':
        write_json(module.IMPORT_ROOT / 'transfer_acceptance.json', {'passed': True})
    with pytest.raises(ValueError):
        module.verify_transfer()


def test_link_never_overwrites_import_or_existing_run(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path / 'imports')
    source = module.IMPORT_ROOT / 'block3_mean'
    source.mkdir(parents=True)
    root = tmp_path / 'run'
    root.mkdir()
    module.link_import(root)
    assert (root / 'block3_mean').is_symlink()
    assert (root / 'block3_mean').resolve() == source
    with pytest.raises(FileExistsError):
        module.link_import(root)


def test_card_comparison_allows_only_named_deployment_and_arm_differences(module):
    source = json.loads((ROOT / 'results/ml2_base17_protocol_n1_20260926/paired_preflight.json').read_text())['cards']['block3_mean']
    run = module.configured_runner()
    run.configure()
    target = dict(source)
    target.update(run.expected_card(Path('/runtime'), 'a' * 40, module.RUN_ROOT, 'token_opd'))
    module.validate_migrated_card(source, target)
    for key in ('learning_rate', 'seed', 'request_seed_rule', 'opd_prompt_protocol',
                'resume_mode', 'max_response_length', 'opd_block_ablation', 'val_n',
                'checkpoint_policy', 'window_supervision_sha256'):
        with pytest.raises(ValueError, match='configuration'):
            module.validate_migrated_card(source, dict(target, **{key: 'drift'}))
    with pytest.raises(ValueError, match='configuration'):
        module.validate_migrated_card(source, dict(target, unexpected_math_flag=True))


def source_evidence(module, root):
    folder = root / 'block3_mean'
    write_json(root / 'queue_state.json', dict(status='failed', job=str(module.SOURCE_RUN / 'queue_jobs/block3_mean_figures')))
    write_json(folder / 'training_state_acceptance.json', dict(passed=True,
        checkpoint_steps=[25, 50, 75, 100], diagnostic_steps=[1, *range(5, 101, 5)],
        states=[dict(step=s, rank=r, rng_scheduler_sampler_valid=True)
                for s in (25, 50, 75, 100) for r in range(4)]))
    write_json(folder / 'rollout_acceptance.json', dict(passed=True, total_rollouts=3200,
        steps=[dict(step=s, count=32, paired_input_sha256=f'{s:064x}') for s in range(1, 101)]))
    for step in (25, 50, 75, 100):
        write_json(folder / f'optimizer_inspection_step{step}.json', dict(passed=True,
            ranks=[dict(rank=r, states=1, nonzero_moment_elements=1) for r in range(4)]))
    (folder / 'exit_code.txt').write_text('0\n')


@pytest.mark.parametrize('change', [None, 'exit', 'queue', 'checkpoint', 'rank', 'optimizer', 'rollout'])
def test_source_finished_block3_all_acceptances_required(module, monkeypatch, tmp_path, change):
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path)
    source_evidence(module, tmp_path)
    folder = tmp_path / 'block3_mean'
    if change == 'exit': (folder / 'exit_code.txt').write_text('1')
    if change == 'queue': write_json(tmp_path / 'queue_state.json', {'status': 'running'})
    if change in ('checkpoint', 'rank'):
        path = folder / 'training_state_acceptance.json'
        value = json.loads(path.read_text())
        value['checkpoint_steps' if change == 'checkpoint' else 'states'].pop()
        write_json(path, value)
    if change == 'optimizer': write_json(folder / 'optimizer_inspection_step50.json', {'passed': False})
    if change == 'rollout': write_json(folder / 'rollout_acceptance.json', {'passed': True, 'total_rollouts': 64})
    if change is None:
        module.validate_source_evidence()
    else:
        with pytest.raises(ValueError):
            module.validate_source_evidence()


def test_required_coverage_includes_weights_states_diagnostics_and_assets(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path / 'import')
    folder = module.IMPORT_ROOT / 'block3_mean'
    (folder / 'checkpoints').mkdir(parents=True)
    (folder / 'checkpoints/weight.pt').write_bytes(b'weight')
    run = module.configured_runner()
    run.configure()
    with pytest.raises(ValueError, match='[Cc]overage|[Mm]issing'):
        module.require_coverage(run, {})


@pytest.mark.parametrize('step', [25, 50, 75, 100])
def test_jointly_omitted_checkpoint_files_rejected_before_evaluation(module, monkeypatch, tmp_path, step):
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path / 'import')
    run = module.configured_runner()
    run.configure()
    run.q.STUDENT, run.q.TEACHER = tmp_path / 'student', tmp_path / 'teacher'
    run.q.base.DATA, run.q.base.GRADER = tmp_path / 'data', tmp_path / 'grader.py'
    names = ['actor/' + name for name in ('config.json', 'generation_config.json',
        'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')]
    names += [f'actor/{kind}_world_size_4_rank_{rank}.pt'
              for kind in ('model', 'optim', 'extra_state') for rank in range(4)]
    names += ['data.pt']
    required = [module.IMPORT_ROOT / name for name in ('input_plan.json', 'environment.json',
        'baseline_alignment.json', 'queue_state.json', 'qualification/selected.json',
        'qualification/pair_summary.json')]
    required += [module.IMPORT_ROOT / 'block3_mean/checkpoints' / f'global_step_{s}' / name
                 for s in (25, 50, 75, 100) for name in names]
    required += [run.q.STUDENT / 'config.json', run.q.TEACHER / 'config.json', run.q.base.GRADER]
    required += [run.q.base.DATA / name for name in ('train.parquet', 'test.parquet', 'manifest.json')]
    required += [run.q.base.DATA / 'eval_jsonl' / f'{task}.jsonl' for task in run.q.shared.TASK_COUNTS]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    protected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    module.require_coverage(run, protected)
    for name in names:
        missing = module.IMPORT_ROOT / 'block3_mean/checkpoints' / f'global_step_{step}' / name
        digest = protected.pop(str(missing))
        missing.unlink()
        with pytest.raises(ValueError, match='Missing manifest coverage'):
            module.require_coverage(run, protected)
        missing.write_bytes(b'fixture')
        protected[str(missing)] = digest


def test_actual_training_shell_is_frozen(module, monkeypatch, tmp_path):
    reference, runtime = tmp_path / 'reference', tmp_path / 'runtime'
    monkeypatch.setattr(module, 'REFERENCE', reference)
    shell = 'scripts/run_revisiting_sampled_block_opd_math.sh'
    names = {*module.FROZEN_SCRIPTS, shell, 'opd_ext/loss.py', 'external/revisiting_opd/sampler.py'}
    for root in (reference, runtime):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('unchanged')
        (root / 'DEPLOYED_COMMIT').write_text(module.SOURCE_COMMIT)
    module.verify_frozen_source(runtime)
    (runtime / shell).write_text('changed training configuration')
    with pytest.raises(ValueError, match='source differs'):
        module.verify_frozen_source(runtime)


def test_frozen_source_rejects_algorithm_or_eval_changes(module, monkeypatch, tmp_path):
    reference, runtime = tmp_path / 'reference', tmp_path / 'runtime'
    monkeypatch.setattr(module, 'REFERENCE', reference)
    monkeypatch.setattr(module, 'FROZEN_SCRIPTS', ('scripts/eval.py',))
    for root in (reference, runtime):
        for name in ('scripts/eval.py', 'opd_ext/loss.py', 'external/revisiting_opd/sampler.py'):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('unchanged')
        (root / 'DEPLOYED_COMMIT').write_text(module.SOURCE_COMMIT)
    protected = module.verify_frozen_source(runtime)
    assert str(runtime / 'opd_ext/loss.py') in protected
    (runtime / 'scripts/eval.py').write_text('changed')
    with pytest.raises(ValueError, match='source'):
        module.verify_frozen_source(runtime)


def test_environment_only_explicit_patch_and_cuda_build_differences(module):
    source = dict(python='3.12.13 source build', packages=dict(torch='2.8.0', vllm='0.11.0',
        transformers='4.57.6', tokenizers='0.22.2'), pip_freeze='sympy==1.14.0\nnumpy==1.26.4\n')
    current = dict(python='3.12.3 ACP build', packages=dict(source['packages'], torch='2.8.0+cu128'),
                   pip_freeze=source['pip_freeze'])
    report = module.validate_environment(source, current)
    assert report['same_hardware'] is False and report['python_patch_difference_recorded'] is True
    for changed in (dict(current, packages=dict(current['packages'], vllm='0.12.0')),
                    dict(current, python='3.13.3'),
                    dict(current, pip_freeze='sympy==1.15.0\nnumpy==1.26.4\n')):
        with pytest.raises(ValueError, match='[Ee]nvironment'):
            module.validate_environment(source, changed)


def test_audit_view_writes_only_new_run(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path / 'import')
    source_evidence(module, module.IMPORT_ROOT)
    folder = module.IMPORT_ROOT / 'block3_mean'
    (folder / 'checkpoints').mkdir()
    (folder / 'diagnostics').mkdir()
    source_hashes = {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    run = module.configured_runner()
    plan = {'sources': []}
    state = json.loads((folder / 'training_state_acceptance.json').read_text())
    rollouts = json.loads((folder / 'rollout_acceptance.json').read_text())

    def audit(q, view, checkpoints, diagnostics):
        assert view != folder and view.parent == tmp_path / 'run/import_audit'
        assert (view / 'checkpoints').resolve() == folder / 'checkpoints'
        write_json(view / 'optimizer_inspection_step25.json', {'passed': True})
        return state

    monkeypatch.setattr(run.n1, 'audit_training', audit)
    monkeypatch.setattr(run, 'audit_rollouts', lambda path, actual, steps: rollouts)
    root = tmp_path / 'run'
    root.mkdir()
    module.audit_import(run, root, plan)
    assert source_hashes == {p: p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def test_orchestration_prepares_only_token_after_four_evals(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path / 'import')
    source_evidence(module, module.IMPORT_ROOT)
    source_card = json.loads((ROOT / 'results/ml2_base17_protocol_n1_20260926/paired_preflight.json').read_text())['cards']['block3_mean']
    write_json(module.IMPORT_ROOT / 'block3_mean/run_card.json', source_card)
    run = module.configured_runner()
    run.configure()
    events = []
    runtime, commit = Path('/runtime'), 'a' * 40
    root = tmp_path / 'run'
    root.mkdir()
    monkeypatch.setattr(run.q.shared, 'verify_hashes', lambda _: None)
    monkeypatch.setattr(run, 'write_comparison', lambda *args: None)

    def runner(command, job, **kwargs):
        events.append(job.name)
        assert 'block3_mean' not in job.name or job.name == 'imported_block3_figures'
        if job.name == 'prepare_token_opd_formal':
            card = dict(source_card)
            card.update(run.expected_card(runtime, commit, root, 'token_opd'))
            write_json(root / 'token_opd/run_card.json', card)
            for name in ('command.sh', 'artifact_hashes.sha256', 'script_hashes.sha256'):
                (root / 'token_opd' / name).write_text('test')
        elif job.name.startswith('gpu_smoke_'):
            write_json(root / 'gpu_smoke' / job.name.removeprefix('gpu_smoke_') / 'acceptance.json', {'passed': True})

    def evaluate(root, name, *args):
        events.append(name)
        return {'passed': True, 'per_task': {task: {} for task in run.q.shared.TASK_COUNTS}}

    monkeypatch.setattr(run.q, 'evaluate_model', evaluate)
    monkeypatch.setattr(run, 'train_variant', lambda *args: events.append('token_gate_and_train') or {'passed': True})
    results = module.run_queue(run, root, runtime, commit, runner, {}, {})
    assert events[:4] == [f'block3_mean_step{s}' for s in (100, 75, 50, 25)]
    assert events.index('prepare_token_opd_formal') > 3
    assert events.index('token_gate_and_train') > events.index('prepare_token_opd_formal')
    assert len(results) == 8


def test_attempt_is_exclusive_and_refuses_any_existing_output(module, tmp_path):
    root = tmp_path / 'run'
    with module.claim_attempt(root):
        with pytest.raises((BlockingIOError, FileExistsError)):
            with module.claim_attempt(root):
                pytest.fail('Concurrent attempt accepted')
        (root / 'queue_state.json').write_text('failed')
    with pytest.raises(FileExistsError):
        with module.claim_attempt(root):
            pytest.fail('Old attempt accepted')


def test_plot_import_gate_is_cpu_only_and_before_queue(module):
    run = module.configured_runner()
    run.configure()
    calls = []
    module.plotting_gate(run, module.RUN_ROOT, lambda *args, **kwargs: calls.append((args, kwargs)))
    command = calls[0][0][0]
    assert command[0] == run.q.base.PYTHON and 'matplotlib' in command[-1]
    assert calls[0][1]['job_env']['CUDA_VISIBLE_DEVICES'] == ''


def test_attempt_accepts_outer_wait_logs_without_queue_manifest(module, tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    for name in ('nohup.log', 'wait.pid', 'wait_state.json', 'migration_wait.lock', 'migration_wait_state.json'):
        (root / name).write_text('{}')
    with module.claim_attempt(root):
        pass
    (root / 'queue_manifest.json').write_text('{}')
    with pytest.raises(FileExistsError):
        with module.claim_attempt(root):
            pytest.fail('Accepted previous queue')


def test_preflight_binds_source_artifacts_plan_environment_and_input_contract(module, monkeypatch, tmp_path):
    assert hasattr(module, 'preflight'), 'Missing fail-closed migration preflight'
    run = module.configured_runner()
    run.configure()
    assert run.LOSS_OVERRIDES == {}
    assert run.q.LOSS_REFERENCE == module.REFERENCE
    assert run.preflight.func is module.preflight
    assert run.q.qualify.runtime_versions is module.runtime_versions


@pytest.mark.parametrize('failure', ['plan', 'input', None])
def test_cpu_contract_stops_before_import_link_or_gpu(module, monkeypatch, tmp_path, failure):
    run = module.configured_runner()
    monkeypatch.setattr(module, 'IMPORT_ROOT', tmp_path / 'imports')
    root = tmp_path / 'run'
    root.mkdir()
    source = {'seed': 21, 'indices': [1, 2]}
    write_json(module.IMPORT_ROOT / 'input_plan.json', source)
    monkeypatch.setattr(run.n1, 'input_plan', lambda q: {} if failure == 'plan' else source)
    accepted = dict(passed=True, actual_collector_exercised=True, eos_token_id=151643,
                    chat_template_used=False, enable_thinking=False, stop_token_ids=[],
                    protocol=run.PROTOCOL,
                    questions_checked=dict(qualification=64, math500=500, aime24=30, aime25=30, amc23=83))
    if failure == 'input': accepted['questions_checked']['qualification'] = 63
    monkeypatch.setattr(run.q, 'input_contract', lambda: accepted)
    monkeypatch.setattr(run, 'grading_gate', lambda: {'passed': True})
    calls = []
    monkeypatch.setattr(module, 'audit_import', lambda *args: calls.append('audit'))
    monkeypatch.setattr(module, 'link_import', lambda *args: calls.append('link'))
    if failure:
        with pytest.raises(ValueError):
            module.cpu_contract(run, root, {})
        assert calls == []
    else:
        assert module.cpu_contract(run, root, {}) == source
        assert calls == ['audit', 'link']


def test_hardware_requires_verified_h100_and_32cpu_quota(module):
    expected = [{'name': 'NVIDIA H100 80GB HBM3', 'memory_mib': 81559}] * 4
    module.validate_hardware(expected, '3200000 100000')
    for cards, quota in [(expected[:3], '3200000 100000'),
                         ([dict(name='A100', memory_mib=81559)] * 4, '3200000 100000'),
                         (expected, '6400000 100000'), (expected, 'max 100000')]:
        with pytest.raises(ValueError):
            module.validate_hardware(cards, quota)


def test_main_failure_records_terminal_failure_without_retry(module, monkeypatch, tmp_path):
    assert hasattr(module, 'main'), 'Missing executable queue entry'
    run = module.configured_runner()
    root = tmp_path / 'run'
    monkeypatch.setattr(module, 'RUN_ROOT', root)
    runtime = tmp_path / 'runtime'
    monkeypatch.setattr(module, 'deployment', lambda *args: (runtime, 'a' * 40))
    monkeypatch.setattr(module, 'controller_env', lambda *args: {})
    monkeypatch.setattr(run.q.jobs, 'wait_for_idle', lambda: None)
    calls = []

    def fail(*args, **kwargs):
        calls.append('runtime_hashes')
        raise RuntimeError('stop here')

    monkeypatch.setattr(run.q.jobs, 'run_job', fail)
    with pytest.raises(RuntimeError, match='stop here'):
        module.main(run, [])
    state = json.loads((root / 'queue_state.json').read_text())
    assert state['status'] == 'failed' and 'stop here' in state['error']
    assert (root / 'queue_manifest.json').exists() and calls == ['runtime_hashes']
    with pytest.raises(FileExistsError):
        module.main(run, [])
    assert calls == ['runtime_hashes']


def test_controller_must_use_the_validated_migration_interpreter(module):
    module.validate_interpreter(str(module.VENV), str(module.VENV / 'bin/python'))
    for prefix, executable in [(str(module.ROOT / 'envs/verl-cu128-v1'), '/usr/bin/python3'),
                               (str(module.VENV), '/usr/bin/python3')]:
        with pytest.raises(ValueError, match='interpreter'):
            module.validate_interpreter(prefix, executable)


@pytest.mark.parametrize('fail_probe', [None, 1, 2])
def test_actual_shared_token_helper_gates_formal_training(module, monkeypatch, tmp_path, fail_probe):
    run = module.configured_runner()
    run.configure()
    root, runtime, commit = tmp_path, Path('/runtime'), 'a' * 40
    events = []
    folder = root / 'token_opd'
    write_json(folder / 'run_card.json', run.expected_card(runtime, commit, root, 'token_opd'))
    write_json(root / 'block3_mean/rollout_acceptance.json', {'passed': True})

    def runner(command, job, **kwargs):
        events.append(job.name)
        if job.name == 'token_opd_ray_gate':
            write_json(root / 'token_opd_ray_gate.json', {'passed': True})
        if job.name.startswith('prepare_token_opd_probe'):
            step = int(job.name[-1])
            probe = root / 'probes/token_opd'
            write_json(probe / 'run_card.json', run.expected_card(runtime, commit, root / 'probes', 'token_opd', step))
            (probe / 'logs').mkdir(exist_ok=True)
        if job.name in ('token_opd_probe1', 'token_opd_probe2'):
            if job.name == f'token_opd_probe{fail_probe}':
                raise RuntimeError('probe failed')
            (job / 'logs').mkdir(parents=True)
            (job / 'logs/job.log').write_text('passed')

    monkeypatch.setattr(run.n1, 'audit_training', lambda *args: {'passed': True})
    monkeypatch.setattr(run, 'audit_rollouts', lambda *args, **kwargs: {'passed': True})
    monkeypatch.setattr(run.n1, 'validate_paired_rollouts', lambda *args: {'passed': True})
    monkeypatch.setattr(run.q.base, 'audit_resume', lambda *args: events.append('resume_verified'))
    monkeypatch.setattr(run.q.shared, 'verify_hashes', lambda *args: None)
    if fail_probe:
        with pytest.raises(RuntimeError, match='probe failed'):
            run.train_variant(root, 'token_opd', runtime, commit, runner, {}, {})
        assert 'token_opd' not in events
    else:
        assert run.train_variant(root, 'token_opd', runtime, commit, runner, {}, {})['passed']
        assert events.index('token_opd_probe1') < events.index('token_opd_probe2')
        assert events.index('token_opd_probe2') < events.index('resume_verified') < events.index('token_opd')

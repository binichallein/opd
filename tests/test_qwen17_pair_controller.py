import importlib.util
from pathlib import Path
import sys
import types

import pytest

PATH = Path(__file__).parents[1] / 'scripts/run_qwen17_pair_diagnostics.py'


def module():
    spec = importlib.util.spec_from_file_location('qwen17_controller', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_queue_order_no_training_and_no_short_circuit_on_base_score():
    m = module()
    jobs = m.job_specs(Path('/release'))
    names = [name for name, _, _ in jobs]
    assert names == [
        'asset_base_student', 'asset_base_teacher', 'prepare_base',
        'base_direct_student', 'base_direct_teacher', 'base_continuation_student', 'base_continuation_teacher', 'summarize_base',
        'asset_instruct_student', 'asset_instruct_teacher', 'prepare_instruct',
        'instruct_smoke_student', 'instruct_smoke_teacher',
        'instruct_direct_student', 'instruct_direct_teacher',
        'instruct_continuation_student', 'instruct_continuation_teacher', 'summarize_instruct']
    for name, argv, gpu in jobs:
        assert not any('train' in str(x) or 'torchrun' in str(x) for x in argv)
        assert gpu == ('_direct_' in name or '_continuation_' in name or '_smoke_' in name)
    assert m.RUN_ROOT.name == '20260921v1_qwen8_to17_diagnostics_ml2'


def test_manifest_evaluation_only():
    m = module()
    manifest = m.queue_manifest('abc')
    assert manifest['training_authorized'] is False
    assert manifest['full_benchmark_eval'] is False
    assert manifest['pairs'] == ['base', 'instruct']
    assert manifest['expected_diagnostic_rollouts'] == 768
    assert manifest['expected_smoke_rollouts'] == 8


@pytest.mark.parametrize('failure', [False, True])
def test_controller_main_is_once_only_and_continues_after_inconclusive_base(tmp_path, monkeypatch, failure):
    m = module()
    monkeypatch.syspath_prepend(str(PATH.parent))
    import run_window_queue as jobs
    import qualify_qwen8_teacher as utility
    root = tmp_path / 'remote'
    control = root / 'analysis_deployments' / ('a' * 40)
    (control / 'scripts').mkdir(parents=True)
    (control / 'DEPLOYED_COMMIT').write_text('a' * 40)
    (root / 'runs').mkdir()
    run = root / 'runs' / 'new'
    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(m, 'RUN_ROOT', run)
    monkeypatch.setattr(m, 'CACHE', tmp_path / 'cache')
    monkeypatch.setattr(m, '__file__', str(control / 'scripts' / PATH.name))
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda path: types.SimpleNamespace(free=500_000_000_000))
    monkeypatch.setattr(m.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(jobs, 'wait_for_idle', lambda *args, **kwargs: None)
    monkeypatch.setattr(jobs.subprocess, 'run', lambda *args, **kwargs: types.SimpleNamespace(stdout='tmpfs rw'))
    cleanup = []
    monkeypatch.setitem(sys.modules, 'run_qwen4_completion_block3', types.SimpleNamespace(
        cleanup_failed_group=lambda pid: cleanup.append(pid), validate_cache_mount=lambda *args: None))
    monkeypatch.setitem(sys.modules, 'qualify_qwen17_pairs', types.SimpleNamespace(old=utility))
    calls = []
    def runner(argv, job, runtime, state, env, **kwargs):
        calls.append(job.name)
        assert runtime == control and env['VLLM_WORKER_MULTIPROC_METHOD'] == 'spawn'
        if failure and job.name == 'base_direct_student':
            job.mkdir(parents=True)
            (job / 'job.pid').write_text('123')
            raise RuntimeError('injected engine failure')
        if job.name.startswith('summarize_'):
            pair = job.name.removeprefix('summarize_')
            folder = run / 'qualification' / pair
            folder.mkdir(parents=True)
            utility.seal_json(folder / 'gate_acceptance.json',
                              {'training_authorized': False, 'passed': False, 'status': 'inconclusive'})
    monkeypatch.setattr(jobs, 'run_job', runner)
    if failure:
        with pytest.raises(RuntimeError, match='engine failure'):
            m.main()
        assert utility.read_json(run / 'queue_state.json')['status'] == 'failed'
        assert cleanup == [123] and 'asset_instruct_student' not in calls
    else:
        m.main()
        assert calls[-1] == 'summarize_instruct' and len(calls) == 19
        state = utility.read_json(run / 'queue_state.json')
        assert state['status'] == 'complete' and state['training_started'] is False
        assert state['pair_statuses'] == {'base': 'inconclusive', 'instruct': 'inconclusive'}
        before = (run / 'queue_manifest.json').read_bytes()
        with pytest.raises(FileExistsError):
            m.main()
        assert (run / 'queue_manifest.json').read_bytes() == before and len(calls) == 19

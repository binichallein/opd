import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def subject():
    assert importlib.util.find_spec('recover_qwen17_base_grpo') is not None
    return importlib.import_module('recover_qwen17_base_grpo')


def test_isolated_recovery_preserves_pair_and_recipe(tmp_path):
    m = subject()
    import run_qwen17_base_grpo_pair as old
    q = m.configured_runner()
    assert q.RUN_ROOT == m.RUN_ROOT != old.RUN_ROOT
    assert q.CACHE == m.CACHE != old.CACHE
    original = old.configured_runner()
    for variant in q.VARIANTS:
        a = q.training_env(tmp_path, 'abc', tmp_path, variant)
        b = original.training_env(tmp_path, 'abc', tmp_path, variant)
        assert {k:v for k,v in a.items() if k != 'LOCAL_CACHE_ROOT'} == {
            k:v for k,v in b.items() if k != 'LOCAL_CACHE_ROOT'}
    assert q.AUTHORIZATION == original.AUTHORIZATION
    assert q.EOS_TOKEN_ID == 151643


def test_relocation_rejects_experiment_changes(tmp_path):
    m = subject(); q = m.configured_runner()
    a = q.expected_card(m.SOURCE, 'block3_mean', m.SOURCE_COMMIT)
    b = q.expected_card(m.RUN_ROOT, 'block3_mean', 'new')
    m.validate_relocation(q, a, b, 'new')
    for key, value in [('learning_rate', 1.), ('seed', 7), ('opd_block_size', 10)]:
        with pytest.raises(ValueError):
            m.validate_relocation(q, a, {**b, key:value}, 'new')


def test_failure_gate_requires_no_formal_training(tmp_path):
    m = subject()
    run = tmp_path / 'block3_mean'; run.mkdir()
    (tmp_path/'queue_state.json').write_text(json.dumps({'status':'failed','job':str(run)}))
    (tmp_path/'queue_manifest.json').write_text(json.dumps({'source_commit':m.SOURCE_COMMIT}))
    incident = tmp_path/m.INCIDENT
    incident.mkdir(parents=True)
    (incident/'driver_stack.txt').write_text('RegisterClient CoreWorker')
    (incident/'ray_logs').mkdir()
    (incident/'ray_logs/raylet.err').write_text('have not registered within the timeout')
    m.validate_failure(tmp_path)
    (run/'raw.jsonl.gz').touch()
    with pytest.raises(ValueError): m.validate_failure(tmp_path)


def test_probe_gate_requires_all_ranks_and_two_steps():
    m = subject()
    good = {'passed':True,'states':[dict(step=s,rank=r,scheduler_step=s)
                                   for s in (1,2) for r in range(4)]}
    m.validate_resume_gate(good)
    with pytest.raises(ValueError): m.validate_resume_gate({**good,'states':good['states'][:-1]})
    with pytest.raises(ValueError): m.validate_resume_gate({**good,'passed':False})


def test_ray_settings_only_affect_worker_prestart():
    m = subject()
    assert m.RAY_ENV == {'RAY_prestart_worker_first_driver':'0','RAY_enable_worker_prestart':'0'}
    assert m.RAY_CONFIG == {'prestart_worker_first_driver':False,'enable_worker_prestart':False}


def test_real_sha_manifest_preserves_training_files(tmp_path):
    m = subject(); q = m.configured_runner()
    old, new = tmp_path/'old', tmp_path/'new'
    old.mkdir(); new.mkdir()
    (old/'script.py').write_text('frozen')
    (new/'script.py').write_text('frozen')
    manifest = tmp_path/'scripts.sha256'
    manifest.write_text(f"{q.assets.sha256(old/'script.py')}  {old/'script.py'}\n")
    m.verify_training_files(q,manifest,old,new)
    (new/'script.py').write_text('changed')
    with pytest.raises(ValueError): m.verify_training_files(q,manifest,old,new)


def test_token_keeps_full_original_probe_path(tmp_path, monkeypatch):
    m = subject(); q = m.configured_runner(); events = []
    monkeypatch.setattr(q.base,'read_json',lambda _: {'passed':True})
    def runner(cmd, path, **kw): events.append(('gate',kw))
    def original(*args): events.append(('train',args[1])); return {'passed':True}
    assert m.train_variant(q,original,tmp_path,'token_opd',tmp_path,'abc',runner,{})['passed']
    assert events[1] == ('train','token_opd')
    assert events[0][1]['job_env']['CUDA_VISIBLE_DEVICES'] == ''

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def test_launch_profile_exists():
    assert importlib.util.find_spec('run_qwen06_instruct_pair') is not None


@pytest.fixture
def subject():
    return importlib.import_module('run_qwen06_instruct_pair')


def test_profile_is_isolated_and_preserves_old_pair(subject):
    import run_qwen17_instruct_pair as old
    before = (old.STUDENT, old.TEACHER, old.RUN_ROOT, old.CACHE)
    q = subject.configured_runner()
    assert q is not old
    assert (old.STUDENT, old.TEACHER, old.RUN_ROOT, old.CACHE) == before
    assert q.STUDENT.name == 'Qwen3-0.6B'
    assert q.TEACHER.name == 'Qwen3-4B'
    assert q.STUDENT_REVISION == '09b42cad3d112e832108974449ccb5e8e0f5b5d1'
    assert q.TEACHER_REVISION == '2c54d5a09e7e92d4f5126b92a5a457448c9593e6'
    assert q.RUN_ROOT == subject.RUN_ROOT and q.CACHE == subject.CACHE
    assert q.CAPABILITY_SHA == subject.CAPABILITY_SHA


@pytest.mark.parametrize('variant', ['block3_mean', 'token_opd'])
@pytest.mark.parametrize('probe', [None, 1, 2])
def test_same_training_contract_only_pair_and_outputs_change(subject, tmp_path, variant, probe):
    import run_qwen17_instruct_pair as old
    q = subject.configured_runner()
    env = q.training_env(tmp_path/'runtime', 'abc', tmp_path, variant, probe)
    old_env = old.training_env(tmp_path/'runtime', 'abc', tmp_path, variant, probe)
    allowed = {'STUDENT_MODEL', 'MATH_TEACHER', 'STUDENT_MODEL_REVISION',
               'TEACHER_MODEL_REVISION', 'PROJECT_NAME', 'EXP_NAME', 'LOCAL_CACHE_ROOT'}
    assert {k: v for k, v in env.items() if k not in allowed} == {
        k: v for k, v in old_env.items() if k not in allowed}
    assert env['PROJECT_NAME'] == 'opd_qwen06_instruct'
    assert env['EXP_NAME'].startswith('qwen06-instruct-')
    assert env['TOTAL_TRAINING_STEPS'] == ('2' if probe else '200')
    assert env['DIAGNOSTIC_SAVE_STEPS'] == {None:'50,100,150,200', 1:'1', 2:'1,2'}[probe]
    assert env['RESUME_MODE'] == ('resume_path' if probe == 2 else 'disable')
    cards = {v: q.expected_card(tmp_path, v, 'abc') for v in q.VARIANTS}
    q.validate_pair(cards, tmp_path, 'abc')
    for name in q.VARIANTS:
        current = q.expected_card(tmp_path, name, 'abc', probe)
        previous = old.expected_card(tmp_path, name, 'abc', probe)
        changed = {k for k in current.keys() | previous.keys() if current.get(k) != previous.get(k)}
        assert changed == {'student_model', 'teacher_model', 'student_model_revision', 'teacher_model_revision'}


def test_order_and_full_eval_are_identical(subject, tmp_path):
    import run_qwen17_instruct_pair as old
    q = subject.configured_runner()
    events = []
    q.execute_ordered(lambda v: events.append(('train', v)) or {'passed': True},
                      lambda n: events.append(('eval', n)) or {'passed': True})
    assert events == [('train', 'block3_mean')] + [
        ('eval', f'block3_mean_step{s}') for s in (200,150,100,50)
    ] + [('eval','student_base'), ('train','token_opd')] + [
        ('eval', f'token_opd_step{s}') for s in (200,150,100,50)]
    old_cmd = old.evaluation_command(tmp_path, tmp_path/'model', tmp_path/'eval')
    new_cmd = q.evaluation_command(tmp_path, tmp_path/'model', tmp_path/'eval')
    assert [str(x).replace(str(q.STUDENT), str(old.STUDENT)) for x in new_cmd] == list(map(str,old_cmd))
    assert q.model_paths(tmp_path,'student_base') == (None, q.STUDENT)


def screen_evidence(subject):
    summary = json.loads((ROOT/'results/qwen06_teacher_screen_20260923/final/pair_summary.json').read_text())
    specs = subject.assets.specifications()
    prep = dict(protocol=subject.screen.protocol(), selection_sha256=subject.screen.prior.SOURCE_SELECTION_SHA,
                models={k: dict(repo=specs[k]['repo'], revision=specs[k]['revision'],
                    path=str(subject.ROOT/'models'/specs[k]['repo'].split('/')[1])) for k in ('i06','i4')})
    return summary, prep


def test_pinned_selected_teacher_gate(subject):
    s, m = screen_evidence(subject)
    subject.validate_screen(s, subject.CAPABILITY_SHA, m)
    for field, value in [('status','rejected'), ('passed',False), ('teacher','i8'), ('failures',['direct_ci'])]:
        bad = copy.deepcopy(s)
        bad['reports']['i4_i06'][field] = value
        with pytest.raises(ValueError):
            subject.validate_screen(bad, subject.CAPABILITY_SHA, m)
    with pytest.raises(ValueError):
        subject.validate_screen(s, 'wrong-hash', m)
    m['models']['i06']['revision'] = 'unreviewed'
    with pytest.raises(ValueError):
        subject.validate_screen(s, subject.CAPABILITY_SHA, m)


def test_real_smoke_schema_and_recipe_paths(subject):
    health = {'num_rollouts': 4, 'generated_think_tags': 0, 'length_stops': 4,
              'scope': '256-token native-input probe, not a truncation estimate for 16K formal diagnostics'}
    subject.validate_smoke(health)
    for bad in ({**health, 'num_rollouts': 3}, {**health, 'generated_think_tags': 1}):
        with pytest.raises(ValueError):
            subject.validate_smoke(bad)
    assert 'scripts/eval_qwen3_math_vllm.py' in subject.RECIPE_FILES
    assert all((ROOT/p).is_file() for p in subject.RECIPE_FILES)
    from recover_historical17_llama import check_socket_budget
    for variant in ('block3_mean', 'token_opd'):
        assert check_socket_budget(subject.CACHE/variant/'train/tmp') <= 107
        assert check_socket_budget(subject.RAY_CACHES[variant]/'gate/tmp') <= 107


def test_source_revision_marker_is_derived_without_overwrite(subject, tmp_path):
    revision = subject.assets.specifications()['i06']['revision']
    model = {'path': str(tmp_path), 'revision': revision}
    original = tmp_path/'model.safetensors'
    original.write_bytes(b'untouched weights')
    hashes = subject.ensure_revision_marker(model)
    marker = tmp_path/'SOURCE_REVISION'
    assert marker.read_text() == revision+'\n'
    assert original.read_bytes() == b'untouched weights'
    assert subject.ensure_revision_marker(model) == hashes
    marker.write_text('different revision\n')
    with pytest.raises(ValueError):
        subject.ensure_revision_marker(model)
    assert marker.read_text() == 'different revision\n'
    marker.unlink()
    marker.symlink_to(original)
    with pytest.raises(ValueError):
        subject.ensure_revision_marker(model)


def test_ray_warmup_precedes_each_arm_without_changing_training(subject, tmp_path, monkeypatch):
    import run_qwen17_instruct_pair as old
    calls = []
    def original(*args):
        calls.append(('train', args[1]))
        return {'passed': True}
    q = subject.configured_runner()
    monkeypatch.setattr(q.base, 'read_json', lambda p: {'passed': True})
    def runner(command, job, **kwargs):
        calls.append(('gate', list(map(str,command)), kwargs))
    result = subject.train_with_ray_gate(q, original, tmp_path, 'block3_mean', tmp_path/'runtime', 'abc', runner, {})
    assert result['passed']
    assert calls[0][0] == 'gate' and calls[1] == ('train','block3_mean')
    assert '--ray-gate-output' in calls[0][1]
    assert calls[0][2]['job_env']['CUDA_VISIBLE_DEVICES'] == ''
    assert old.STUDENT.name == 'Qwen3-1.7B'

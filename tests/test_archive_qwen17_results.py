import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def subject():
    path = Path(__file__).resolve().parents[1]/'scripts/archive_qwen17_results.py'
    assert path.exists(), 'Validated experiment archive exporter is missing'
    return importlib.import_module('archive_qwen17_results')


def accepted(s, names):
    model = {'passed':True, 'per_task':{k:{'num_examples':n,'num_rollouts':8*n,
        'avg_at_8':.5,'pass_at_8':.75} for k,n in s.TASKS.items()},
        'rollout_archive':{'passed':True,'num_rollouts':5144,'num_archive_files':32}}
    return {'passed':True,'complete':len(names)==9,'models':{n:model for n in names}}


def test_only_accepted_models_and_pending_remain_explicit():
    s = subject()
    names = list(s.EXPECTED - {'token_opd_step100','token_opd_step50'})
    d = accepted(s,names)
    s.validate_acceptance(d)
    assert s.pending_models(d) == ['token_opd_step100','token_opd_step50']
    for bad in [{**d,'complete':True},{**d,'passed':False}]:
        with pytest.raises(ValueError):s.validate_acceptance(bad)


def test_missing_task_or_failed_model_rejected():
    s = subject()
    d = accepted(s,['student_base'])
    d['models']['student_base']['per_task'].pop('aime25')
    with pytest.raises(ValueError):s.validate_acceptance(d)
    d = accepted(s,['student_base'])
    d['models']['student_base']['passed'] = False
    with pytest.raises(ValueError):s.validate_acceptance(d)


def test_remote_paths_only_map_to_expected_results(tmp_path):
    s = subject()
    remote = s.RECOVERY/'evaluations/token_opd_step200/outputs/summary.json'
    assert s.local_path(tmp_path,remote) == tmp_path/'evaluations/token_opd_step200/outputs/summary.json'
    assert s.local_path(tmp_path,s.DATA/'eval_jsonl/aime24.jsonl') == tmp_path/'eval_jsonl/aime24.jsonl'
    for bad in [s.RECOVERY/'merged/weights.safetensors',Path('/etc/passwd'),
                s.RECOVERY/'evaluations/../../secret']:
        with pytest.raises(ValueError):s.local_path(tmp_path,bad)


def test_hash_mismatch_is_not_accepted(tmp_path):
    s = subject()
    p = tmp_path/'a.json'
    p.write_text('{}')
    assert s.verify_file(p,s.sha256(p))['sha256'] == s.sha256(p)
    with pytest.raises(ValueError):s.verify_file(p,'0'*64)


def test_csv_preserves_separate_benchmarks_and_unrounded_values():
    s = subject()
    d = accepted(s,['student_base','token_opd_step200'])
    rows = s.result_rows(d)
    assert len(rows) == 8
    assert {r['benchmark'] for r in rows} == set(s.TASKS)
    assert all(r['avg_at_8'] == .5 for r in rows)
    assert not any('macro' in r for r in rows)


def test_final_snapshot_fetches_final_queue_and_comparison(monkeypatch, tmp_path):
    s = subject()
    commands = []
    monkeypatch.setattr(s.subprocess, 'run', lambda cmd, **kwargs: commands.append(cmd))
    s.fetch(tmp_path, accepted(s, s.EXPECTED))
    assert any(f'ml2:{s.RECOVERY}/queue_state.json' in cmd for cmd in commands)
    assert any(f'ml2:{s.RECOVERY}/paired_comparison.json' in cmd for cmd in commands)
    assert all(cmd[:3] == ['rsync','-az','--partial'] for cmd in commands)
    assert all('--delete' not in cmd for cmd in commands)


def test_partial_snapshot_never_fetches_unaccepted_eval(monkeypatch, tmp_path):
    s = subject()
    commands = []
    monkeypatch.setattr(s.subprocess, 'run', lambda cmd, **kwargs: commands.append(cmd))
    s.fetch(tmp_path, accepted(s, ['student_base']))
    evaluation_sources = [a for cmd in commands for a in cmd if a.startswith('ml2:') and '/evaluations/' in a]
    assert evaluation_sources == [f'ml2:{s.SOURCE}/evaluations/student_base']
    assert not any(f'ml2:{s.RECOVERY}/paired_comparison.json' in cmd for cmd in commands)


def test_fetch_never_overwrites_verified_snapshot(monkeypatch, tmp_path):
    s = subject()
    commands = []
    monkeypatch.setattr(s.subprocess, 'run', lambda cmd, **kwargs: commands.append(cmd))
    (tmp_path/'backup_acceptance.json').write_text('{"passed":true}')
    with pytest.raises(FileExistsError):
        s.fetch(tmp_path, accepted(s, s.EXPECTED))
    assert commands == []


def test_export_csv_uses_git_friendly_lf_without_changing_scores(tmp_path):
    s = subject()
    snapshot = tmp_path/'snapshot'
    output = tmp_path/'export'
    for arm in ('block3_mean', 'token_opd'):
        (snapshot/arm/'figures').mkdir(parents=True)
    acceptance = accepted(s, ['student_base'])
    (snapshot/'evaluation_acceptance.json').write_text(json.dumps(acceptance))
    s.export(snapshot, output, acceptance, {'verified_at':'test', 'training':{}})
    csv_bytes = (output/'metrics.csv').read_bytes()
    assert b'\r' not in csv_bytes
    assert len(list(s.csv.DictReader(csv_bytes.decode().splitlines()))) == 4

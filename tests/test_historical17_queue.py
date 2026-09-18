import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


def module():
    path = Path(__file__).parents[1] / 'scripts/run_historical17_reeval_llama.py'
    spec = importlib.util.spec_from_file_location('historical17_queue', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_six_original_checkpoints_and_no_interrupted_llama_weights():
    m = module()
    sources = m.historical_models()
    assert list(sources) == ['token_opd_step50', 'block3_mean_step50',
                             'token_opd_step100', 'block3_mean_step100',
                             'token_opd_step200', 'block3_mean_step200']
    for name, path in sources.items():
        variant, step = name.rsplit('_step', 1)
        assert path == m.HISTORICAL_RUNS[variant] / f'checkpoints/global_step_{step}/actor/huggingface'
        assert '20260918v1' not in str(path)


def test_historical_config_requires_original_sampling_and_data():
    m = module()
    model = next(iter(m.historical_models().values()))
    config = {**m.shared.EVAL_VALUES, 'grader': 'verl', 'model_path': str(model),
              'eval_jsonl_dir': str(m.base.DATA / 'eval_jsonl'), 'tasks': list(m.shared.TASK_COUNTS),
              'gpus': ['0', '1', '2', '3']}
    m.validate_historical_config(config, model)
    for key, bad in [('n', 1), ('eval_seed', 22), ('enable_thinking', True), ('max_tokens', 8192)]:
        with pytest.raises(ValueError):
            m.validate_historical_config({**config, key: bad}, model)


def test_incomplete_reevaluation_blocks_llama_dispatch():
    m = module()
    calls = []
    def evaluate(name, path):
        calls.append(name)
        return {'passed': len(calls) != 3}
    with pytest.raises(ValueError):
        m.execute_ordered(evaluate, lambda: calls.append('llama'))
    assert len(calls) == 3 and 'llama' not in calls


def test_all_six_evaluations_must_precede_llama():
    m = module()
    calls = []
    def evaluate(name, path):
        calls.append(name)
        return {'passed': True}
    results = m.execute_ordered(evaluate, lambda: calls.append('llama'))
    assert calls == list(m.historical_models()) + ['llama']
    assert len(results) == 6


def test_legacy_llama_training_uses_fresh_original_and_new_paths(tmp_path):
    m = module()
    for variant in m.llama.VARIANTS:
        env = m.llama.training_env(tmp_path, 'a' * 40, tmp_path / 'new', variant,
                                  training_protocol=m.LEGACY_PROTOCOL, cache=tmp_path / 'cache')
        assert env['OPD_PROMPT_PROTOCOL'] == m.LEGACY_PROTOCOL
        assert env['RESUME_MODE'] == 'disable' and env['RESUME_FROM_PATH'] == ''
        assert env['STUDENT_MODEL'].endswith('/Llama-3.2-1B-Instruct')
        assert env['LOCAL_CACHE_ROOT'] == str(tmp_path / 'cache/train')


def test_archive_command_is_opt_in_without_changing_eval_config(tmp_path):
    m = module()
    model = next(iter(m.historical_models().values()))
    cmd = m.evaluation_command(tmp_path, model, tmp_path / 'outputs')
    assert '--retain-rollouts' in cmd
    assert cmd[cmd.index('--grader') + 1] == 'external'
    assert cmd[cmd.index('--n') + 1] == '8'
    assert cmd[cmd.index('--eval-seed') + 1] == '21'
    assert '--enable-thinking' not in cmd


@pytest.mark.parametrize('protocol', ['llama32_nonthinking_v1', 'llama32_historical17_v1'])
def test_launcher_uses_modelscope_source_revisions_for_both_llama_protocols(tmp_path, protocol):
    root = tmp_path / 'runtime'
    (root / 'scripts').mkdir(parents=True)
    (root / 'scripts/run_revisiting_sampled_block_opd_math.sh').touch()
    student, teacher = tmp_path / 'student', tmp_path / 'teacher'
    for folder, revision in ((student, 'student-rev'), (teacher, 'teacher-rev')):
        folder.mkdir()
        (folder / 'SOURCE_REVISION').write_text(revision + '\n')
    binary = tmp_path / 'bin'
    binary.mkdir()
    ssh = binary / 'ssh'
    ssh.write_text('#!/bin/sh\nprintf "%s" "$2"\nexit 91\n')
    ssh.chmod(0o755)
    env = {**os.environ, 'PATH': f'{binary}:' + os.environ['PATH'], 'REMOTE': 'ml2',
           'REMOTE_ROOT': str(root), 'LAUNCH_TRANSPORT': 'ssh', 'PREPARE_ONLY': 'true',
           'STUDENT_MODEL': str(student), 'MATH_TEACHER': str(teacher), 'SOURCE_COMMIT': 'unknown',
           'STUDENT_MODEL_REVISION': 'student-rev', 'TEACHER_MODEL_REVISION': 'teacher-rev',
           'OPD_PROMPT_PROTOCOL': protocol}
    launcher = Path(__file__).parents[1] / 'scripts/launch_revisiting_block_opd_formal_train.sh'
    captured = subprocess.run(['bash', str(launcher)], env=env, text=True, capture_output=True)
    assert captured.returncode == 91, captured.stderr
    revision_checks = captured.stdout.split('test -x ', 1)[0]
    assert 'revision_file=' in revision_checks
    check = subprocess.run(['bash', '-c', revision_checks], text=True, capture_output=True)
    assert check.returncode == 0, check.stderr

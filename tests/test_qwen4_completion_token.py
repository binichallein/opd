import copy
import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def controller():
    assert importlib.util.find_spec('run_qwen4_completion_token') is not None
    return importlib.import_module('run_qwen4_completion_token')


def test_token_env_preserves_all_block3_scientific_settings():
    m = controller()
    left, right = m.formal.training_env(), m.training_env()
    changes = {k for k in set(left) | set(right) if left.get(k) != right.get(k)}
    assert changes == {'RUN_ROOT', 'VARIANT', 'EXP_NAME', 'LOSSLESS_ROLLOUT_DIR',
                       'OPD_DIAG_OUTPUT_DIR', 'LOCAL_CACHE_ROOT', 'BASELINE_ALIGNMENT'}
    assert right['VARIANT'] == 'token_opd'
    assert right['RESUME_MODE'] == 'disable' and right['RESUME_FROM_PATH'] == ''
    assert right['STUDENT_MODEL'] == str(m.gate.assets.STUDENT)
    assert right['DIAGNOSTIC_SAVE_STEPS'] == '50,100,150,200'
    assert right['TOTAL_TRAINING_STEPS'] == '200'
    assert right['OPD_DIAG_INTERVAL'] == '1' and right['ROLLOUT_ATTEMPT_ID'] == 'formal'
    assert right['LOCAL_CACHE_ROOT'] == str(m.CACHE / 'train')
    assert right['OPD_PROMPT_PROTOCOL'] == 'qwen3_completion_boxed_v1'
    assert right['OPD_REQUEST_SEED_RULE'] == 'sha256_step_question_sample_v1'
    assert right['REMOTE'] == 'ml2' and right['PREPARE_ONLY'] == 'true'


def test_token_card_changes_only_explicit_fields_from_completed_block3():
    m = controller()
    reference = {'variant': 'block3_mean', 'opd_block_size': 3,
                 'opd_block_advantage_mode': 'mean', 'seed': 21, 'learning_rate': 2e-6,
                 'source_commit': m.TRAIN_COMMIT, **m.formal.formal_card_overrides()}
    card = {**reference, **m.card_changes()}
    m.validate_card(card, reference)
    for key, wrong in [('seed', 7), ('learning_rate', 1e-6), ('opd_block_size', 3),
                       ('opd_block_advantage_mode', 'mean'), ('resume_mode', 'auto'),
                       ('diagnostic_save_steps', '50,100,200'), ('opd_diag_interval', 5)]:
        with pytest.raises(ValueError):
            m.validate_card({**card, key: wrong}, reference)


def ready_fixture(m, root):
    def write(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        m.jobs.write_json(path, obj)
    manifest = {'eval_steps': [200, 150, 100, 50], 'runtime_commit': m.TRAIN_COMMIT,
                'control_commit': m.EVAL_COMMIT, 'training_root': str(m.formal.RUN_ROOT),
                'prompt_protocol': m.gate.PROTOCOL, 'retain_rollouts': True}
    write(root / 'queue_manifest.json', manifest)
    write(root / 'queue_state.json', {'status': 'complete', 'evaluated_steps': [200, 150, 100, 50],
                                     'protected_inputs_verified': True})
    models = {}
    for step in (200, 150, 100, 50):
        tasks = {t: {'num_examples': n, 'num_rollouts': 8*n} for t, n in m.shared.TASK_COUNTS.items()}
        archive = {'passed': True, 'num_examples': 643, 'num_rollouts': 5144, 'num_archive_files': 32,
                   'per_task': {t: {**v, 'num_archive_files': 8} for t, v in tasks.items()}}
        a = {'passed': True, 'completion_protocol_verified': True, 'per_task': tasks,
             'model': str(root / 'merged' / f'block3_mean_step{step}'),
             'grader_sha256': m.shared.grading.HISTORICAL_GRADER_SHA256,
             'rollout_archive': archive, 'sha256': {}}
        folder = root / 'evaluations' / f'block3_mean_step{step}'
        write(folder / 'acceptance.json', a)
        (folder / 'exit_code.txt').write_text('0\n')
        models[str(step)] = a
    write(root / 'evaluation_acceptance.json', {'passed': True, 'complete': True,
                                               'completed_steps': [200, 150, 100, 50], 'models': models})


def test_predecessor_waits_for_all_four_accepted_checkpoints(tmp_path):
    m = controller()
    m.jobs.write_json(tmp_path / 'queue_state.json', {'status': 'running'})
    assert m.predecessor_ready(tmp_path) is False
    ready_fixture(m, tmp_path)
    assert m.predecessor_ready(tmp_path) is True


@pytest.mark.parametrize('mutation', ['failed', 'incomplete', 'missing50', 'wrong_prompt', 'wrong_count',
                                      'missing_archive', 'wrong_grader', 'nonzero_exit'])
def test_predecessor_rejects_failed_or_unverified_evaluation(tmp_path, mutation):
    m = controller()
    ready_fixture(m, tmp_path)
    if mutation == 'failed':
        m.jobs.write_json(tmp_path / 'queue_state.json', {'status': 'failed', 'error': '143'})
    elif mutation in ('incomplete', 'missing50'):
        p = tmp_path / 'evaluation_acceptance.json'
        a = m.base.read_json(p)
        if mutation == 'incomplete':
            a['complete'] = False
        else:
            del a['models']['50']
        m.jobs.write_json(p, a)
    elif mutation == 'wrong_prompt':
        p = tmp_path / 'queue_manifest.json'
        a = m.base.read_json(p)
        a['prompt_protocol'] = 'legacy'
        m.jobs.write_json(p, a)
    elif mutation == 'nonzero_exit':
        (tmp_path / 'evaluations/block3_mean_step50/exit_code.txt').write_text('1')
    else:
        p = tmp_path / 'evaluation_acceptance.json'
        a = m.base.read_json(p)
        model = a['models']['50']
        if mutation == 'wrong_count':
            model['per_task']['math500']['num_rollouts'] = 3999
        elif mutation == 'missing_archive':
            model['rollout_archive']['num_archive_files'] = 31
        else:
            model['grader_sha256'] = 'wrong'
        m.jobs.write_json(p, a)
        m.jobs.write_json(tmp_path / 'evaluations/block3_mean_step50/acceptance.json', model)
    with pytest.raises(ValueError):
        m.predecessor_ready(tmp_path)


def test_command_comparison_allows_paths_and_variant_not_science():
    m = controller()
    before = f"RUN_DIR='{m.formal.RUN}'\nexport TMPDIR='{m.formal.CACHE}/train/tmp'\nVARIANT='block3_mean' EXP_NAME='qwen4-completion-block3-mean' LEARNING_RATE='2e-6'\n"
    after = f"RUN_DIR='{m.RUN}'\nexport TMPDIR='{m.CACHE}/train/tmp'\nVARIANT='token_opd' EXP_NAME='qwen4-completion-token-opd' LEARNING_RATE='2e-6'\n"
    m.validate_command(after, before)
    with pytest.raises(ValueError):
        m.validate_command(after.replace("2e-6", "1e-6"), before)


def test_post_train_audit_requires_four_weights_and_every_step():
    m = controller()
    cmd = list(map(str, m.checkpoint_audit_command()))
    assert cmd[cmd.index('--variant')+1] == 'token_opd'
    assert cmd[cmd.index('--checkpoint-steps')+1] == '50,100,150,200'
    assert cmd[cmd.index('--expected-diag-interval')+1] == '1'
    assert cmd[cmd.index('--expected-diagnostic-steps')+1] == ','.join(map(str, range(1, 201)))
    assert '--skip-eval' in cmd


def test_paired_rollout_comparison_checks_prompt_seed_not_generated_text():
    m = controller()
    rows = [{'step': 1, 'source_extra_info': {'question': '2+2'},
             'prompt_token_ids': [1, 2], 'request_identity': f'id{i}',
             'sampling': {'seed': i}, 'protocol': m.gate.PROTOCOL, 'enable_thinking': False,
             'mask_policy': 'historical_eos_mask', 'eos_token_id': 151643,
             'request_turn': 0, 'response_tensor_width': 16384, 'response_token_ids': [3]}
            for i in range(32)]
    right = copy.deepcopy(rows)
    for row in right:
        row['response_token_ids'] = [4, 5]
    m.validate_paired_rows(rows, right, 1)
    right[0]['sampling']['seed'] = 999
    with pytest.raises(ValueError):
        m.validate_paired_rows(rows, right, 1)
    with pytest.raises(ValueError):
        m.validate_paired_rows(rows[:31], rows, 1)

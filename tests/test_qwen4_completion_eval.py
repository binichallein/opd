import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def controller():
    assert importlib.util.find_spec('run_qwen4_completion_eval') is not None
    return importlib.import_module('run_qwen4_completion_eval')


def test_order_is_four_completed_block_checkpoints_only():
    m = controller()
    assert m.STEPS == (200, 150, 100, 50)
    assert m.RUN_ROOT != m.TRAIN_ROOT
    for step in m.STEPS:
        actor, model = m.model_paths(step)
        assert actor == m.TRAIN_ROOT / f'block3_mean/checkpoints/global_step_{step}/actor'
        assert model == m.RUN_ROOT / f'merged/block3_mean_step{step}'
    with pytest.raises(ValueError):
        m.model_paths(1)


def test_eval_command_preserves_full_historical_sampling_but_uses_completion():
    m = controller()
    command = m.evaluation_command(Path('/model'), Path('/output'))
    expected = {'--model-path': '/model', '--output-dir': '/output', '--n': '8',
                '--max-tokens': '16384', '--eval-seed': '21', '--gpus': '0,1,2,3',
                '--temperature': '1.0', '--top-p': '0.9', '--grader': 'external',
                '--prompt-protocol': 'qwen3_completion_boxed_v1'}
    for flag, value in expected.items():
        assert command[command.index(flag)+1] == value
    assert command[command.index('--tasks')+1:command.index('--n')] == ['math500', 'aime24', 'aime25', 'amc23']
    assert '--retain-rollouts' in command
    assert '--replace' not in command and '--enable-thinking' not in command
    assert str(m.RUNTIME / 'scripts/eval_qwen3_math_vllm.py') == command[1]


def test_incomplete_or_wrong_training_is_rejected():
    m = controller()
    state = {'status': 'complete', 'checkpoint_steps': [50, 100, 150, 200]}
    card = {'source_commit': m.TRAIN_COMMIT, 'opd_prompt_protocol': m.PROTOCOL,
            'variant': 'block3_mean', 'student_model': str(m.assets.STUDENT),
            'opd_block_size': 3, 'opd_block_advantage_mode': 'mean'}
    m.validate_training(state, {'passed': True}, card)
    for wrong in [{'status': 'running', 'checkpoint_steps': state['checkpoint_steps']},
                  {'status': 'complete', 'checkpoint_steps': [50, 100, 200]}]:
        with pytest.raises(ValueError):
            m.validate_training(wrong, {'passed': True}, card)
    for key, value in [('source_commit', 'old'), ('opd_prompt_protocol', 'legacy'),
                       ('variant', 'token_opd'), ('opd_block_size', 1)]:
        with pytest.raises(ValueError):
            m.validate_training(state, {'passed': True}, {**card, key: value})


def test_actual_rollout_prompt_and_stop_contract():
    m = controller()
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()),
                                eos_token_id=151643, bos_token_id=None)
    question = '2+2?'
    row = {'problem': question, 'prompt_protocol': m.PROTOCOL, 'enable_thinking': False,
           'rendered_prompt': m.completion_math_prompt(question),
           'prompt_token_ids': m.completion_input_ids(tokenizer, question),
           'eos_token_id': 151643, 'sampling': {'stop_token_ids': []}}
    m.validate_completion_row(row, tokenizer)
    for key, value in [('prompt_protocol', 'legacy'), ('enable_thinking', True),
                       ('prompt_token_ids', [1]), ('rendered_prompt', 'old chat'),
                       ('sampling', {'stop_token_ids': [151645]}), ('eos_token_id', 151645)]:
        with pytest.raises(ValueError):
            m.validate_completion_row({**row, key: value}, tokenizer)


def test_runtime_cache_and_report_never_schedule_training():
    m = controller()
    assert str(m.CACHE).startswith('/dev/shm/')
    result = m.per_benchmark_results({200: {'per_task': {task: {'avg_at_8': .5} for task in m.shared.TASK_COUNTS}}})
    assert set(result) == set(m.shared.TASK_COUNTS)
    assert result['math500']['step200']['avg_at_8'] == .5
    assert all('macro' not in name for name in result)

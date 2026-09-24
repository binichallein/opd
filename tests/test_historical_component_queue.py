import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module():
    path = ROOT / 'scripts/run_historical_component_ablation.py'
    assert path.is_file(), 'Historical B/C queue is not implemented'
    spec = importlib.util.spec_from_file_location('historical_components_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize('variant,mode', [('adv3', 'adv_only'), ('joint3', 'joint_tokenmean')])
@pytest.mark.parametrize('probe', [None, 1, 2])
def test_old_protocol_and_full_state_contract(module, variant, mode, probe):
    run = module.configured_runner()
    env = run.training_env(Path('/runtime'), 'abc', Path('/run'), variant, probe)
    card = run.expected_card(Path('/run'), variant, 'abc', probe)
    assert env['VARIANT'] == variant
    assert env['STUDENT_MODEL'] == str(module.historical.QWEN_INITIAL)
    assert env['OPD_PROMPT_PROTOCOL'] == 'qwen3_historical17_v1'
    assert env['OPD_REQUEST_SEED_RULE'] == 'legacy'
    assert env['DIAGNOSTIC_SAVE_STEPS'] == {None:'50,100,150,200',1:'1',2:'1,2'}[probe]
    assert env['OPD_DIAG_INTERVAL'] == ('1' if probe else '5')
    assert env['LOSSLESS_ROLLOUT_DIR'] == f'/run/{variant}/rollouts'
    assert card['opd_block_ablation'] == mode
    assert card['opd_block_size'] == 3
    assert card['ray_num_cpus'] == 64
    assert card['val_n'] == 1
    assert card['resume_mode'] == ('resume_path' if probe == 2 else 'disable')
    if probe == 2:
        assert env['RESUME_FROM_PATH'] == f'/run/{variant}/checkpoints/global_step_1'
    assert run.EVALUATION_PROTOCOL == 'legacy'
    assert run.EXPECTED_EVAL_COUNT == 8


def test_exact_order_and_no_unrequested_a_d_training(module):
    calls = []
    module.execute_ordered(lambda v: calls.append(('train', v)) or {'passed':True},
        lambda n: calls.append(('eval', n)) or {'passed':True})
    assert calls == [(kind, name) for v in ('adv3', 'joint3')
        for kind, name in [('train', v), *[('eval', f'{v}_step{s}') for s in (200,150,100,50)]]]


def test_launchers_and_actor_route_ablation():
    for path in ('scripts/run_revisiting_sampled_block_opd_math.sh',
                 'scripts/launch_revisiting_block_opd_formal_train.sh'):
        source = (ROOT / path).read_text()
        assert 'adv3)' in source and 'joint3)' in source
        assert 'opd_block_ablation' in source
    source = (ROOT / 'external/revisiting_opd/verl/workers/actor/dp_actor.py').read_text()
    assert '**ablation_kwargs' in source


def test_pair_rejects_changed_seed_or_loss_mode(module):
    run = module.configured_runner()
    cards = {v:run.expected_card(Path('/run'), v, 'abc') for v in run.VARIANTS}
    run.validate_pair(cards, Path('/run'), 'abc')
    cards['joint3']['seed'] = 42
    with pytest.raises(ValueError):
        run.validate_pair(cards, Path('/run'), 'abc')


def test_eval_keeps_historical_grader_and_archive(module):
    run = module.configured_runner()
    cmd = list(map(str, run.evaluation_command(Path('/runtime'), Path('/model'), Path('/out'))))
    assert cmd[cmd.index('--grader')+1] == 'external'
    assert cmd[cmd.index('--prompt-protocol')+1] == 'legacy'
    assert '--retain-rollouts' in cmd


def test_historical_manifest_maps_paths_to_hashes(module, tmp_path):
    manifest = tmp_path / 'hashes'
    digest = 'a' * 64
    manifest.write_text(f'{digest}  /model/config.json\n')
    assert module.manifest_hashes(manifest) == {'/model/config.json': digest}

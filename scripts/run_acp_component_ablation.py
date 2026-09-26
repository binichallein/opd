#!/usr/bin/env python3
"""ACP loss-only ablations against the completed, immutable 100-step pair."""

from functools import partial
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import qualify_qwen17_base_grpo as modules
import run_acp_qwen17_n1 as original

BASELINE_ROOT = original.RUN_ROOT
BASELINE_COMMIT = '1867093b21799c36be22bb2fd167c87e5aaa039e'
RUN_ROOT = original.ROOT / 'runs/20260926v2_qwen8_to17_instruct_components_n1_seed21_acp'
MODES = {'adv3': 'adv_only', 'scale3': 'token_scale'}


def training_env(run, previous, runtime, commit, root, variant, probe=None):
    env = previous(runtime, commit, root, variant, probe)
    env.update(PROJECT_NAME='opd_acp_qwen17_components', EXP_NAME=f'acp-qwen17-{variant}-n1',
        BASELINE_ALIGNMENT='Loss-only ACP ablations against 1867093 Token/Block3; all other training conditions fixed')
    return env


def expected_card(previous, runtime, commit, root, variant, probe=None):
    card = previous(runtime, commit, root, variant, probe)
    card.update(project_name='opd_acp_qwen17_components', opd_block_size=3,
                opd_block_advantage_mode='mean', opd_block_ablation=MODES[variant])
    return card


def validate_baseline_card(expected, actual):
    allowed = {'source_commit', 'variant', 'project_name', 'experiment_name', 'opd_block_size',
               'opd_block_advantage_mode', 'opd_block_ablation', 'diagnostic_output_dir',
               'lossless_rollout_dir'}
    original.require_equal_card({k:v for k,v in expected.items() if k not in allowed}, actual)


def preflight(run, previous, runtime):
    protected = previous(runtime)
    q = run.q
    state = q.base.read_json(BASELINE_ROOT / 'queue_state.json')
    accepted = q.base.read_json(BASELINE_ROOT / 'evaluation_acceptance.json')
    if state.get('status') != 'complete' or not accepted.get('complete') or len(accepted.get('models', {})) != 8:
        raise ValueError('Completed baseline pair with eight full evaluations required')
    paths = [BASELINE_ROOT / name for name in ('queue_manifest.json', 'queue_state.json',
        'evaluation_acceptance.json', 'per_benchmark_results.json', 'input_plan.json',
        'paired_rollout_acceptance.json')]
    expected = run.expected_card(runtime, runtime.name, RUN_ROOT, 'adv3')
    environment_path = BASELINE_ROOT / 'final_environment.json'
    environment = q.base.read_json(environment_path)
    freeze = subprocess.run([str(q.base.PYTHON), '-m', 'pip', 'freeze'],
                            check=True, capture_output=True, text=True).stdout
    if sys.version != environment['python'] or set(freeze.splitlines()) != set(environment['pip_freeze'].splitlines()):
        raise ValueError('Baseline training environment changed')
    paths.append(environment_path)
    for variant in ('block3_mean', 'token_opd'):
        card = q.base.read_json(BASELINE_ROOT / variant / 'run_card.json')
        if card['source_commit'] != BASELINE_COMMIT:
            raise ValueError('Wrong historical control source')
        validate_baseline_card(expected, card)
        paths += [BASELINE_ROOT / variant / name for name in
                  ('run_card.json', 'rollout_acceptance.json', 'training_state_acceptance.json',
                   'artifact_hashes.sha256')]
    # Sampler equality is independent of model outputs and checked before any GPU work.
    plan = run.n1.input_plan(q)
    if plan != q.base.read_json(BASELINE_ROOT / 'input_plan.json'):
        raise ValueError('Baseline and ablation prompt schedules differ')
    for name in ('scripts/eval_math_batched.py', 'opd_ext/math_protocol.py',
                 'opd_ext/request_seeds.py', 'scripts/run_window_queue.py'):
        reference = run.ROOT / 'deployments' / BASELINE_COMMIT / name
        if q.assets.sha256(runtime / name) != q.assets.sha256(reference):
            raise ValueError(f'Evaluation or sampling code changed: {name}')
    protected.update({str(p):q.assets.sha256(p) for p in paths})
    q.jobs.write_json(RUN_ROOT / 'baseline_alignment.json', dict(passed=True,
        baseline_root=str(BASELINE_ROOT), baseline_commit=BASELINE_COMMIT,
        same_hardware_environment=True, same_input_plan=True,
        new_methods=MODES, no_baseline_retraining=True,
        limitations=['Single training seed; sequential runs do not guarantee bitwise replay.',
                    'adv3 changes credit sharing only; scale3 changes policy-loss scale only.',
                    'Legacy Block3 also changes the PPO ratio; these two arms are not a complete factorial.']))
    return protected


def execute_ordered(train, evaluate):
    results = {}
    for variant in MODES:
        if train(variant).get('passed') is not True:
            raise ValueError(f'Training acceptance failed: {variant}')
        for step in original.STEPS:
            name = f'{variant}_step{step}'
            results[name] = evaluate(name)
            if results[name].get('passed') is not True:
                raise ValueError(f'Full evaluation acceptance failed: {name}')
    return results


def write_comparison(q, root, results):
    if set(results) != {f'{v}_step{s}' for v in MODES for s in original.STEPS}:
        raise ValueError('Eight complete ablation evaluations required')
    baselines = q.base.read_json(BASELINE_ROOT / 'per_benchmark_results.json')
    report = dict(training_protocol=original.PROTOCOL, evaluation_protocol=original.PROTOCOL,
                  training_seed=21, primary_checkpoint=100, baseline_root=str(BASELINE_ROOT), per_benchmark={})
    paired = {}
    for variant in MODES:
        actual = q.base.read_json(root / variant / 'rollout_acceptance.json')
        for baseline in ('block3_mean', 'token_opd'):
            paired[f'{variant}_vs_{baseline}'] = original.n1.validate_paired_rollouts(
                q.base.read_json(BASELINE_ROOT / baseline / 'rollout_acceptance.json'), actual)
    for task in q.shared.TASK_COUNTS:
        values = {**baselines[task], **{name:r['per_task'][task] for name,r in results.items()}}
        report['per_benchmark'][task] = dict(models=values, differences_pp={
            f'{variant}_minus_{baseline}_step{step}': {
                metric:100*(values[f'{variant}_step{step}'][metric]-values[f'{baseline}_step{step}'][metric])
                for metric in ('avg_at_8', 'pass_at_8')}
            for variant in MODES for baseline in ('token_opd', 'block3_mean') for step in original.STEPS})
    q.jobs.write_json(root / 'baseline_paired_rollout_acceptance.json', dict(passed=True, comparisons=paired))
    q.jobs.write_json(root / 'paired_comparison.json', report)


def configured_runner():
    run = modules.private_module('_acp_component_runner', original.__file__)
    run.q = modules.private_module('_acp_component_instruct', original.q.__file__)
    run.q.base = modules.private_module('_acp_component_base', original.q.base.__file__)
    run.RUN_ROOT, run.CACHE = RUN_ROOT, Path('/dev/shm/a17c')
    run.ENTRYPOINT = Path(__file__).resolve()
    run.VARIANTS = run.q.VARIANTS = run.q.base.VARIANTS = tuple(MODES)
    run.REFERENCE = run.ROOT / 'deployments' / BASELINE_COMMIT
    run.ORDERING = 'adv3 probe/train/eval100,75,50,25 -> scale3 probe/train/eval100,75,50,25'
    source_manifest = run.ENTRYPOINT.parents[1] / 'configs/acp_component_source_hashes.json'
    run.LOSS_OVERRIDES = run.q.base.read_json(source_manifest)
    run.training_env = partial(training_env, run, run.training_env)
    run.expected_card = partial(expected_card, run.expected_card)
    run.preflight = partial(preflight, run, run.preflight)
    run.execute_ordered = execute_ordered
    run.write_comparison = write_comparison
    return run


if __name__ == '__main__':
    configured_runner().main()

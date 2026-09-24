#!/usr/bin/env python3
"""Prepare only: historical Token/Block3 pair, 32x1 responses, 100 updates."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from prepare_historical_single_rollout import load_recipe, reserve_root

METHODS = {'token_opd': (1, 'sum'), 'block3_mean': (3, 'mean')}
SAVE_STEPS = (25, 50, 75, 100)
PROJECT = 'opd_historical_pair_n1'
ALIGNMENT = 'Historical 1.7B Base / public4B GRPO Token-vs-legacy-Block3 repeat; 32x1, 100 steps, old prompts/seeds retained'


def training_env(run, runtime, commit, root, variant):
    if variant not in METHODS:
        raise ValueError('Only Token OPD and full legacy Block3 Mean are requested')
    env = run.base.launcher_env(runtime, commit, root, variant)
    env.update(PROJECT_NAME=PROJECT, EXP_NAME=f'historical17-{variant}-n1-step100',
        STUDENT_MODEL=str(run.STUDENT), MATH_TEACHER=str(run.TEACHER),
        STUDENT_MODEL_REVISION='', TEACHER_MODEL_REVISION='',
        OPD_PROMPT_PROTOCOL=run.PROTOCOL, OPD_REQUEST_SEED_RULE='legacy',
        TRAIN_BATCH_SIZE='32', ROLLOUT_GROUP_SIZE='1', TOTAL_TRAINING_STEPS='100',
        DIAGNOSTIC_SAVE_STEPS=','.join(map(str, SAVE_STEPS)), LOSSLESS_ROLLOUT_DIR=str(root/variant/'rollouts'),
        ROLLOUT_ATTEMPT_ID='formal', LOCAL_CACHE_ROOT=str(Path('/dev/shm/hp1')/variant/'train'),
        BASELINE_ALIGNMENT=ALIGNMENT)
    return env


def expected_card(source, root, variant):
    size, mode = METHODS[variant]
    return dict(source, variant=variant, opd_block_size=size, opd_block_advantage_mode=mode,
        opd_block_ablation='legacy', project_name=PROJECT,
        experiment_name=f'historical17-{variant}-n1-step100', train_batch_size=32,
        rollout_group_size=1, total_training_steps=100, diagnostic_save_steps=','.join(map(str, SAVE_STEPS)),
        diagnostic_output_dir=str(root/variant/'diagnostics'),
        lossless_rollout_dir=str(root/variant/'rollouts'), baseline_alignment=ALIGNMENT)


def validate_card(source, actual, root, variant):
    expected = expected_card(source, root, variant)
    differences = {k: [expected.get(k), actual.get(k)] for k in expected.keys() | actual.keys()
                   if expected.get(k) != actual.get(k)}
    if differences or expected.keys() != actual.keys():
        raise ValueError(f'Unapproved configuration drift: {differences}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    recipe, run, commit = load_recipe(args.runtime)
    import socket
    run.validate_host(socket.gethostname())
    source_folder = recipe.RUN_ROOT/'adv3'
    source = json.loads((source_folder/'run_card.json').read_text())
    run.validate_card(source, recipe.RUN_ROOT, 'adv3', commit)
    reserve_root(args.run_root)
    cards = {}
    for variant in METHODS:
        env = training_env(run, args.runtime, commit, args.run_root, variant)
        with (args.run_root/f'prepare_{variant}.log').open('xb') as log:
            subprocess.run(['bash', str(args.runtime/'scripts/launch_revisiting_block_opd_formal_train.sh')],
                cwd=args.runtime, env={**os.environ, **env}, stdout=log, stderr=subprocess.STDOUT, check=True)
        folder = args.run_root/variant
        cards[variant] = json.loads((folder/'run_card.json').read_text())
        validate_card(source, cards[variant], args.run_root, variant)
        for name in ('artifact_hashes.sha256', 'script_hashes.sha256'):
            if recipe.manifest_hashes(folder/name) != recipe.manifest_hashes(source_folder/name):
                raise ValueError(f'Input/runtime bytes changed: {name}')
        subprocess.run(['bash', '-n', str(folder/'command.sh')], check=True)
    allowed = {'variant', 'opd_block_size', 'opd_block_advantage_mode', 'experiment_name',
               'diagnostic_output_dir', 'lossless_rollout_dir'}
    left, right = cards.values()
    paired_diff = {k for k in left.keys() | right.keys() if left.get(k) != right.get(k)}
    if paired_diff - allowed:
        raise ValueError(f'Paired configuration mismatch: {paired_diff}')
    evaluation = dict(checkpoint_steps=list(reversed(SAVE_STEPS)), include_initial_student=True,
        tasks=run.shared.TASK_COUNTS, responses_per_problem=8, prompt_protocol=run.EVALUATION_PROTOCOL,
        grader_path=str(run.base.GRADER), grader_sha256=run.shared.grading.HISTORICAL_GRADER_SHA256,
        per_benchmark_metrics=['avg@8', 'pass@8'], retain_all_rollouts=True,
        initial_student_command=list(map(str, run.evaluation_command(
            args.runtime, run.STUDENT, args.run_root/'evaluations/initial_student'))))
    report = dict(prepared=True, training_started=False, launch_authorized=False,
        created_at=run.jobs.now(), source_commit=commit, preparer_sha256=run.assets.sha256(Path(__file__)),
        source_recipe_card=str(source_folder/'run_card.json'),
        source_recipe_card_sha256=run.assets.sha256(source_folder/'run_card.json'),
        source_recipe_note='Reuses non-loss settings only; B/C losses are NOT requested',
        initialization='Original student independently; never continue B Step50',
        total_steps=100, prompts_per_step=32, responses_per_prompt=1, ppo_batch_size=32,
        checkpoint_steps=list(SAVE_STEPS), cards=cards, evaluation=evaluation,
        paired_differences=sorted(paired_diff), full_legacy_block_loss=True,
        prompt_changed=False, per_request_seed_changed=False,
        files={f'{variant}/{name}': run.assets.sha256(args.run_root/variant/name)
               for variant in METHODS for name in ('run_card.json', 'command.sh',
                    'artifact_hashes.sha256', 'script_hashes.sha256')})
    run.jobs.write_json(args.run_root/'preparation.json', report)
    run.jobs.write_json(args.run_root/'preparation_acceptance.json', dict(passed=True,
        created_at=run.jobs.now(), prepared_sha256=run.assets.sha256(args.run_root/'preparation.json'),
        actual_cards_verified=True, paired_manifests_identical=True, command_syntax_verified=True,
        training_started=False, checkpoint_steps=list(SAVE_STEPS)))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

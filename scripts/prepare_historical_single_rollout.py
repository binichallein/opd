#!/usr/bin/env python3
"""Prepare, but never execute, historical B/C with 32 prompts and one response."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def single_rollout_env(old):
    required = dict(TRAIN_BATCH_SIZE='4', ROLLOUT_GROUP_SIZE='8',
                    TOTAL_TRAINING_STEPS='200', PPO_MINI_BATCH_SIZE='32',
                    OPD_REQUEST_SEED_RULE='legacy', RESUME_MODE='disable', PREPARE_ONLY='true')
    if any(old.get(k) != v for k, v in required.items()):
        raise ValueError('Unexpected historical recipe; refusing implicit configuration changes')
    return dict(old, TRAIN_BATCH_SIZE='32', ROLLOUT_GROUP_SIZE='1')


def validate_card_diff(old, new):
    diff = {k: {'old': old.get(k), 'new': new.get(k)} for k in old.keys() | new.keys()
            if old.get(k) != new.get(k)}
    allowed = {'train_batch_size', 'rollout_group_size', 'diagnostic_output_dir', 'lossless_rollout_dir'}
    if (set(diff) - allowed or new.get('train_batch_size') != 32
            or new.get('rollout_group_size') != 1
            or old.get('train_batch_size') != 4 or old.get('rollout_group_size') != 8):
        raise ValueError(f'Unapproved run-card change: {diff}')
    return diff


def reserve_root(root):
    root.mkdir(parents=True, exist_ok=False)


def load_recipe(runtime):
    sys.path[:0] = [str(runtime), str(runtime / 'scripts'), str(runtime / 'external/revisiting_opd')]
    import run_historical_component_ablation as recipe
    commit = (runtime / 'DEPLOYED_COMMIT').read_text().strip()
    if runtime != recipe.ROOT / 'deployments' / commit:
        raise ValueError('Expected the immutable ml2 training deployment')
    return recipe, recipe.configured_runner(), commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    recipe, run, commit = load_recipe(args.runtime)
    import socket
    run.validate_host(socket.gethostname())
    if args.source_root != recipe.RUN_ROOT:
        raise ValueError('This preparation must derive from the stopped historical B/C attempt')
    reserve_root(args.run_root)
    report = dict(prepared=False, training_started=False, launch_authorized=False,
                  source_root=str(args.source_root), runtime=str(args.runtime), source_commit=commit,
                  preparer_sha256=run.assets.sha256(Path(__file__)), variants={},
                  initialization='Original student independently for each arm, not Step50',
                  eval_changed=False, per_request_seed_changed=False)
    for variant in recipe.MODES:
        env = single_rollout_env(run.training_env(args.runtime, commit, args.run_root, variant))
        old = json.loads((args.source_root / variant / 'run_card.json').read_text())
        run.validate_card(old, args.source_root, variant, commit)
        with (args.run_root / f'prepare_{variant}.log').open('xb') as log:
            subprocess.run(['bash', str(args.runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh')],
                           cwd=args.runtime, env={**os.environ, **env}, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        folder = args.run_root / variant
        new = json.loads((folder / 'run_card.json').read_text())
        differences = validate_card_diff(old, new)
        for name in ('artifact_hashes.sha256', 'script_hashes.sha256'):
            if recipe.manifest_hashes(folder / name) != recipe.manifest_hashes(args.source_root / variant / name):
                raise ValueError(f'Changed input/runtime manifest: {name}')
        report['variants'][variant] = dict(differences=differences,
            files={name: run.assets.sha256(folder / name) for name in
                   ('run_card.json', 'command.sh', 'env.txt', 'artifact_hashes.sha256', 'script_hashes.sha256')})
    report.update(prepared=True, created_at=run.jobs.now(), total_steps=200,
                  prompts_per_step=32, responses_per_prompt=1, responses_per_step=32,
                  checkpoint_steps=[50, 100, 150, 200], fresh_prompt_occurrences=6400,
                  limitation='Prompt exposure increases versus old 4x8; no 8x speedup is guaranteed')
    run.jobs.write_json(args.run_root / 'preparation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

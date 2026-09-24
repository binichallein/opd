#!/usr/bin/env python3
"""One isolated old-recipe Step50->51 recovery audit, never a new n1 training run."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys

from prepare_historical_single_rollout import load_recipe, reserve_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe, run, commit = load_recipe(args.runtime)
    import socket
    run.validate_host(socket.gethostname())
    reserve_root(args.output)
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    source = recipe.RUN_ROOT / 'adv3'
    checkpoint = source / 'checkpoints/global_step_50'
    state = args.output / 'audit_state.json'
    cache = Path('/dev/shm/h50')
    env = dict(os.environ, PATH=str(run.base.PYTHON.parent) + ':' + os.environ.get('PATH', ''),
               PYTHONPATH=f'{args.runtime}:{args.runtime}/external/revisiting_opd',
               CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', ENGINE='vllm', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               EVAL_GRADE_UTILS_PATH=str(run.base.GRADER))
    if env.get('RAY_TMPDIR'):
        raise ValueError('Unexpected inherited Ray directory')
    for key, name in dict(TMPDIR='tmp', VLLM_CACHE_ROOT='vllm', TRITON_CACHE_DIR='triton',
                          TORCHINDUCTOR_CACHE_DIR='inductor', CUDA_CACHE_PATH='cuda',
                          OUTLINES_CACHE_DIR='outlines').items():
        (cache / name).mkdir(parents=True, exist_ok=True)
        env[key] = str(cache / name)
    run.jobs.wait_for_idle()
    protected = {str(p): run.assets.sha256(p) for p in checkpoint.rglob('*') if p.is_file()}
    if len(protected) < 13:
        raise ValueError('Incomplete original checkpoint')
    run.jobs.write_json(args.output / 'source_checkpoint_hashes.json', protected)
    training = run.training_env(args.runtime, commit, args.output, 'adv3')
    training.update(RESUME_MODE='resume_path', RESUME_FROM_PATH=str(checkpoint),
                    TOTAL_TRAINING_STEPS='200', STOP_AFTER_STEP='51', DIAGNOSTIC_SAVE_STEPS='51',
                    ROLLOUT_ATTEMPT_ID='resume_step50_verification', LOCAL_CACHE_ROOT=str(cache / 'train'))
    job = args.output / 'adv3'
    try:
        run.jobs.run_job(['bash', args.runtime / 'scripts/launch_revisiting_block_opd_formal_train.sh'],
                         args.output / 'prepare', args.runtime, state, {**env, **training})
        card = json.loads((job / 'run_card.json').read_text())
        assert card['train_batch_size'] == 4 and card['rollout_group_size'] == 8
        assert card['total_training_steps'] == 200 and card['stop_after_step'] == 51
        run.jobs.run_job(['bash', job / 'command.sh'], job, args.runtime, state, env, gpu=True)
        import torch
        import run_qwen8_teacher_pair as optimizer_audit
        torch.set_num_threads(2)
        resumed = job / 'checkpoints/global_step_51'
        data = torch.load(resumed / 'data.pt', map_location='cpu', weights_only=False)
        for rank in range(4):
            extra = torch.load(resumed / f'actor/extra_state_world_size_4_rank_{rank}.pt',
                               map_location='cpu', weights_only=False)
            run.base.validate_resume_state(extra, data, 51)
            name = f'actor/model_world_size_4_rank_{rank}.pt'
            if run.assets.sha256(resumed / name) == protected[str(checkpoint / name)]:
                raise ValueError(f'No model update for rank {rank}')
        optimizer_audit.verify_optimizer_state(job, (51,))
        rows = run.read_archive(job / 'rollouts/resume_step50_verification/step_000051/raw.jsonl.gz')
        old = run.read_archive(source / 'rollouts/formal/step_000051/raw.jsonl.gz')
        if len(rows) != 32 or len(old) != 32:
            raise ValueError('Expected 32 resumed trajectories')
        for row, reference in zip(rows, old):
            recipe.validate_legacy_rollout(row, 51)
            if row['source_extra_info'] != reference['source_extra_info']:
                raise ValueError('Dataloader did not resume at the expected next batch')
        run.jobs.wait_for_idle()
        run.jobs.write_json(state, dict(passed=True, status='complete', source_step=50,
            resumed_step=51, completed_at=run.jobs.now(), total_scheduler_horizon=200,
            full_state_restored=True, original_next_batch_matches=True, raw_rollouts=32,
            new_n1_training_started=False, verifier_sha256=run.assets.sha256(Path(__file__))))
    except BaseException as exc:
        pid = job / 'job.pid'
        if pid.exists():
            run.cleanup_failed_group(int(pid.read_text()))
        run.jobs.write_json(state, dict(passed=False, status='failed', error=repr(exc),
                                       updated_at=run.jobs.now()))
        raise
    finally:
        run.shared.verify_hashes(protected)
        run.jobs.write_json(args.output / 'source_preservation.json',
                            dict(passed=True, files=len(protected), checked_at=run.jobs.now()))


if __name__ == '__main__':
    main()

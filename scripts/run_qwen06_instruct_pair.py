#!/usr/bin/env python3
"""ml2 Qwen3-4B -> 0.6B Instruct, reusing the accepted 8B -> 1.7B recipe."""

import argparse
from functools import partial
import importlib.util
import os
from pathlib import Path
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import prepare_qwen06_screen_assets as assets
import qualify_qwen06_teachers as screen
import run_qwen17_instruct_pair as previous

ROOT = previous.ROOT
RUN_ROOT = ROOT / 'runs/20260923v1_qwen06_instruct_blockfirst_seed21_ml2'
CACHE = Path('/dev/shm/q06i')
RAY_CACHES = {'block3_mean': CACHE/'rb', 'token_opd': CACHE/'rt'}
QUALIFICATION = ROOT / 'runs/20260923v1_qwen06_teacher_screen_ml2/qualification'
CAPABILITY_SHA = 'e19bad52b2fa48c9b074c0fdc661689287b5d35fdc49ecd9e967a6acdd62988c'
TRAIN_REFERENCE = ROOT / 'deployments/be736b5fac4f26ff59f4e2c21abafc456c2503f6'
RECIPE_FILES = ('opd_ext/math_protocol.py', 'opd_ext/request_seeds.py',
                'scripts/launch_revisiting_block_opd_formal_train.sh',
                'scripts/eval_qwen3_math_vllm.py')


def validate_screen(summary, digest, preparation):
    result = summary.get('reports', {}).get('i4_i06', {})
    if (digest != CAPABILITY_SHA or summary.get('cell_errors') != {}
            or summary.get('protocol') != screen.protocol()
            or summary.get('training_started') is not False
            or preparation.get('protocol') != screen.protocol()
            or preparation.get('selection_sha256') != screen.prior.SOURCE_SELECTION_SHA
            or result.get('status') != 'passed' or result.get('passed') is not True
            or result.get('failures') != [] or result.get('teacher') != 'i4'
            or result.get('student') != 'i06' or not result.get('checks')
            or not all(result['checks'].values())):
        raise ValueError('The pinned successful i4_i06 screening gate is required')
    specs = assets.specifications()
    for key in ('i06', 'i4'):
        model = preparation.get('models', {}).get(key, {})
        expected = dict(repo=specs[key]['repo'], revision=specs[key]['revision'],
                        path=str(ROOT/'models'/specs[key]['repo'].split('/')[1]))
        if any(model.get(k) != value for k, value in expected.items()):
            raise ValueError(f'Screening model identity changed: {key}')


def validate_smoke(health):
    if health.get('num_rollouts') != 4 or health.get('generated_think_tags') != 0:
        raise ValueError('Native GPU non-thinking check is not accepted')


def preflight(run, runtime):
    if os.environ.get('RAY_ADDRESS') or os.environ.get('RAY_TMPDIR'):
        raise ValueError('Unexpected inherited Ray connection or temporary directory')
    old = screen.old
    summary = old.read_sealed(QUALIFICATION/'pair_summary.json')
    preparation = old.read_sealed(QUALIFICATION/'prepare_manifest.json')
    validate_screen(summary, assets.sha256(QUALIFICATION/'pair_summary.json'), preparation)
    if (summary['prepare_sha256'] != assets.sha256(QUALIFICATION/'prepare_manifest.json')
            or preparation['selected_sha256'] != assets.sha256(QUALIFICATION/'selected.json')):
        raise ValueError('Screening preparation/question evidence changed')
    protected = run.common_preflight(runtime)
    for key in ('i06', 'i4'):
        model = assets.ensure_asset(key)
        if model != preparation['models'][key]:
            raise ValueError('Training assets differ from the qualified original models')
        protected.update(model['hashes'])
    for name in ('pair_summary.json', 'prepare_manifest.json', 'selected.json'):
        protected[str(QUALIFICATION/name)] = assets.sha256(QUALIFICATION/name)
    if set(summary['evidence']) != set(preparation['cells']):
        raise ValueError('Incomplete screening evidence inventory')
    for key, digest in summary['evidence'].items():
        folder = QUALIFICATION/'cells'/key
        completion = old.read_sealed(folder/'completion.json')
        if (assets.sha256(folder/'completion.json') != digest or completion['complete'] is not True
                or completion['cell'] != key):
            raise ValueError('Changed screening completion evidence')
        protected[str(folder/'completion.json')] = digest
        protected.update({str(folder/name): h for name, h in completion['files'].items()})
    for key in ('i06', 'i4'):
        path = QUALIFICATION/'cells'/f'{key}_smoke'/'smoke_health.json'
        health = old.read_sealed(path)
        validate_smoke(health)
        protected[str(path)] = assets.sha256(path)
    # New model/queue identity only; preserve the accepted training and evaluation implementation.
    for name in RECIPE_FILES:
        if assets.sha256(runtime/name) != assets.sha256(TRAIN_REFERENCE/name):
            raise ValueError(f'8B -> 1.7B recipe source changed: {name}')
    old.verify_hashes(protected)
    return protected


def train_with_ray_gate(run, original, root, variant, runtime, commit, runner, protected):
    output = root/f'{variant}_ray_gate.json'
    runner([run.base.PYTHON, runtime/'scripts/run_qwen06_instruct_pair.py',
            '--ray-gate-output', output, '--ray-gate-variant', variant],
           root/f'queue_jobs/{variant}_ray_warmup',
           job_env={'CUDA_VISIBLE_DEVICES': ''}, deadline_epoch=time.time()+360)
    if run.base.read_json(output).get('passed') is not True:
        raise ValueError('CPU Ray startup/worker gate failed')
    return original(root, variant, runtime, commit, runner, protected)


def configured_runner():
    # A private module namespace reuses the queue without mutating the historical pair's globals.
    spec = importlib.util.spec_from_file_location('_qwen06_instruct_queue', previous.__file__)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    models = assets.specifications()
    run.RUN_ROOT, run.CACHE = RUN_ROOT, CACHE
    run.STUDENT = ROOT/'models'/models['i06']['repo'].split('/')[1]
    run.TEACHER = ROOT/'models'/models['i4']['repo'].split('/')[1]
    run.STUDENT_REVISION = models['i06']['revision']
    run.TEACHER_REVISION = models['i4']['revision']
    run.QUALIFICATION, run.CAPABILITY_SHA = QUALIFICATION, CAPABILITY_SHA
    run.PROJECT_NAME = 'opd_qwen06_instruct'
    run.EXPERIMENT_PREFIX = 'qwen06-instruct'
    run.DISPLAY_LABEL = 'Qwen0.6 Instruct'
    run.preflight = partial(preflight, run)
    run.train_variant = partial(train_with_ray_gate, run, run.train_variant)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ray-gate-output', type=Path)
    parser.add_argument('--ray-gate-variant', choices=previous.VARIANTS)
    args = parser.parse_args()
    if args.ray_gate_output:
        import socket
        import recover_qwen17_instruct_token as recovery
        previous.validate_host(socket.gethostname())
        if args.ray_gate_variant is None or args.ray_gate_output != RUN_ROOT/f'{args.ray_gate_variant}_ray_gate.json':
            raise ValueError('Only this attempt may receive Ray gate artifacts')
        recovery.CACHE = RAY_CACHES[args.ray_gate_variant]
        recovery.ray_gate(args.ray_gate_output)
    else:
        configured_runner().main()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Bounded ml2 update/resume acceptance. Never launches formal training or eval."""

import argparse
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import run_qwen4_pair as old
from opd_ext.math_protocol import QWEN_COMPLETION_PROTOCOL as PROTOCOL, completion_input_ids
from opd_ext.request_seeds import SEED_RULE, request_identities, request_seed

assets, base, jobs = old.assets, old.base, old.jobs
ROOT = assets.ROOT
RUN_ROOT = ROOT/'runs/20260920v1_qwen4_completion_gate_seed21_ml2'
CACHE = Path('/limx_embap/tos/q4/c1')
VARIANTS = ('block3_mean', 'token_opd')


def training_env(runtime, commit, root, variant, probe_step):
    if probe_step not in (1, 2):
        raise ValueError('This controller permits two-step probes only')
    env = old.training_env(runtime, commit, root, variant, probe_step)
    env.update(PROJECT_NAME='opd_qwen4_completion_gate', EXP_NAME=f'qwen4-completion-{variant}-probe',
               OPD_PROMPT_PROTOCOL=PROTOCOL, OPD_REQUEST_SEED_RULE=SEED_RULE,
               LOCAL_CACHE_ROOT=str(CACHE/'train'),
               BASELINE_ALIGNMENT='Bounded completion and independent request seed acceptance; original historical loss/mask')
    return env


def read_archive(path):
    digest = path.with_suffix(path.suffix+'.sha256').read_text().split()[0]
    if assets.sha256(path) != digest:
        raise ValueError('Archive hash mismatch')
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def validate_seed_settings(row, identity, step):
    expected = {'temperature': 1., 'top_p': .9, 'top_k': -1, 'max_tokens': 16384,
                'n': 1, 'ignore_eos': False, 'stop_token_ids': [], 'seed': request_seed(identity, turn=0)}
    if (row['protocol'] != PROTOCOL or row['enable_thinking'] is not False or row['step'] != step
            or row['mask_policy'] != 'historical_eos_mask' or row['eos_token_id'] != 151643
            or row['request_identity'] != identity or row['request_turn'] != 0
            or any(row['sampling'].get(k) != v for k, v in expected.items())):
        raise ValueError('Completion seed/protocol/sampling mismatch')


def audit_rollouts(run, steps):
    import torch
    from transformers import AutoTokenizer
    from verl.utils.torch_functional import get_response_mask
    from diagnose_token_truncation import analyze_tokens
    tokenizer = AutoTokenizer.from_pretrained(assets.STUDENT, local_files_only=True)
    evidence = []
    for step in steps:
        paths = list((run/'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))
        if len(paths) != 1:
            raise ValueError('Incomplete or duplicate step archives')
        rows = read_archive(paths[0])
        old_rows = read_archive(old.RUN_ROOT/f'block3_mean/rollouts/formal/step_{step:06d}/raw.jsonl.gz')
        if len(rows) != 32 or len(old_rows) != 32 or len({r['traj_uid'] for r in rows}) != 32:
            raise ValueError('Incomplete trajectory coverage')
        sources = [old_rows[i]['source_extra_info'] for i in range(0, 32, 8)]
        identities = request_identities(sources, group_size=8, global_seed=21, step=step)
        for row, original, identity in zip(rows, old_rows, identities):
            if row['source_extra_info'] != original['source_extra_info']:
                raise ValueError('Data order differs from historical seed21 schedule')
            expected_ids = completion_input_ids(tokenizer, row['source_extra_info']['question'])
            if row['prompt_token_ids'] != expected_ids:
                raise ValueError('Actual training prompt differs from evaluation input IDs')
            validate_seed_settings(row, identity, step)
            ids, count = row['training_response_token_ids'], row['response_length']
            mask = get_response_mask(torch.tensor([ids]), 151643, dtype=torch.long)[0].tolist()
            if (len(ids) != 16384 or row['response_tensor_width'] != 16384
                    or count != len(row['response_token_ids']) or not 0 <= count <= 16384
                    or row['padding_length'] != 16384-count or ids[:count] != row['response_token_ids']
                    or mask != row['training_response_mask']
                    or len(row['training_rollout_log_probs']) != 16384
                    or not all(math.isfinite(p) for p in row['training_rollout_log_probs'])
                    or row['finish_reason'] not in ('length', 'stop')
                    or (row['finish_reason'] == 'length' and count != 16384)):
                raise ValueError('Actual training mask/logprob/termination mismatch')
        unique = [len({tuple(r['response_token_ids']) for r in rows[i:i+8]}) for i in range(0, 32, 8)]
        stats = [analyze_tokens(r['response_token_ids'], 16384) for r in rows]
        evidence.append({'step': step, 'n': len(rows), 'sha256': assets.sha256(paths[0]),
                         'independent_seeds': len({r['sampling']['seed'] for r in rows}),
                         'unique_responses_per_question': unique,
                         'length_stops': sum(r['finish_reason'] == 'length' for r in rows),
                         'periodic_tails': sum(s['tail_period'] is not None for s in stats),
                         'format_errors': sum('\\boxed' not in r['response_text'] for r in rows),
                         'think_tags': sum(r['generated_think_tags'] for r in rows),
                         'historical_data_order_matched': True, 'eval_prompt_ids_matched': True})
    return evidence


def input_contract():
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.utils.tokenizer import hf_tokenizer
    from agent_system.environments.env_manager import MathEnvironmentManager
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    tokenizer = hf_tokenizer(str(assets.STUDENT), local_files_only=True)
    config = OmegaConf.create({'data': {'opd_prompt_protocol': PROTOCOL,
        'apply_chat_template_kwargs': {'enable_thinking': False}, 'max_prompt_length': 2048,
        'truncation': 'middle', 'return_raw_chat': True}})
    manager = MathEnvironmentManager.__new__(MathEnvironmentManager)
    manager.config = config
    collector = TrajectoryCollector(config, tokenizer)
    checked = {}
    for task in jobs.TASKS:
        rows = [json.loads(line) for line in (base.DATA/'eval_jsonl'/f'{task}.jsonl').read_text().splitlines()]
        manager.tasks = [r['problem'] for r in rows]
        observations = manager.build_text_obs(manager.tasks)
        for question, observation in zip(manager.tasks, observations):
            raw = np.empty(1, dtype=object)
            raw[0] = [{'role': 'user', 'content': question}]
            gen = DataProto.from_single_dict({'input_ids': torch.zeros((1, 1), dtype=torch.long),
                'raw_prompt': raw, 'data_source': np.array(['dapo-math-17k'], dtype=object)})
            processed = collector.preprocess_single_sample(0, gen, {'text': [observation]})
            ids = completion_input_ids(tokenizer, question)
            if (processed['raw_prompt_ids'] != ids
                    or processed['input_ids'][processed['attention_mask'].bool()].tolist() != ids):
                raise ValueError('Actual collector/eval tokenization mismatch')
        checked[task] = len(rows)
    return {'passed': True, 'actual_collector_exercised': True, 'eval_questions_checked': checked,
            'protocol': PROTOCOL, 'chat_template_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-rollouts', type=Path)
    args = parser.parse_args()
    if args.audit_rollouts:
        print(json.dumps(audit_rollouts(args.audit_rollouts, (1, 2))), flush=True)
        return
    runtime = Path(__file__).resolve().parents[1]
    commit = (runtime/'DEPLOYED_COMMIT').read_text().strip()
    if runtime != ROOT/'deployments'/commit:
        raise ValueError('Immutable ml2 deployment required')
    from recover_historical17_llama import check_socket_budget
    check_socket_budget(CACHE/'train/tmp')
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    RUN_ROOT.mkdir(exist_ok=True)
    state = RUN_ROOT/'queue_state.json'
    with (RUN_ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (RUN_ROOT/'queue_manifest.json').exists():
            raise FileExistsError('No automatic retry or overwrite of an existing gate')
        jobs.write_json(RUN_ROOT/'queue_manifest.json', {'source_commit': commit, 'created_at': jobs.now(),
            'student': str(assets.STUDENT), 'teacher': str(assets.TEACHER), 'variants': VARIANTS,
            'steps': [1, 2], 'scope': 'engineering update/resume gate, not formal method validation',
            'prompt_protocol': PROTOCOL, 'request_seed_rule': SEED_RULE,
            'formal_training_autostart': False, 'full_eval_autostart': False, 'retain_all_checkpoints': True})
        (RUN_ROOT/'queue.pid').write_text(str(os.getpid())+'\n')
        env = dict(os.environ, PATH=f'{base.VENV}/bin:'+os.environ.get('PATH', ''),
                   PYTHONPATH=f'{runtime}:{runtime}/external/revisiting_opd', CUDA_VISIBLE_DEVICES='0,1,2,3',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                   RAY_DEDUP_LOGS='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', ENGINE='vllm',
                   VLLM_WORKER_MULTIPROC_METHOD='spawn', EVAL_GRADE_UTILS_PATH=str(base.GRADER))
        if env.get('RAY_TMPDIR'):
            raise ValueError('Unexpected RAY_TMPDIR override')
        for key, suffix in {'TMPDIR': 'tmp', 'VLLM_CACHE_ROOT': 'vllm', 'TORCHINDUCTOR_CACHE_DIR': 'inductor',
                            'TRITON_CACHE_DIR': 'triton', 'CUDA_CACHE_PATH': 'cuda', 'OUTLINES_CACHE_DIR': 'outlines'}.items():
            (CACHE/suffix).mkdir(parents=True, exist_ok=True)
            env[key] = str(CACHE/suffix)
        def runner(command, job, job_env=None, **kwargs):
            jobs.run_job(command, job, runtime, state, {**env, **(job_env or {})}, **kwargs)
        try:
            jobs.wait_for_idle()
            runner(['sha256sum', '-c', '.expected.sha256'], RUN_ROOT/'queue_jobs/runtime_hashes')
            protected = assets.verify_assets()
            if assets.sha256(base.DATA/'train.parquet') != jobs.TRAIN_SHA:
                raise ValueError('Changed training data')
            protected[str(base.DATA/'train.parquet')] = jobs.TRAIN_SHA
            jobs.write_json(RUN_ROOT/'protected_inputs.json', protected)
            if shutil.disk_usage(ROOT).free < 1_000_000_000_000:
                raise ValueError('Insufficient space to retain all full probe states')
            jobs.write_json(RUN_ROOT/'input_contract.json', input_contract())
            cards = {}
            for variant in VARIANTS:
                runner(['bash', runtime/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                       RUN_ROOT/f'queue_jobs/prepare_{variant}_1',
                       job_env=training_env(runtime, commit, RUN_ROOT, variant, 1))
                cards[variant] = base.read_json(RUN_ROOT/variant/'run_card.json')
            allowed = {'variant', 'opd_block_size', 'opd_block_advantage_mode', 'experiment_name',
                       'diagnostic_output_dir', 'lossless_rollout_dir'}
            left, right = (cards[v] for v in VARIANTS)
            if any(left.get(k) != right.get(k) for k in (set(left)|set(right))-allowed):
                raise ValueError('Uncontrolled paired run-card difference')
            jobs.write_json(RUN_ROOT/'initial_paired_cards.json', cards)
            reports = {}
            for variant in VARIANTS:
                run = RUN_ROOT/variant
                for step in (1, 2):
                    if step == 2:
                        runner(['bash', runtime/'scripts/launch_revisiting_block_opd_formal_train.sh'],
                               RUN_ROOT/f'queue_jobs/prepare_{variant}_2',
                               job_env=training_env(runtime, commit, RUN_ROOT, variant, 2))
                    job = RUN_ROOT/f'queue_jobs/{variant}_probe{step}'
                    runner(['bash', run/'command.sh'], job, gpu=True)
                    shutil.copyfile(job/'logs/job.log', run/'logs/nohup.log')
                    runner(old.audit_command(runtime, run, commit, probe_step=step),
                           RUN_ROOT/f'queue_jobs/{variant}_audit{step}')
                    jobs.write_json(run/f'rollout_acceptance_step{step}.json',
                                    {'passed': True, 'steps': audit_rollouts(run, range(1, step+1))})
                base.audit_resume(run)
                reports[variant] = audit_rollouts(run, (1, 2))
                runner([base.PLOT_PYTHON, runtime/'scripts/analyze_single_opd_diagnostics.py',
                        '--run-dir', run, '--output-dir', run/'figures', '--label', f'Qwen4 {variant} completion probe'],
                       RUN_ROOT/f'queue_jobs/{variant}_figures')
            for step in (1, 2):
                arms = [read_archive(next((RUN_ROOT/v/'rollouts').glob(f'*/step_{step:06d}/raw.jsonl.gz'))) for v in VARIANTS]
                fields = ('prompt_token_ids', 'source_extra_info', 'request_identity', 'sampling')
                if any(any(a[k] != b[k] for k in fields) for a, b in zip(*arms)):
                    raise ValueError('Actual paired input/seed schedule mismatch')
            old.shared.verify_hashes(protected)
            jobs.wait_for_idle()
            jobs.write_json(RUN_ROOT/'gate_acceptance.json', {'passed': True, 'arms': reports,
                'formal_training_started': False, 'limitation': 'Two updates do not establish long-run quality or gains'})
            jobs.write_json(state, {'status': 'complete', 'updated_at': jobs.now(), 'formal_training_started': False})
        except BaseException as exc:
            previous = base.read_json(state) if state.exists() else {}
            jobs.write_json(state, {**previous, 'status': 'failed', 'error': str(exc), 'updated_at': jobs.now()})
            raise


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Read-only ml2 result backup, hash validation, and small Git-ready exports."""

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
SOURCE = ROOT/'runs/20260921v1_qwen17_instruct_blockfirst_seed21_ml2'
RECOVERY = ROOT/'runs/20260922v1_qwen17_instruct_token_recovery_seed21_ml2'
DATA = ROOT/'data/math_opd_dapo17k_hf_full_eval4'
TASKS = {'math500':500,'aime24':30,'aime25':30,'amc23':83}
EXPECTED = {'student_base'} | {f'{v}_step{s}' for v in ('block3_mean','token_opd') for s in (50,100,150,200)}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def pending_models(acceptance):
    return sorted(EXPECTED-set(acceptance['models']))


def validate_acceptance(acceptance):
    models = acceptance.get('models',{})
    if (acceptance.get('passed') is not True or not models or not set(models)<=EXPECTED
            or acceptance.get('complete') != (set(models)==EXPECTED)):
        raise ValueError('Require accepted results with an honest complete/pending status')
    for result in models.values():
        if result.get('passed') is not True or set(result.get('per_task',{})) != set(TASKS):
            raise ValueError('Incomplete or failed model evaluation')
        for task,n in TASKS.items():
            metric = result['per_task'][task]
            if metric.get('num_examples')!=n or metric.get('num_rollouts')!=8*n:
                raise ValueError('Incomplete benchmark sample count')
            for key in ('avg_at_8','pass_at_8'):
                if not isinstance(metric.get(key),(int,float)) or not 0<=metric[key]<=1:
                    raise ValueError('Invalid score')
        archive = result.get('rollout_archive',{})
        if (archive.get('passed') is not True or archive.get('num_rollouts')!=5144
                or archive.get('num_archive_files')!=32):
            raise ValueError('Missing full evaluation rollout archive')


def local_path(snapshot, remote):
    remote = Path(remote)
    if '..' in remote.parts:
        raise ValueError('Path traversal is not allowed')
    for root in (SOURCE,RECOVERY):
        if remote.is_relative_to(root/'evaluations'):
            return snapshot/remote.relative_to(root)
    if remote.is_relative_to(DATA/'eval_jsonl'):
        return snapshot/'eval_jsonl'/remote.relative_to(DATA/'eval_jsonl')
    raise ValueError(f'Unexpected evaluation artifact path: {remote}')


def verify_file(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return {'sha256':actual,'bytes':path.stat().st_size}


def result_rows(acceptance):
    return [{'model':name,'benchmark':task,**result['per_task'][task]}
            for name,result in sorted(acceptance['models'].items()) for task in TASKS]


def fetch(snapshot, acceptance):
    if (snapshot/'backup_acceptance.json').exists():
        raise FileExistsError('Choose a new snapshot; verified backups are immutable')

    def copy(source, destination, *, training=False):
        destination.mkdir(parents=True,exist_ok=True)
        cmd = ['rsync','-az','--partial']
        if training:
            cmd += ['--exclude=checkpoints/']
        print(f'Copy {source} -> {destination}',flush=True)
        subprocess.run([*cmd,f'ml2:{source}',str(destination)+'/'],check=True)
    for arm,root in (('block3_mean',SOURCE),('token_opd',RECOVERY)):
        copy(root/arm,snapshot,training=True)
    for name in sorted(acceptance['models']):
        root = RECOVERY if name.startswith('token_opd_') else SOURCE
        copy(root/'evaluations'/name,snapshot/'evaluations')
    copy(DATA/'eval_jsonl',snapshot)
    metadata = ['queue_manifest.json','paired_rollout_acceptance.json','recovery_preflight.json',
                'ray_gate.json','startup_acceptance.json','recovery_command_comparison.json']
    if acceptance['complete']:
        metadata += ['queue_state.json','paired_comparison.json']
    for filename in metadata:
        copy(RECOVERY/filename,snapshot/'queue_metadata')


def verify_snapshot(snapshot, acceptance):
    verified = {}
    for name,result in acceptance['models'].items():
        if read_json(snapshot/'evaluations'/name/'acceptance.json') != result:
            raise ValueError('Per-model acceptance differs from frozen root snapshot')
        hashes = result['sha256']
        for remote,digest in result['rollout_archive']['sha256'].items():
            if hashes.get(remote)!=digest:
                raise ValueError('Archive checksum differs from root acceptance')
        for remote,digest in hashes.items():
            path = local_path(snapshot,remote)
            verified[str(path.relative_to(snapshot))] = verify_file(path,digest)
    training = {}
    for arm in ('block3_mean','token_opd'):
        folder = snapshot/arm
        audit = read_json(folder/'rollout_acceptance.json')
        if (audit.get('passed') is not True or audit.get('total_rollouts')!=6400
                or [r['step'] for r in audit['steps']]!=list(range(1,201))):
            raise ValueError('Incomplete training rollout audit')
        for item in audit['steps']:
            paths = list((folder/'rollouts').glob(f'*/step_{item["step"]:06d}/raw.jsonl.gz'))
            if len(paths)!=1 or item['count']!=32:
                raise ValueError('Missing/duplicate training archive')
            verified[str(paths[0].relative_to(snapshot))] = verify_file(paths[0],item['sha256'])
        rows = [json.loads(line) for line in (folder/'diagnostics/scalars.jsonl').read_text().splitlines()]
        if ([r['step'] for r in rows]!=list(range(1,201))
                or not all(math.isfinite(v) for row in rows for v in row.values() if type(v) in (int,float))):
            raise ValueError('Incomplete/nonfinite training diagnostic snapshot')
        if len(list((folder/'diagnostics').glob('step_*.npz')))!=200:
            raise ValueError('Missing position diagnostic arrays')
        card = read_json(folder/'run_card.json')
        training[arm] = {'run_card':card,'rollout_count':6400,'diagnostic_steps':200,
            'length_stop_rollouts':sum(r['length_stops'] for r in audit['steps']),
            'generated_think_tags':sum(r['generated_think_tags'] for r in audit['steps']),
            'checkpoints_remote_only':[50,100,150,200]}
    # Checksums also protect metadata, figures and diagnostic arrays on subsequent local reads.
    for path in sorted(snapshot.rglob('*')):
        if path.is_file() and path.name!='backup_acceptance.json':
            rel = str(path.relative_to(snapshot))
            if rel not in verified:
                verified[rel] = {'sha256':sha256(path),'bytes':path.stat().st_size}
    return {'passed':True,'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_evaluation_complete':acceptance['complete'],'accepted_models':sorted(acceptance['models']),
            'pending_models':pending_models(acceptance),'eval_rollouts':5144*len(acceptance['models']),
            'training':training,'files':verified,'exclusions':['model/checkpoint weights','training parquet']}


def export(snapshot, output, acceptance, verification):
    output.mkdir(parents=True,exist_ok=False)
    rows = result_rows(acceptance)
    summary = {'snapshot':snapshot.name,'local_backup':str(snapshot.resolve()),
        'verified_at':verification['verified_at'],'complete':acceptance['complete'],
        'pending_models':pending_models(acceptance),'score_unit':'fraction; multiply by 100 for percent',
        'training':verification['training'],'per_model':{name:result['per_task'] for name,result in acceptance['models'].items()},
        'source_acceptance_sha256':sha256(snapshot/'evaluation_acceptance.json'),
        'limitations':['One training seed (21); checkpoint results are not independent replications.',
                      'Block3 uses shared advantage, joint PPO ratio and changed reduction.',
                      'Format error follows the archived historical metric; not mathematical incorrectness.']}
    write_json(output/'results.json',summary)
    write_json(output/'backup_acceptance.json',verification)
    shutil.copy2(snapshot/'evaluation_acceptance.json',output/'source_evaluation_acceptance.json')
    with (output/'metrics.csv').open('w',newline='') as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for arm in ('block3_mean','token_opd'):
        shutil.copytree(snapshot/arm/'figures',output/'figures'/arm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Choose a new export directory; accepted snapshots are immutable')
    acceptance = read_json(args.snapshot_dir/'evaluation_acceptance.json')
    validate_acceptance(acceptance)
    if args.fetch:
        fetch(args.snapshot_dir,acceptance)
    verified = verify_snapshot(args.snapshot_dir,acceptance)
    write_json(args.snapshot_dir/'backup_acceptance.json',verified)
    export(args.snapshot_dir,args.output_dir,acceptance,verified)
    print(json.dumps({'passed':True,'models':verified['accepted_models'],'pending':verified['pending_models'],
        'local_files':len(verified['files']),'local_bytes':sum(x['bytes'] for x in verified['files'].values()),
        'output':str(args.output_dir)}),flush=True)


if __name__ == '__main__':
    main()

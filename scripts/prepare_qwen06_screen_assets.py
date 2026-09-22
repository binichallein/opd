#!/usr/bin/env python3
"""Pinned ModelScope assets for inference-only eight-pair teacher screening."""

import argparse
import fcntl
import json
from pathlib import Path
import urllib.parse
import urllib.request

import prepare_qwen17_pair_assets as previous

ROOT = previous.ROOT
CONFIG = Path(__file__).resolve().parents[1] / 'configs/experiments/qwen06_teacher_screen_assets.json'
sha256 = previous.sha256
OWNER = '.qwen06_screen_owner.json'


def specifications():
    return json.loads(CONFIG.read_text())['models']


def verify_files(directory, files):
    directory = Path(directory)
    hashes = {}
    for name, record in files.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe file name')
        path = directory / name
        size = previous.legacy._regular(path)
        if 'Size' in record and size != record['Size']:
            raise ValueError(f'Wrong size: {path}')
        digest = sha256(path)
        if digest != record['Sha256']:
            raise ValueError(f'Wrong SHA256: {path}')
        hashes[str(path)] = digest
    return hashes


def require_owned_download(path, spec):
    marker = Path(path) / OWNER
    identity = dict(owner='qwen06_screen_assets_v1', repo=spec['repo'],
                    revision=spec['revision'], files=spec['files'])
    if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != identity:
        raise ValueError('Existing directory lacks matching download ownership')
    return identity


def ensure_asset(key, download=False):
    spec = specifications()[key]
    path = ROOT / 'models' / spec['repo'].split('/')[1]
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink asset path is not allowed')
    if download and spec['download_allowed']:
        if not path.exists():
            path.mkdir()
            with (path / OWNER).open('x') as stream:
                json.dump(dict(owner='qwen06_screen_assets_v1', repo=spec['repo'],
                               revision=spec['revision'], files=spec['files']), stream, indent=2)
        require_owned_download(path, spec)
        lockpath = path / '.download.lock'
        if lockpath.is_symlink():
            raise ValueError('Symlink download lock')
        with lockpath.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name, record in spec['files'].items():
                previous._download_file(dict(spec, path=path), name, record)
    hashes = verify_files(path, spec['files'])
    weights = {name for name in spec['files'] if name.endswith('.safetensors')}
    if (path / 'model.safetensors.index.json').exists():
        previous.legacy.verify_index(path, weights)
    elif weights != {'model.safetensors'}:
        raise ValueError('Unexpected weight layout')
    config = json.loads((path / 'config.json').read_text())
    if config.get('architectures') != ['Qwen3ForCausalLM']:
        raise ValueError('Unexpected architecture')
    return dict(key=key, path=str(path), repo=spec['repo'], revision=spec['revision'],
                source=spec['source'], hashes=hashes, config=config)


def freeze_config():
    """Generate metadata from structured official API responses, never model bytes."""
    pins = {'b06': ('Qwen3-0.6B-Base', 'master'),
            'i06': ('Qwen3-0.6B', '09b42cad3d112e832108974449ccb5e8e0f5b5d1'),
            'b4': ('Qwen3-4B-Base', 'bbd6fc8d23e8788d987b7b970cbb7bd31c826e38'),
            'i4': ('Qwen3-4B', '2c54d5a09e7e92d4f5126b92a5a457448c9593e6'),
            'b8': ('Qwen3-8B-Base', '932bc907a0f908fd665867dec24af47c2f57e719'),
            'i8': ('Qwen3-8B', '26028140be3ee69b82b1d1450179ab71bb1121b9')}
    models = {}
    for key, (name, revision) in pins.items():
        def fetch(rev):
            url = ('https://modelscope.cn/api/v1/models/Qwen/' + name + '/repo/files?' +
                   urllib.parse.urlencode({'Revision': rev, 'Recursive': 'true'}))
            with urllib.request.urlopen(url, timeout=60) as response:
                body = json.load(response)
            if body.get('Success') is not True:
                raise ValueError('ModelScope query failed')
            return url, body['Data']
        url, data = fetch(revision)
        if revision == 'master':
            candidates = {r['Revision'] for r in data['Files']
                          if r['Revision'].startswith(data['LatestCommitter']['ShortId'])}
            if len(candidates) != 1:
                raise ValueError('Cannot resolve full snapshot revision')
            revision = candidates.pop()
            url, data = fetch(revision)
        if data['LatestCommitter']['ShortId'] != revision[:8]:
            raise ValueError('Wrong pinned snapshot')
        files = {r['Path']: {field: r[field] for field in ('Size', 'Sha256', 'Revision')}
                 for r in data['Files'] if r['Type'] == 'blob'}
        if key not in ('i06', 'i4'):
            # Older local snapshots lack some docs; freeze every inference-critical file.
            files = {n: r for n, r in files.items() if n.endswith('.safetensors') or n in (
                'config.json', 'generation_config.json', 'model.safetensors.index.json',
                'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json')}
        models[key] = dict(repo='Qwen/' + name, revision=revision, source=url,
                           download_allowed=key in ('i06', 'i4'), files=files)
        print(key, revision, len(files), flush=True)
    evidence = json.loads((CONFIG.parents[2] / 'paper/iclr2027/internal/historical_evidence.json').read_text())['models']['qwen4_grpo']
    models['g4'] = dict(repo=evidence['hub_repo'],
                        revision=evidence['local_download_revision_evidence']['revision'],
                        source='Read-only public lllyx snapshot; revision from local download cache, hashes from historical evidence; not an original explicit revision pin',
                        download_allowed=False,
                        files={Path(p).name: {'Sha256': h} for p, h in evidence['recorded_asset_hashes'].items()})
    with CONFIG.open('x') as stream:
        json.dump(dict(version=1, models=models), stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-config', action='store_true')
    parser.add_argument('--model', choices=['b06', 'i06', 'b4', 'i4', 'g4', 'b8', 'i8'])
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.freeze_config:
        freeze_config()
    else:
        print(json.dumps(ensure_asset(args.model, args.download), indent=2), flush=True)

#!/usr/bin/env python3
"""Pinned original BF16 Llama assets from ModelScope only; no GPU or pip changes."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT = Path('/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd')
STUDENT = ROOT / 'models/Llama-3.2-1B-Instruct'
TEACHER = ROOT / 'models/Llama-3.2-3B-Instruct'
SPECS = {
    'student': {'repo': 'LLM-Research/Llama-3.2-1B-Instruct', 'path': STUDENT,
        'revision': 'd3e551343d4d81508a0d226656b826c217e463cd',
        'weights': {'model.safetensors': '1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f'}},
    'teacher': {'repo': 'LLM-Research/Llama-3.2-3B-Instruct', 'path': TEACHER,
        'revision': '4e7231b81c151c73632184994ac9a0149fcb22fd',
        'weights': {'model-00001-of-00002.safetensors': '13cbd6d16e927a0c5bad54102514e6e18b4a47b3a6eb911e39d678d328d19f55',
                    'model-00002-of-00002.safetensors': '7b770216613ac5c34d7c54bdff1fa616bc4e338a9d0b20af6303e48c295ee23c'}},
}
COMMON_FILES = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'LICENSE.txt', 'USE_POLICY.md', 'README.md')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select_files(source, spec):
    if source.get('Success') is not True:
        raise ValueError('ModelScope source query failed')
    records = {r['Path']: r for r in source['Data']['Files'] if r['Type'] == 'blob'}
    names = set(COMMON_FILES) | set(spec['weights'])
    if len(spec['weights']) > 1:
        names.add('model.safetensors.index.json')
    if not names.issubset(records):
        raise ValueError('Missing original model files')
    for name in names:
        row = records[name]
        if row['Size'] <= 0 or not re.fullmatch('[0-9a-f]{64}', row['Sha256']):
            raise ValueError(f'Invalid source record: {name}')
        if name in spec['weights'] and row['Sha256'] != spec['weights'][name]:
            raise ValueError(f'Unexpected weight identity: {name}')
    return {name: records[name] for name in sorted(names)}


def verify_file(path, record):
    if path.stat().st_size != record['Size'] or sha256(path) != record['Sha256']:
        raise ValueError(f'ModelScope size/hash mismatch: {path}')


def download(spec):
    directory = spec['path']
    directory.mkdir(parents=True, exist_ok=False)
    base = f"https://modelscope.cn/api/v1/models/{spec['repo']}"
    query = urllib.parse.urlencode({'Revision': spec['revision'], 'Recursive': 'true'})
    with urllib.request.urlopen(f'{base}/repo/files?{query}', timeout=60) as response:
        source = json.load(response)
    records = select_files(source, spec)
    (directory / 'source_metadata.json').write_text(json.dumps(source, indent=2) + '\n')
    for name, record in records.items():
        query = urllib.parse.urlencode({'Revision': spec['revision'], 'FilePath': name})
        target = directory / name
        partial = target.with_suffix(target.suffix + '.partial')
        print(f"Downloading {spec['repo']}@{spec['revision']} {name} ({record['Size']} bytes)", flush=True)
        with urllib.request.urlopen(f'{base}/repo?{query}', timeout=300) as response, partial.open('xb') as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        verify_file(partial, record)
        partial.rename(target)
    # SOURCE_REVISION records ModelScope, not a Hugging Face commit.
    (directory / 'SOURCE_REVISION').write_text(spec['revision'] + '\n')
    manifest = {'provider': 'modelscope', 'developer': 'Meta', 'repo': spec['repo'],
                'revision': spec['revision'], 'sha256': {str(p): sha256(p) for p in directory.iterdir() if p.is_file()}}
    (directory / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify_assets():
    protected = {}
    for spec in SPECS.values():
        directory = spec['path']
        manifest_path = directory / 'asset_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('provider') != 'modelscope' or manifest.get('repo') != spec['repo']
                or manifest.get('revision') != spec['revision']):
            raise ValueError('Wrong asset source')
        records = select_files(json.loads((directory / 'source_metadata.json').read_text()), spec)
        for name, record in records.items():
            verify_file(directory / name, record)
        expected = {str(directory / name) for name in records} | {
            str(directory / 'SOURCE_REVISION'), str(directory / 'source_metadata.json')}
        if set(manifest['sha256']) != expected:
            raise ValueError('Incomplete asset manifest')
        if (directory / 'SOURCE_REVISION').read_text().strip() != spec['revision']:
            raise ValueError('Wrong pinned revision')
        for name, digest in manifest['sha256'].items():
            if sha256(Path(name)) != digest:
                raise ValueError('Changed asset file')
        protected.update(manifest['sha256'])
        protected[str(manifest_path)] = sha256(manifest_path)
        config = json.loads((directory / 'config.json').read_text())
        gen = json.loads((directory / 'generation_config.json').read_text())
        if (config['architectures'] != ['LlamaForCausalLM'] or config['vocab_size'] != 128256
                or gen['eos_token_id'] != [128001, 128008, 128009]):
            raise ValueError('Unexpected Llama architecture or EOS')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json'):
        if sha256(STUDENT / name) != sha256(TEACHER / name):
            raise ValueError(f'Teacher/student token protocol differs: {name}')
    return protected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        for spec in SPECS.values():
            download(spec)
    protected = verify_assets()
    print(json.dumps({'passed': True, 'files_verified': len(protected), 'provider': 'modelscope'}), flush=True)


if __name__ == '__main__':
    main()

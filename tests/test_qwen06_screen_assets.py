import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))


def subject():
    return importlib.import_module('prepare_qwen06_screen_assets')


def test_exact_model_ids_and_roles():
    s = subject()
    specs = s.specifications()
    assert set(specs) == {'b06', 'i06', 'b4', 'i4', 'g4', 'b8', 'i8'}
    assert specs['i06']['repo'] == 'Qwen/Qwen3-0.6B'
    assert specs['i4']['repo'] == 'Qwen/Qwen3-4B'
    assert specs['g4']['repo'] == 'lllyx/Qwen3-4B-Base-GRPO'
    assert {k for k,v in specs.items() if v['download_allowed']} == {'i06','i4'}


def test_file_hash_mismatch_rejected(tmp_path):
    s = subject()
    p = tmp_path/'config.json'
    p.write_text('{}')
    with pytest.raises(ValueError):
        s.verify_files(tmp_path, {'config.json': {'Sha256':'0'*64, 'Size':2}})
    assert s.verify_files(tmp_path, {'config.json': {'Sha256':s.sha256(p), 'Size':2}})


def test_config_source_contains_full_revisions_and_hashes():
    s = subject()
    for key, spec in s.specifications().items():
        assert len(spec['revision']) == 40
        assert any(n.endswith('.safetensors') for n in spec['files'])
        assert 'tokenizer.json' in spec['files'] and 'config.json' in spec['files']
        for name,row in spec['files'].items():
            assert Path(name).name == name
            assert len(row['Sha256']) == 64


def test_existing_unowned_directory_never_adopted(tmp_path):
    s = subject()
    spec = {'repo':'Qwen/test', 'revision':'a'*40, 'files':{}}
    p = tmp_path/'model'
    p.mkdir()
    with pytest.raises(ValueError, match='ownership'):
        s.require_owned_download(p, spec)

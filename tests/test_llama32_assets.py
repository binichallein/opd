import importlib.util
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location('llama_assets', Path(__file__).parents[1] / 'scripts/prepare_llama32_assets.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_selection_requires_original_weights_and_rejects_missing_shard():
    m = module()
    spec = m.SPECS['teacher']
    records = [{'Path': name, 'Type': 'blob', 'Sha256': sha, 'Size': 1}
               for name, sha in spec['weights'].items()]
    records += [{'Path': name, 'Type': 'blob', 'Sha256': 'a' * 64, 'Size': 1}
                for name in m.COMMON_FILES + ('model.safetensors.index.json',)]
    chosen = m.select_files({'Success': True, 'Data': {'Files': records}}, spec)
    assert set(spec['weights']).issubset(chosen)
    with pytest.raises(ValueError):
        m.select_files({'Success': True, 'Data': {'Files': records[1:]}}, spec)
    records[0]['Sha256'] = 'b' * 64
    with pytest.raises(ValueError):
        m.select_files({'Success': True, 'Data': {'Files': records}}, spec)


def test_file_validation_checks_size_and_hash(tmp_path):
    m = module()
    file = tmp_path / 'file'
    file.write_bytes(b'abc')
    record = {'Size': 3, 'Sha256': m.sha256(file)}
    m.verify_file(file, record)
    with pytest.raises(ValueError):
        m.verify_file(file, {**record, 'Size': 2})
    with pytest.raises(ValueError):
        m.verify_file(file, {**record, 'Sha256': '0' * 64})

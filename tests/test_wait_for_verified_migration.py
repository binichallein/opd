import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/wait_for_verified_migration.py'
    assert path.exists(), 'Verified migration waiter is missing'
    spec = importlib.util.spec_from_file_location('migration_waiter_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_waits_without_acceptance(module, tmp_path):
    assert module.transfer_ready(tmp_path) is False


def test_failed_transfer_stops_queue(module, tmp_path):
    (tmp_path/'transfer_state_v2.json').write_text(json.dumps({'status':'failed','error':'hash mismatch'}))
    with pytest.raises(RuntimeError, match='hash mismatch'):
        module.transfer_ready(tmp_path)


def accepted(tmp_path):
    manifest = tmp_path/'migration_manifest.json'
    manifest.write_text(json.dumps({'files':[{'size_bytes':42}]}))
    result = dict(passed=True,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  files_verified=1,total_bytes=42)
    (tmp_path/'transfer_acceptance.json').write_text(json.dumps(result))
    return result


def test_only_complete_hash_bound_acceptance_is_ready(module, tmp_path):
    result = accepted(tmp_path)
    assert module.transfer_ready(tmp_path) is True
    for key, value in [('passed',False),('manifest_sha256','changed'),('files_verified',0),('total_bytes',0)]:
        (tmp_path/'transfer_acceptance.json').write_text(json.dumps({**result,key:value}))
        with pytest.raises(ValueError):
            module.transfer_ready(tmp_path)

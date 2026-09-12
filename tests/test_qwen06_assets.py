import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_qwen06_assets.py"


def module():
    assert SCRIPT.is_file(), "Qwen06 asset preparation is not implemented"
    spec = importlib.util.spec_from_file_location("qwen06_assets", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_fixed_official_revision_and_weight_identity_are_accepted():
    mod = module()
    source = {"id": mod.REPO, "sha": mod.REVISION, "siblings": [
        {"rfilename": "model.safetensors", "lfs": {"sha256": mod.WEIGHT_SHA}}]}
    mod.validate_source(source)
    for key in ["id", "sha"]:
        bad = {**source, key: "other"}
        with pytest.raises(ValueError):
            mod.validate_source(bad)
    source["siblings"][0]["lfs"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="weight"):
        mod.validate_source(source)


def test_file_verification_handles_git_blobs_and_lfs_without_changing_bytes(tmp_path):
    mod = module()
    path = tmp_path / "test.json"
    content = b'{"test":true}\n'
    path.write_bytes(content)
    git_sha = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    sha = hashlib.sha256(content).hexdigest()
    mod.verify_file(path, {"size": len(content), "blobId": git_sha})
    mod.verify_file(path, {"size": len(content), "lfs": {"sha256": sha}})
    with pytest.raises(ValueError):
        mod.verify_file(path, {"size": len(content), "lfs": {"sha256": "0" * 64}})
    assert path.read_bytes() == content


def test_tokenizer_mapping_and_prompt_template_must_align(tmp_path):
    mod = module()
    roots = [tmp_path / n for n in ["student", "teacher", "reference"]]
    data = {"model": {"vocab": {"a": 0, "b": 1}}, "added_tokens": []}
    for root in roots:
        root.mkdir()
        (root / "tokenizer.json").write_text(json.dumps(data))
        (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "fixed"}))
    mod.tokenizer_alignment(*roots)
    (roots[0] / "tokenizer.json").write_text(json.dumps({**data, "model": {"vocab": {"a": 1, "b": 0}}}))
    with pytest.raises(ValueError, match="tokenizer"):
        mod.tokenizer_alignment(*roots)
    (roots[0] / "tokenizer.json").write_text(json.dumps(data))
    (roots[0] / "tokenizer_config.json").write_text(json.dumps({"chat_template": "changed"}))
    with pytest.raises(ValueError, match="template"):
        mod.tokenizer_alignment(*roots)

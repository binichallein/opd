import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.parse

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_qwen4_assets.py"
REVISION = "bbd6fc8d23e8788d987b7b970cbb7bd31c826e38"
WEIGHTS = {
    "model-00001-of-00003.safetensors": "4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241",
    "model-00002-of-00003.safetensors": "f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826",
    "model-00003-of-00003.safetensors": "c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965",
}
FILES = {
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "vocab.json", "merges.txt", "README.md", "LICENSE", "configuration.json",
    "model.safetensors.index.json", *WEIGHTS,
}
# Relevant fields from the live pinned ModelScope /repo/files response.
LIVE_FILES = {
    "config.json": (727, "da48f956e3b8d2166900f2beaf16406a2d51ba65", "304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1"),
    "configuration.json": (73, "da48f956e3b8d2166900f2beaf16406a2d51ba65", "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9"),
    "generation_config.json": (138, "da48f956e3b8d2166900f2beaf16406a2d51ba65", "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"),
    "LICENSE": (11343, REVISION, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    "merges.txt": (1671853, "d9aa38ad95ba56308836e1d6e71c3c9c3d62fea1", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model-00001-of-00003.safetensors": (3957900840, "63ad053162aae96de39eb610be500750ef252099", WEIGHTS["model-00001-of-00003.safetensors"]),
    "model-00002-of-00003.safetensors": (3987450520, "63ad053162aae96de39eb610be500750ef252099", WEIGHTS["model-00002-of-00003.safetensors"]),
    "model-00003-of-00003.safetensors": (99630640, "63ad053162aae96de39eb610be500750ef252099", WEIGHTS["model-00003-of-00003.safetensors"]),
    "model.safetensors.index.json": (32819, "da48f956e3b8d2166900f2beaf16406a2d51ba65", "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c"),
    "README.md": (2937, "bfedc9bb35d0793e2163dd83819f452bc2e501ba", "9fd20ab531a1dc75ae18fcde658dd69d04173fdb93311091c38a7098e3d4b4a1"),
    "tokenizer.json": (7031645, "da48f956e3b8d2166900f2beaf16406a2d51ba65", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "tokenizer_config.json": (9678, "9cdc0ec75a56202a429bdf270b721c9e5bb9def9", "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
    "vocab.json": (2776833, "da48f956e3b8d2166900f2beaf16406a2d51ba65", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}


def test_asset_preparation_script_exists():
    assert SCRIPT.is_file(), "Pinned Qwen4 asset preparation is not implemented"


@pytest.fixture
def assets(monkeypatch):
    assert SCRIPT.is_file(), "Pinned Qwen4 asset preparation is not implemented"
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("qwen4_assets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def no_network(*args, **kwargs):
        pytest.fail("Asset tests must never contact a remote service")

    monkeypatch.setattr(module.urllib.request, "urlopen", no_network)
    return module


def source_for(weights):
    return {"Success": True, "Data": {"LatestCommitter": {"ShortId": "bbd6fc8d"}, "Files": [
        {"Path": name, "Type": "blob", "Revision": revision, "Size": size,
         "Sha256": weights.get(name, digest)} for name, (size, revision, digest) in LIVE_FILES.items()
    ]}}


def test_exact_modelscope_identity_and_historical_paths(assets):
    assert assets.REPO == "Qwen/Qwen3-4B-Base"
    assert assets.REVISION == REVISION
    assert assets.WEIGHTS == WEIGHTS
    assert assets.ROOT == Path("/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd")
    assert assets.STUDENT == assets.ROOT / "models/Qwen3-4B-Base"
    assert assets.TEACHER == assets.ROOT / "models/Qwen3-4B-Base-GRPO"
    import prepare_qwen06_assets
    assert assets.REFERENCE == prepare_qwen06_assets.REFERENCE
    assert assets.tokenizer_alignment is prepare_qwen06_assets.tokenizer_alignment
    assert assets.HISTORICAL_MANIFEST == assets.ROOT / (
        "runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/artifact_hashes.sha256")


def test_selects_all_non_code_assets_and_verifies_pinned_response(assets):
    source = source_for(WEIGHTS)
    source["Data"]["Files"].append({"Path": "custom_model.py", "Type": "blob"})
    assert set(assets.select_files(source)) == FILES
    assert assets.FILE_SHA256 == {name: record[2] for name, record in LIVE_FILES.items()}


def test_live_metadata_file_revision_is_last_modifying_commit_not_snapshot(assets):
    selected = assets.select_files(source_for(WEIGHTS))
    assert set(selected) == FILES
    assert selected["config.json"]["Revision"] == "da48f956e3b8d2166900f2beaf16406a2d51ba65"
    assert selected["model-00001-of-00003.safetensors"]["Revision"] == "63ad053162aae96de39eb610be500750ef252099"
    assert selected["tokenizer_config.json"]["Revision"] == "9cdc0ec75a56202a429bdf270b721c9e5bb9def9"
    assert selected["LICENSE"]["Revision"] == REVISION


@pytest.mark.parametrize("fault", ["api_failure", "revision", "file_revision", "missing_revision", "weight",
                                    "missing_shard", "missing_config", "duplicate", "hash", "size",
                                    "snapshot_short_id", "nonweight_hash"])
def test_rejects_incomplete_or_unpinned_api_response(assets, fault):
    source = source_for(WEIGHTS)
    rows = source["Data"]["Files"]
    if fault == "api_failure":
        source["Success"] = False
    elif fault == "revision":
        source["Data"]["Revision"] = "master"
    elif fault == "file_revision":
        source["Data"]["Revision"] = REVISION
        rows[0]["Revision"] = "master"
    elif fault == "snapshot_short_id":
        source["Data"]["LatestCommitter"]["ShortId"] = "12345678"
    elif fault == "nonweight_hash":
        next(row for row in rows if row["Path"] == "config.json")["Sha256"] = "b" * 64
    elif fault == "missing_revision":
        for row in rows:
            row.pop("Revision")
    elif fault == "weight":
        next(row for row in rows if row["Path"] in WEIGHTS)["Sha256"] = "b" * 64
    elif fault in ("missing_shard", "missing_config"):
        name = next(iter(WEIGHTS)) if fault == "missing_shard" else "configuration.json"
        rows[:] = [row for row in rows if row["Path"] != name]
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "hash":
        rows[0]["Sha256"] = "invalid"
    else:
        rows[0]["Size"] = 0
    with pytest.raises(ValueError):
        assets.select_files(source)


def test_file_validation_checks_hash_not_just_size(assets, tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"abc")
    record = {"Size": 3, "Sha256": assets.sha256(path)}
    assets.verify_file(path, record)
    path.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="size/hash"):
        assets.verify_file(path, record)
    path.unlink()
    with pytest.raises(ValueError, match="size/hash"):
        assets.verify_file(path, record)


@pytest.fixture
def prepared(assets, tmp_path, monkeypatch):
    student, teacher, reference = [tmp_path / name for name in ("student", "teacher", "reference")]
    historical = tmp_path / "historical/artifact_hashes.sha256"
    for key, value in {"STUDENT": student, "TEACHER": teacher, "REFERENCE": reference,
                       "HISTORICAL_MANIFEST": historical}.items():
        monkeypatch.setattr(assets, key, value)
    config = {"architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 32768}
    tokenizer = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
                 "added_tokens": []}
    contents = {name: (name + "\n").encode() for name in FILES}
    contents.update({
        "config.json": json.dumps(config).encode(),
        "generation_config.json": b'{"eos_token_id": 151643}',
        "tokenizer_config.json": b'{"chat_template": "fixed"}',
        "tokenizer.json": json.dumps(tokenizer).encode(),
        "vocab.json": b'{"a": 0, "b": 1}',
        "configuration.json": b'{}',
        "model.safetensors.index.json": json.dumps({"weight_map": {
            f"layer.{i}": name for i, name in enumerate(WEIGHTS)}}).encode(),
    })
    weights = {name: hashlib.sha256(contents[name]).hexdigest() for name in WEIGHTS}
    monkeypatch.setattr(assets, "WEIGHTS", weights)
    monkeypatch.setattr(assets, "FILE_SHA256", {
        name: hashlib.sha256(content).hexdigest() for name, content in contents.items()})
    source = source_for(weights)
    for row in source["Data"]["Files"]:
        row.update(Size=len(contents[row["Path"]]),
                   Sha256=hashlib.sha256(contents[row["Path"]]).hexdigest())
    for directory in (teacher, reference):
        directory.mkdir()
        for name, content in contents.items():
            (directory / name).write_bytes(content)
    historical.parent.mkdir()
    teacher_hashes = {str(teacher / name): assets.sha256(teacher / name) for name in FILES}
    historical.write_text("".join(f"{digest}  {name}\n" for name, digest in teacher_hashes.items())
                          + "0" * 64 + "  /unrelated/data.parquet\n")
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        parsed = urllib.parse.urlsplit(url)
        assert parsed.scheme == "https" and parsed.netloc == "modelscope.cn"
        assert parsed.path.startswith("/api/v1/models/Qwen/Qwen3-4B-Base/repo")
        query = urllib.parse.parse_qs(parsed.query)
        assert query["Revision"] == [REVISION]
        if parsed.path.endswith("/files"):
            assert query["Recursive"] == ["true"]
            return io.BytesIO(json.dumps(source).encode())
        return io.BytesIO(contents[query["FilePath"][0]])

    monkeypatch.setattr(assets.urllib.request, "urlopen", fake_urlopen)
    return assets, source, contents, teacher_hashes, urls


def test_download_student_only_and_verify_complete_absolute_hash_map(prepared):
    assets, _, _, teacher_hashes, urls = prepared
    manifest = assets.download()
    assert len(urls) == len(FILES) + 1
    assert manifest["provider"] == "modelscope"
    assert manifest["developer"] == "Qwen"
    assert manifest["repo"] == assets.REPO and manifest["revision"] == REVISION
    request_url = manifest["source_request_url"]
    assert request_url == urls[0]
    assert urllib.parse.parse_qs(urllib.parse.urlsplit(request_url).query)["Revision"] == [REVISION]
    assert not (assets.STUDENT / "HF_REVISION").exists()
    assert (assets.STUDENT / "SOURCE_REVISION").read_text().strip() == REVISION
    expected = {str(assets.STUDENT / name) for name in FILES | {"SOURCE_REVISION", "source_metadata.json"}}
    assert set(manifest["sha256"]) == expected
    protected = assets.verify_assets()
    assert expected | {str(assets.STUDENT / "asset_manifest.json")} <= protected.keys()
    assert teacher_hashes.items() <= protected.items()
    assert str(assets.HISTORICAL_MANIFEST) in protected
    assert "/unrelated/data.parquet" not in protected
    assert all(Path(name).is_absolute() and assets.sha256(Path(name)) == digest
               for name, digest in protected.items())
    assert len(urls) == len(FILES) + 1, "Verification must be offline"


def test_existing_destination_is_never_overwritten(prepared):
    assets, _, _, _, urls = prepared
    assets.STUDENT.mkdir()
    sentinel = assets.STUDENT / "existing"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        assets.download()
    assert sentinel.read_text() == "keep" and urls == []


def test_download_rejects_unpinned_metadata_before_fetching_assets(prepared):
    assets, source, _, _, urls = prepared
    source["Data"]["Revision"] = "master"
    with pytest.raises(ValueError, match="revision"):
        assets.download()
    assert len(urls) == 1
    assert not (assets.STUDENT / "SOURCE_REVISION").exists()
    assert not (assets.STUDENT / "asset_manifest.json").exists()


def test_download_stops_on_corrupt_bytes_and_retains_failed_attempt(prepared):
    assets, _, contents, _, urls = prepared
    contents["LICENSE"] = b"corrupt"
    with pytest.raises(ValueError, match="size/hash"):
        assets.download()
    assert len(urls) == 2
    assert (assets.STUDENT / "LICENSE.partial").exists()
    assert not (assets.STUDENT / "asset_manifest.json").exists()
    with pytest.raises(FileExistsError):
        assets.download()


def rewrite_student_file(assets, name, value):
    target = assets.STUDENT / name
    target.write_text(json.dumps(value))
    source_path = assets.STUDENT / "source_metadata.json"
    source = json.loads(source_path.read_text())
    row = next(row for row in source["Data"]["Files"] if row["Path"] == name)
    row.update(Size=target.stat().st_size, Sha256=assets.sha256(target))
    assets.FILE_SHA256[name] = assets.sha256(target)
    source_path.write_text(json.dumps(source))
    manifest_path = assets.STUDENT / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in (target, source_path):
        manifest["sha256"][str(path)] = assets.sha256(path)
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("fault", ["provider", "repo", "revision", "hash_coverage", "extra_file",
                                    "request_url", "missing_request_url",
                                    "bytes", "revision_marker", "source_revision", "shard_index"])
def test_offline_verification_rejects_changed_student(prepared, fault):
    assets, _, _, _, _ = prepared
    assets.download()
    manifest_path = assets.STUDENT / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if fault in ("provider", "repo", "revision"):
        manifest[fault] = "wrong"
        manifest_path.write_text(json.dumps(manifest))
    elif fault == "hash_coverage":
        manifest["sha256"].pop(str(assets.STUDENT / "LICENSE"))
        manifest_path.write_text(json.dumps(manifest))
    elif fault in ("request_url", "missing_request_url"):
        if fault == "request_url":
            manifest["source_request_url"] = manifest["source_request_url"].replace(REVISION, "master")
        else:
            manifest.pop("source_request_url")
        manifest_path.write_text(json.dumps(manifest))
    elif fault == "extra_file":
        (assets.STUDENT / "HF_REVISION").write_text(REVISION)
    elif fault == "bytes":
        (assets.STUDENT / next(iter(WEIGHTS))).write_bytes(b"corrupt")
    elif fault == "revision_marker":
        (assets.STUDENT / "SOURCE_REVISION").write_text("master")
    elif fault == "source_revision":
        path = assets.STUDENT / "source_metadata.json"
        source = json.loads(path.read_text())
        source["Data"]["Files"][0]["Revision"] = "master"
        path.write_text(json.dumps(source))
    else:
        rewrite_student_file(assets, "model.safetensors.index.json",
                             {"weight_map": {"layer": "../other.safetensors"}})
    with pytest.raises(ValueError):
        assets.verify_assets()


@pytest.mark.parametrize("key,value", [("architectures", ["OtherForCausalLM"]),
                                      ("max_position_embeddings", 16384)])
def test_requires_qwen3_architecture_and_32k_context(prepared, key, value):
    assets, _, _, _, _ = prepared
    assets.download()
    config = json.loads((assets.STUDENT / "config.json").read_text())
    config[key] = value
    rewrite_student_file(assets, "config.json", config)
    with pytest.raises(ValueError, match="architecture|context"):
        assets.verify_assets()


@pytest.mark.parametrize("fault", ["changed_weight", "missing_manifest", "missing_shard_hash",
                                    "missing_tokenizer_hash", "extra_weight", "duplicate_hash"])
def test_teacher_must_match_complete_historical_manifest(prepared, fault):
    assets, _, _, _, _ = prepared
    assets.download()
    if fault == "changed_weight":
        (assets.TEACHER / next(iter(WEIGHTS))).write_bytes(b"changed")
    elif fault == "missing_manifest":
        assets.HISTORICAL_MANIFEST.unlink()
    elif fault in ("missing_shard_hash", "missing_tokenizer_hash"):
        name = next(iter(WEIGHTS)) if fault == "missing_shard_hash" else "tokenizer.json"
        lines = assets.HISTORICAL_MANIFEST.read_text().splitlines()
        assets.HISTORICAL_MANIFEST.write_text("\n".join(line for line in lines
                                                        if not line.endswith("/" + name)) + "\n")
    elif fault == "extra_weight":
        (assets.TEACHER / "extra.safetensors").write_bytes(b"untracked")
    else:
        text = assets.HISTORICAL_MANIFEST.read_text()
        assets.HISTORICAL_MANIFEST.write_text(text + text.splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        assets.verify_assets()


def test_runs_real_historical_tokenizer_alignment(prepared):
    assets, _, _, _, _ = prepared
    assets.download()
    assets.verify_assets()
    path = assets.REFERENCE / "tokenizer_config.json"
    path.write_text('{"chat_template":"different"}')
    with pytest.raises(ValueError, match="template"):
        assets.verify_assets()


def test_historical_teacher_only_token_exception_is_preserved(prepared):
    assets, _, _, _, _ = prepared
    assets.download()
    teacher_path = assets.TEACHER / "tokenizer.json"
    tokenizer = json.loads(teacher_path.read_text())
    tokenizer["added_tokens"] = [
        {"id": i, "content": text, "single_word": False, "lstrip": False, "rstrip": False,
         "normalized": False, "special": False}
        for i, text in [(151665, "<tool_response>"), (151666, "</tool_response>"),
                        (151667, "<think>"), (151668, "</think>")]]
    teacher_path.write_text(json.dumps(tokenizer))
    lines = assets.HISTORICAL_MANIFEST.read_text().splitlines()
    assets.HISTORICAL_MANIFEST.write_text("\n".join(
        f"{assets.sha256(teacher_path)}  {teacher_path}" if line.endswith(str(teacher_path)) else line
        for line in lines) + "\n")
    assert str(teacher_path) in assets.verify_assets()


@pytest.mark.parametrize("download", [False, True])
def test_main_downloads_only_student_then_verifies(assets, monkeypatch, capsys, download):
    calls = []
    monkeypatch.setattr(assets, "download", lambda: calls.append("student"))

    def verify():
        calls.append("verify")
        return {"/verified/model": "a" * 64}

    monkeypatch.setattr(assets, "verify_assets", verify)
    assets.main(["--download"] if download else [])
    assert calls == (["student", "verify"] if download else ["verify"])
    assert json.loads(capsys.readouterr().out) == {
        "passed": True, "files_verified": 1, "provider": "modelscope"}

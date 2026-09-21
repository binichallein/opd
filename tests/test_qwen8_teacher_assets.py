"""Offline contracts for the official pinned 8B Base teacher preparer."""

import copy
import errno
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.parse

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_qwen8_teacher_assets.py"
REVISION = "932bc907a0f908fd665867dec24af47c2f57e719"
INITIAL = "26fb095fa44e582c29507c6ec61cba1e2ed67bf6"
WEIGHT_REVISION = "9ed78453e256ac6407ffba7369d9815ef4697fcc"
# ModelScope /repo/files, queried at master and again at the full revision.
LIVE_FILES = {
    ".gitattributes": (2127, INITIAL, "088e25d6bceaa8943f4f292006210358292349cc36e7578f80c62fce75535f46"),
    "README.md": (2938, REVISION, "0eb9f991baa4f92c9ef708d1e9c7a01ae5b19d05516d38f8833549aa8b06a07e"),
    "config.json": (729, INITIAL, "3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a"),
    "configuration.json": (73, INITIAL, "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9"),
    "generation_config.json": (138, INITIAL, "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"),
    "merges.txt": (1671853, "00b4a13ba84ba74b07cc974f4f3c6c591d36ec44", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model-00001-of-00005.safetensors": (3996250744, WEIGHT_REVISION, "9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b"),
    "model-00002-of-00005.safetensors": (3993160032, WEIGHT_REVISION, "9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f"),
    "model-00003-of-00005.safetensors": (3959604768, WEIGHT_REVISION, "c7dd5a191c8da555def0714550375b3c0117751add408fcdda0c2a79ac0186bc"),
    "model-00004-of-00005.safetensors": (3187841392, WEIGHT_REVISION, "ad8b708792105133e03fe19e2d56c89c709f677f01285f8b3ede8947d59fd9fe"),
    "model-00005-of-00005.safetensors": (1244659840, WEIGHT_REVISION, "fbf24915d47ea030bb68ab0b9488f4515a907185baa6dc26837c9c3f2326a550"),
    "model.safetensors.index.json": (32878, INITIAL, "cd39d5e44c5aadfd1954b598883aea1c4a98822a2f7723d214ef2278b7fec2e4"),
    "tokenizer.json": (7031645, INITIAL, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "tokenizer_config.json": (9678, "dfb1b602fb76fd49e0884d5b3f126145dbe75ac4", "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
    "vocab.json": (2776833, INITIAL, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def test_preparer_exists():
    assert SCRIPT.is_file(), "Pinned official 8B teacher preparer is not implemented"


@pytest.fixture
def assets(monkeypatch):
    assert SCRIPT.is_file(), "Pinned official 8B teacher preparer is not implemented"
    monkeypatch.syspath_prepend(str(SCRIPT.parent))

    def forbidden(*args, **kwargs):
        pytest.fail("Unit tests must not use the network or launch external processes")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    spec = importlib.util.spec_from_file_location("qwen8_teacher_assets_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    return module


def source_for():
    return {"Success": True, "Data": {"LatestCommitter": {"Id": "", "ShortId": REVISION[:8]},
            "Files": [{"Path": name, "Type": "blob", "Size": size,
                       "Revision": revision, "Sha256": digest}
                      for name, (size, revision, digest) in LIVE_FILES.items()]}}


def test_pins_complete_official_snapshot_and_reuses_helpers(assets):
    import prepare_qwen4_assets as current4BBase
    assert assets.REPO == "Qwen/Qwen3-8B-Base"
    assert assets.REVISION == REVISION and len(assets.REVISION) == 40
    assert assets.ROOT == current4BBase.ROOT
    assert assets.TEACHER == assets.ROOT / "models/Qwen3-8B-Base"
    assert assets.current4BBase is current4BBase
    assert assets.sha256 is current4BBase.sha256
    assert assets.verify_file is current4BBase.verify_file
    assert assets.verify_index is current4BBase.verify_index
    assert assets.FILE_SHA256 == {name: row[2] for name, row in LIVE_FILES.items()}
    assert assets.FILE_SIZES == {name: row[0] for name, row in LIVE_FILES.items()}
    assert assets.FILE_REVISIONS == {name: row[1] for name, row in LIVE_FILES.items()}
    assert set(assets.WEIGHTS) == {name for name in LIVE_FILES if name.endswith(".safetensors")}
    assert set(assets.select_files(source_for())) == set(LIVE_FILES)
    for name in TOKENIZER_FILES:
        assert assets.FILE_SHA256[name] == current4BBase.FILE_SHA256[name]


@pytest.mark.parametrize("fault", [
    "failure", "missing_data", "missing_files", "snapshot", "short_id", "full_id",
    "revision", "missing_revision", "size", "boolean_size", "hash", "missing",
    "duplicate", "extra", "traversal", "absolute", "tree", "row_type",
])
def test_metadata_fails_closed(assets, fault):
    source = source_for()
    data, rows = source["Data"], source["Data"]["Files"]
    if fault == "failure":
        source["Success"] = False
    elif fault == "missing_data":
        source["Data"] = None
    elif fault == "missing_files":
        data.pop("Files")
    elif fault == "snapshot":
        data["Revision"] = "master"
    elif fault == "short_id":
        data["LatestCommitter"]["ShortId"] = "12345678"
    elif fault == "full_id":
        data["LatestCommitter"]["Id"] = REVISION[:8] + "0" * 32
    elif fault == "revision":
        rows[0]["Revision"] = "b" * 40
    elif fault == "missing_revision":
        rows[0].pop("Revision")
    elif fault in ("size", "boolean_size"):
        rows[0]["Size"] = True if fault == "boolean_size" else rows[0]["Size"] + 1
    elif fault == "hash":
        rows[0]["Sha256"] = "b" * 64
    elif fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "extra":
        rows.append({**rows[0], "Path": "untracked.py"})
    elif fault in ("traversal", "absolute"):
        rows[0]["Path"] = "../old/config.json" if fault == "traversal" else "/old/config.json"
    elif fault == "tree":
        rows[0]["Type"] = "tree"
    else:
        rows[0] = None
    with pytest.raises(ValueError):
        assets.select_files(source)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data)), **(headers or {})}

    def getcode(self):
        return self.status


@pytest.fixture
def prepared(assets, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "TEACHER", tmp_path / "models/Qwen3-8B-Base")
    student = tmp_path / "models/Qwen3-4B-Base"
    monkeypatch.setattr(assets.current4BBase, "STUDENT", student)
    contents = {name: (name + "\n").encode() for name in LIVE_FILES}
    contents.update({
        "config.json": json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                                   "max_position_embeddings": 32768, "eos_token_id": 151643}).encode(),
        "generation_config.json": b'{"eos_token_id": 151643}',
        "tokenizer_config.json": b'{"eos_token": "<|endoftext|>"}',
        "tokenizer.json": b'{"added_tokens": [{"id": 151643, "content": "<|endoftext|>", "special": true}]}',
        "model.safetensors.index.json": json.dumps({"weight_map": {
            f"layer.{i}": name for i, name in enumerate(assets.WEIGHTS)}}).encode(),
    })
    digests = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
    monkeypatch.setattr(assets, "FILE_SHA256", digests)
    monkeypatch.setattr(assets, "FILE_SIZES", {name: len(value) for name, value in contents.items()})
    monkeypatch.setattr(assets, "WEIGHTS", {name: digests[name] for name in assets.WEIGHTS})
    source = source_for()
    for row in source["Data"]["Files"]:
        row.update(Size=assets.FILE_SIZES[row["Path"]], Sha256=digests[row["Path"]])
    student.mkdir(parents=True)
    old_contents = {str(student / name): contents[name] for name in (
        *TOKENIZER_FILES, "config.json", "generation_config.json")}
    old_contents[str(tmp_path / "historical.manifest")] = b"historical protection\n"
    for name, value in old_contents.items():
        Path(name).write_bytes(value)
    old_hashes = {name: assets.sha256(Path(name)) for name in old_contents}
    calls = []

    def verify_current():
        calls.append("current4BBase.verify_assets")
        for name, digest in old_hashes.items():
            if not Path(name).is_file() or assets.sha256(Path(name)) != digest:
                raise ValueError("Changed current4B asset")
        return dict(old_hashes)

    monkeypatch.setattr(assets.current4BBase, "verify_assets", verify_current)
    requests = []

    def fetch(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        parsed = urllib.parse.urlsplit(url)
        query = urllib.parse.parse_qs(parsed.query)
        assert parsed.scheme == "https" and parsed.netloc == "modelscope.cn"
        assert parsed.path.startswith("/api/v1/models/Qwen/Qwen3-8B-Base/repo")
        assert query["Revision"] == [REVISION]
        requests.append(request)
        if parsed.path.endswith("/files"):
            assert query["Recursive"] == ["true"]
            return Response(json.dumps(source).encode())
        name = query["FilePath"][0]
        value = contents[name]
        offset = 0
        if not isinstance(request, str) and request.get_header("Range"):
            offset = int(request.get_header("Range").removeprefix("bytes=").removesuffix("-"))
        headers = {"Content-Range": f"bytes {offset}-{len(value)-1}/{len(value)}"} if offset else {}
        return Response(value[offset:], 206 if offset else 200, headers)

    monkeypatch.setattr(assets.urllib.request, "urlopen", fetch)
    return assets, source, contents, old_hashes, calls, requests, fetch


def test_download_and_offline_verify_return_complete_protected_hashes(prepared):
    assets, _, contents, old_hashes, calls, requests, _ = prepared
    protected = assets.download()
    assert len(requests) == len(contents) + 1
    assert old_hashes.items() <= protected.items()
    assert calls
    files = {str(path) for path in assets.TEACHER.iterdir()}
    assert files <= protected.keys()
    assert {assets.TEACHER / name for name in contents} <= set(assets.TEACHER.iterdir())
    manifest_path = assets.TEACHER / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["repo"] == assets.REPO and manifest["revision"] == REVISION
    assert manifest["provider"] == "modelscope" and manifest["developer"] == "Qwen"
    assert manifest["source_request_url"] == assets.SOURCE_REQUEST_URL
    assert set(manifest["sha256"]) == files - {str(manifest_path)}
    assert set(manifest["size_bytes"]) == set(manifest["sha256"])
    for name, digest in protected.items():
        assert Path(name).is_absolute() and assets.sha256(Path(name)) == digest
    for name, size in manifest["size_bytes"].items():
        assert Path(name).stat().st_size == size
    assert (assets.TEACHER / "SOURCE_REVISION").read_text() == REVISION + "\n"
    assert not (assets.TEACHER / "HF_REVISION").exists()
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in assets.TEACHER.iterdir()}
    assert assets.verify_assets() == protected
    assert assets.download() == protected
    assert len(requests) == len(contents) + 1, "Completed downloads and verification must be offline"
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize("fault", [
    "missing", "invalid_json", "duplicate_json_key", "not_object", "provider", "developer",
    "repo", "revision", "request_url", "missing_hash", "extra_hash", "hash", "missing_size",
    "size", "boolean_size", "source_missing", "source_corrupt", "source_unpinned",
    "marker", "missing_file", "changed_file", "extra_file", "symlink", "manifest_symlink",
])
def test_verification_fails_closed_on_corrupt_or_missing_manifest_and_assets(prepared, fault):
    assets, _, _, _, _, requests, _ = prepared
    assets.download()
    path = assets.TEACHER / "asset_manifest.json"
    manifest = json.loads(path.read_text())
    name = str(assets.TEACHER / "config.json")
    if fault == "missing":
        path.unlink()
    elif fault in ("invalid_json", "duplicate_json_key", "not_object"):
        path.write_text({"invalid_json": "{", "duplicate_json_key": '{"repo": 1, "repo": 2}',
                         "not_object": "[]"}[fault])
    elif fault in ("provider", "developer", "repo", "revision", "request_url"):
        manifest["source_request_url" if fault == "request_url" else fault] = "wrong"
        path.write_text(json.dumps(manifest))
    elif fault in ("missing_hash", "extra_hash", "hash", "missing_size", "size", "boolean_size"):
        if fault == "missing_hash":
            manifest["sha256"].pop(name)
        elif fault == "extra_hash":
            manifest["sha256"]["/outside/file"] = "a" * 64
        elif fault == "hash":
            manifest["sha256"][name] = "a" * 64
        elif fault == "missing_size":
            manifest["size_bytes"].pop(name)
        else:
            manifest["size_bytes"][name] = True if fault == "boolean_size" else 1
        path.write_text(json.dumps(manifest))
    elif fault.startswith("source_"):
        source_path = assets.TEACHER / "source_metadata.json"
        if fault == "source_missing":
            source_path.unlink()
        elif fault == "source_corrupt":
            source_path.write_text("{")
        else:
            source = json.loads(source_path.read_text())
            source["Data"]["Revision"] = "master"
            source_path.write_text(json.dumps(source))
    elif fault == "marker":
        (assets.TEACHER / "SOURCE_REVISION").write_text("master\n")
    elif fault == "missing_file":
        Path(name).unlink()
    elif fault == "changed_file":
        Path(name).write_bytes(b"x" * Path(name).stat().st_size)
    elif fault == "extra_file":
        (assets.TEACHER / "HF_REVISION").write_text(REVISION)
    else:
        target = path if fault == "manifest_symlink" else Path(name)
        external = assets.ROOT / "external"
        external.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(external)
    n = len(requests)
    with pytest.raises(ValueError):
        assets.verify_assets()
    assert len(requests) == n


def test_download_refuses_existing_corrupt_manifest_without_repair(prepared):
    assets, _, _, _, _, requests, _ = prepared
    assets.download()
    path = assets.TEACHER / "asset_manifest.json"
    path.write_text("{")
    n = len(requests)
    with pytest.raises(ValueError):
        assets.download()
    assert path.read_text() == "{" and len(requests) == n


def test_old_assets_are_verified_before_any_new_write_or_request(prepared):
    assets, _, _, old_hashes, _, requests, _ = prepared
    Path(next(iter(old_hashes))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="current4B"):
        assets.download()
    assert not assets.TEACHER.exists() and not requests


def test_verify_propagates_current4b_helper_failure(prepared):
    assets, _, _, old_hashes, _, _, _ = prepared
    assets.download()
    Path(next(iter(old_hashes))).unlink()
    with pytest.raises(ValueError, match="current4B"):
        assets.verify_assets()


@pytest.mark.parametrize("fault", ["foreign", "symlink", "parent_symlink", "old_destination"])
def test_download_never_writes_foreign_or_linked_directories(prepared, monkeypatch, fault):
    assets, _, _, old_hashes, _, requests, _ = prepared
    old = assets.current4BBase.STUDENT
    if fault == "foreign":
        assets.TEACHER.mkdir()
        (assets.TEACHER / "sentinel").write_text("keep")
    elif fault == "symlink":
        assets.TEACHER.symlink_to(old, target_is_directory=True)
    elif fault == "parent_symlink":
        link = assets.ROOT / "alias"
        link.symlink_to(assets.ROOT / "models", target_is_directory=True)
        monkeypatch.setattr(assets, "TEACHER", link / "Qwen3-8B-Base")
    else:
        monkeypatch.setattr(assets, "TEACHER", old)
    with pytest.raises((ValueError, FileExistsError)):
        assets.download()
    assert not requests
    assert all(assets.sha256(Path(name)) == digest for name, digest in old_hashes.items())
    if fault == "foreign":
        assert (assets.TEACHER / "sentinel").read_text() == "keep"


def start_interrupted(prepared, monkeypatch, prefix=b"."):
    assets, _, contents, _, _, _, fetch = prepared
    name = ".gitattributes"

    class Interrupted(Response):
        def read(self, size=-1):
            if self.tell():
                raise urllib.error.URLError("interrupted")
            return super().read(len(prefix))

    def interrupt(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        if "FilePath=" not in url:
            return fetch(request, timeout)
        return Interrupted(prefix, headers={"Content-Length": str(len(contents[name]))})

    with monkeypatch.context() as patch:
        patch.setattr(assets, "MAX_ATTEMPTS", 1)
        patch.setattr(assets.urllib.request, "urlopen", interrupt)
        with pytest.raises((OSError, RuntimeError)):
            assets.download()
    partial = assets.TEACHER / (name + ".partial")
    assert partial.read_bytes() == prefix
    assert not (assets.TEACHER / "asset_manifest.json").exists()
    return partial, contents[name]


def test_resume_reuses_partial_and_verified_files(prepared, monkeypatch):
    assets, _, _, _, _, requests, _ = prepared
    partial, content = start_interrupted(prepared, monkeypatch)
    protected = assets.download()
    assert (assets.TEACHER / ".gitattributes").read_bytes() == content
    assert not partial.exists()
    assert any(not isinstance(r, str) and r.get_header("Range") == "bytes=1-" for r in requests)
    assert str(assets.TEACHER / "asset_manifest.json") in protected


@pytest.mark.parametrize("fault", ["ignored", "wrong_start", "wrong_total", "encoded"])
def test_invalid_range_response_preserves_partial(prepared, monkeypatch, fault):
    assets, _, _, _, _, _, _ = prepared
    partial, content = start_interrupted(prepared, monkeypatch)

    def bad_range(request, timeout):
        headers = {"Content-Range": f"bytes 1-{len(content)-1}/{len(content)}"}
        if fault == "wrong_start":
            headers["Content-Range"] = f"bytes 0-{len(content)-1}/{len(content)}"
        elif fault == "wrong_total":
            headers["Content-Range"] = f"bytes 1-{len(content)-1}/{len(content)+1}"
        elif fault == "encoded":
            headers["Content-Encoding"] = "gzip"
        return Response(content if fault == "ignored" else content[1:],
                        200 if fault == "ignored" else 206, headers)

    monkeypatch.setattr(assets.urllib.request, "urlopen", bad_range)
    with pytest.raises(ValueError):
        assets.download()
    assert partial.read_bytes() == b"."
    assert not (assets.TEACHER / "asset_manifest.json").exists()


def test_bounded_retries_then_resumable_failure(prepared, monkeypatch):
    assets, _, _, _, _, _, fetch = prepared
    attempts = []

    def unavailable(request, timeout):
        url = request if isinstance(request, str) else request.full_url
        if "FilePath=" not in url:
            return fetch(request, timeout)
        attempts.append(request)
        raise urllib.error.URLError("temporarily unavailable")

    monkeypatch.setattr(assets.urllib.request, "urlopen", unavailable)
    with pytest.raises((RuntimeError, OSError)):
        assets.download()
    assert len(attempts) == assets.MAX_ATTEMPTS
    assert not (assets.TEACHER / "asset_manifest.json").exists()
    monkeypatch.setattr(assets.urllib.request, "urlopen", fetch)
    assert assets.download() == assets.verify_assets()


def test_bounded_metadata_retries_do_not_create_destination(prepared, monkeypatch):
    assets, _, _, _, _, _, _ = prepared
    attempts = []

    def unavailable(request, timeout):
        attempts.append(request)
        raise urllib.error.URLError("metadata unavailable")

    monkeypatch.setattr(assets.urllib.request, "urlopen", unavailable)
    with pytest.raises((RuntimeError, OSError)):
        assets.download()
    assert len(attempts) == assets.MAX_ATTEMPTS and not assets.TEACHER.exists()


def test_corrupt_download_is_preserved_and_never_finalized(prepared):
    assets, _, contents, _, _, _, _ = prepared
    contents[".gitattributes"] = b"x" * len(contents[".gitattributes"])
    with pytest.raises(ValueError, match="size/hash"):
        assets.download()
    partial = assets.TEACHER / ".gitattributes.partial"
    assert partial.read_bytes() == contents[".gitattributes"]
    assert not (assets.TEACHER / ".gitattributes").exists()
    assert not (assets.TEACHER / "asset_manifest.json").exists()
    with pytest.raises(ValueError):
        assets.verify_assets()


@pytest.mark.parametrize("fault", ["corrupt_final", "corrupt_complete_partial", "oversize",
                                    "symlink", "hardlink", "missing_source", "wrong_marker"])
def test_resume_fails_closed_without_overwriting_bytes(prepared, monkeypatch, fault):
    assets, _, _, _, _, requests, _ = prepared
    partial, content = start_interrupted(prepared, monkeypatch)
    if fault == "corrupt_final":
        partial.rename(assets.TEACHER / ".gitattributes")
    elif fault == "corrupt_complete_partial":
        partial.write_bytes(b"x" * len(content))
    elif fault == "oversize":
        partial.write_bytes(content + b"extra")
    elif fault in ("symlink", "hardlink"):
        external = assets.ROOT / "external"
        external.write_bytes(b".")
        partial.unlink()
        if fault == "symlink":
            partial.symlink_to(external)
        else:
            partial.hardlink_to(external)
    elif fault == "missing_source":
        (assets.TEACHER / "source_metadata.json").unlink()
    else:
        (assets.TEACHER / "SOURCE_REVISION").write_text("master")
    before = {str(p): p.read_bytes() for p in assets.TEACHER.iterdir()}
    n = len(requests)
    with pytest.raises(ValueError):
        assets.download()
    assert before == {str(p): p.read_bytes() for p in assets.TEACHER.iterdir()}
    assert len(requests) == n


def test_complete_partial_is_promoted_without_redownload(prepared, monkeypatch):
    assets, _, _, _, _, requests, _ = prepared
    partial, content = start_interrupted(prepared, monkeypatch)
    partial.write_bytes(content)
    n = len(requests)
    assets.download()
    assert len(requests) - n == len(LIVE_FILES) - 1
    assert not partial.exists()


def test_empty_owned_partial_resumes(prepared, monkeypatch):
    assets, _, _, _, _, _, _ = prepared
    partial, content = start_interrupted(prepared, monkeypatch)
    partial.write_bytes(b"")
    assets.download()
    assert not partial.exists()
    assert (assets.TEACHER / ".gitattributes").read_bytes() == content


def test_concurrent_downloader_fails_without_writing(prepared, monkeypatch):
    assets, _, _, _, _, requests, _ = prepared
    start_interrupted(prepared, monkeypatch)
    before = {str(p): p.read_bytes() for p in assets.TEACHER.iterdir()}
    n = len(requests)
    with assets._download_lock():
        with pytest.raises(ValueError, match="progress"):
            assets.download()
    assert before == {str(p): p.read_bytes() for p in assets.TEACHER.iterdir()}
    assert len(requests) == n


def test_metadata_identity_error_is_not_retried(prepared):
    assets, source, _, _, _, requests, _ = prepared
    source["Data"]["Revision"] = "master"
    with pytest.raises(ValueError):
        assets.download()
    assert len(requests) == 1 and not assets.TEACHER.exists()


def test_finalized_file_is_reused_during_resume(prepared, monkeypatch):
    assets, _, contents, _, _, requests, _ = prepared
    start_interrupted(prepared, monkeypatch)
    ready = assets.TEACHER / "README.md"
    ready.write_bytes(contents["README.md"])
    timestamp = ready.stat().st_mtime_ns
    n = len(requests)
    assets.download()
    assert ready.stat().st_mtime_ns == timestamp
    assert len(requests) - n == len(LIVE_FILES) - 1


def test_download_emits_per_file_transfer_and_verified_progress(prepared, capsys):
    assets, _, contents, _, _, _, _ = prepared
    assets.download()
    progress = capsys.readouterr().out
    for name in contents:
        assert f"Downloading {assets.REPO}@{REVISION} {name}" in progress
        assert f"Verified {name}" in progress


@pytest.mark.parametrize("fallback", [False, True])
def test_nfs_hardlink_prohibition_does_not_block_publication(prepared, monkeypatch, fallback):
    assets, _, _, _, _, _, _ = prepared

    def denied(*args, **kwargs):
        raise PermissionError(errno.EPERM, "NFS forbids hard links")

    monkeypatch.setattr(assets.os, "link", denied)
    if fallback:
        monkeypatch.setattr(assets, "_rename_noreplace", lambda source, destination: False)
    protected = assets.download()
    assert protected == assets.verify_assets()
    assert not list(assets.TEACHER.glob("*.partial"))


@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP,
                                  errno.EXDEV, errno.EPERM, errno.EACCES, errno.EEXIST])
def test_rename_syscall_fallback_is_limited_to_unsupported_errors(assets, tmp_path, monkeypatch, error):
    partial, target = tmp_path / "asset.partial", tmp_path / "asset"
    partial.write_bytes(b"new bytes")

    class Rename:
        def __call__(self, source_fd, source, dest_fd, destination, flags):
            assert source_fd == dest_fd == -100 and flags == 1
            assets.ctypes.set_errno(error)
            return -1

    class Libc:
        renameat2 = Rename()

    monkeypatch.setattr(assets.sys, "platform", "linux")
    monkeypatch.setattr(assets.ctypes, "CDLL", lambda *args, **kwargs: Libc())
    if error in (errno.EACCES, errno.EEXIST):
        with pytest.raises(OSError) as raised:
            assets._publish(partial, target)
        assert raised.value.errno == error
        assert partial.read_bytes() == b"new bytes" and not target.exists()
    else:
        assets._publish(partial, target)
        assert target.read_bytes() == b"new bytes" and not partial.exists()


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("existing", [False, True, "symlink"])
def test_publication_never_overwrites_destination(assets, tmp_path, monkeypatch, fallback, existing):
    partial, target = tmp_path / "asset.partial", tmp_path / "asset"
    partial.write_bytes(b"new bytes")
    if existing == "symlink":
        target.symlink_to(tmp_path / "absent")
    elif existing:
        target.write_bytes(b"original")
    if fallback:
        assert hasattr(assets, "_rename_noreplace"), "Missing NFS-safe publication"
        monkeypatch.setattr(assets, "_rename_noreplace", lambda source, destination: False)
    if existing:
        with pytest.raises(FileExistsError):
            assets._publish(partial, target)
        assert partial.read_bytes() == b"new bytes"
        assert target.is_symlink() if existing == "symlink" else target.read_bytes() == b"original"
    else:
        assets._publish(partial, target)
        assert target.read_bytes() == b"new bytes" and not partial.exists()


def test_exclusive_copy_failure_retains_source_and_cannot_verify(prepared, monkeypatch):
    assets, _, _, _, _, _, _ = prepared
    assert hasattr(assets, "_rename_noreplace"), "Missing NFS-safe publication"
    monkeypatch.setattr(assets, "_rename_noreplace", lambda source, destination: False)

    def interrupted(source, destination, length):
        destination.write(source.read(1))
        raise OSError(errno.ENOSPC, "copy interrupted")

    monkeypatch.setattr(assets.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="copy interrupted"):
        assets.download()
    assert (assets.TEACHER / "source_metadata.json.partial").is_file()
    assert not (assets.TEACHER / "asset_manifest.json").exists()
    with pytest.raises(ValueError):
        assets.verify_assets()


def test_curl_uses_pinned_url_resume_and_finite_timeouts(prepared, monkeypatch):
    assets, _, _, _, _, _, _ = prepared
    partial, content = start_interrupted(prepared, monkeypatch)
    monkeypatch.setattr(assets.shutil, "which", lambda name: "/usr/bin/curl")
    commands = []

    def curl(command, **kwargs):
        commands.append(command)
        assert command[:2] == ["/usr/bin/curl", "--disable"]
        assert command[command.index("--continue-at") + 1] == "-"
        assert command[command.index("--proto") + 1] == "=https"
        assert command[command.index("--proto-redir") + 1] == "=https"
        assert int(command[command.index("--max-time") + 1]) > 0
        assert kwargs["timeout"] > 0 and kwargs["check"] is True
        url = command[command.index("--url") + 1]
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        assert query["Revision"] == [REVISION]
        name = query["FilePath"][0]
        output = Path(command[command.index("--output") + 1])
        value = prepared[2][name]
        offset = output.stat().st_size if output.exists() else 0
        with output.open("ab") as stream:
            stream.write(value[offset:])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(assets.subprocess, "run", curl)
    assets.download()
    assert len(commands) == len(LIVE_FILES)
    assert not partial.exists() and (assets.TEACHER / ".gitattributes").read_bytes() == content


def test_curl_retries_are_bounded_and_keep_partial(prepared, monkeypatch):
    assets, _, _, _, _, _, _ = prepared
    partial, _ = start_interrupted(prepared, monkeypatch)
    monkeypatch.setattr(assets.shutil, "which", lambda name: "/usr/bin/curl")
    attempts = []

    def curl(command, **kwargs):
        attempts.append(command)
        raise subprocess.CalledProcessError(28, command)

    monkeypatch.setattr(assets.subprocess, "run", curl)
    with pytest.raises((RuntimeError, OSError)):
        assets.download()
    assert len(attempts) == assets.MAX_ATTEMPTS
    assert partial.read_bytes() == b"."


@pytest.mark.parametrize("name", TOKENIZER_FILES)
def test_tokenizer_alignment_requires_identical_bytes(prepared, name):
    assets, source, contents, _, _, _, _ = prepared
    contents[name] += b" "
    assets.FILE_SHA256[name] = hashlib.sha256(contents[name]).hexdigest()
    assets.FILE_SIZES[name] = len(contents[name])
    next(row for row in source["Data"]["Files"] if row["Path"] == name).update(
        Sha256=assets.FILE_SHA256[name], Size=len(contents[name]))
    with pytest.raises(ValueError, match="tokenizer|alignment"):
        assets.download()
    assert not (assets.TEACHER / "asset_manifest.json").exists()


@pytest.mark.parametrize("name,value", [
    ("config.json", {"architectures": ["Other"], "max_position_embeddings": 32768}),
    ("config.json", {"architectures": ["Qwen3ForCausalLM"], "max_position_embeddings": 16384}),
    ("generation_config.json", {"eos_token_id": 151645}),
    ("generation_config.json", {"eos_token_id": [151643, 151645]}),
    ("model.safetensors.index.json", {"weight_map": {"layer": "../old/weights.safetensors"}}),
])
def test_semantic_checks_precede_manifest_publication(prepared, name, value):
    assets, source, contents, _, _, _, _ = prepared
    contents[name] = json.dumps(value).encode()
    assets.FILE_SHA256[name] = hashlib.sha256(contents[name]).hexdigest()
    assets.FILE_SIZES[name] = len(contents[name])
    next(row for row in source["Data"]["Files"] if row["Path"] == name).update(
        Sha256=assets.FILE_SHA256[name], Size=len(contents[name]))
    with pytest.raises(ValueError):
        assets.download()
    assert not (assets.TEACHER / "asset_manifest.json").exists()


@pytest.mark.parametrize("download", [False, True])
def test_cli_is_explicit_and_returns_verification_summary(assets, monkeypatch, capsys, download):
    calls = []
    for name in ("download", "verify_assets"):
        def operation(name=name):
            calls.append(name)
            return {"/verified/file": "a" * 64}
        monkeypatch.setattr(assets, name, operation)
    assets.main(["--download"] if download else [])
    assert calls == (["download"] if download else ["verify_assets"])
    assert json.loads(capsys.readouterr().out) == {
        "passed": True, "files_verified": 1, "provider": "modelscope"}


def test_import_has_no_filesystem_network_or_gpu_actions(tmp_path):
    assert SCRIPT.is_file()
    code = """
import importlib.util, pathlib, socket, subprocess, sys
def forbidden(*args, **kwargs):
    raise AssertionError('import side effect')
socket.socket.connect = forbidden
socket.create_connection = forbidden
subprocess.Popen = forbidden
pathlib.Path.mkdir = forbidden
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent))
spec = importlib.util.spec_from_file_location('asset_import_check', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.REPO == 'Qwen/Qwen3-8B-Base'
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(SCRIPT)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

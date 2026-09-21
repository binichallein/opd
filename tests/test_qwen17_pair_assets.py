"""CPU-only contracts for the bounded, pinned Qwen8 -> Qwen1.7 assets."""

import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import urllib.parse
import urllib.request

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_qwen17_pair_assets.py"
IDENTITIES = {
    "base_student": ("Qwen/Qwen3-1.7B-Base", "b0786a09cd6ee101cd8c90e30a5727beb8230544", 2048, 28, 151643, 11, 1),
    "base_teacher": ("Qwen/Qwen3-8B-Base", "932bc907a0f908fd665867dec24af47c2f57e719", 4096, 36, 151643, 15, 5),
    "instruct_student": ("Qwen/Qwen3-1.7B", "4855588ea1a12789f2e965e5f52a9e4a24c94b2a", 2048, 28, 151645, 13, 2),
    "instruct_teacher": ("Qwen/Qwen3-8B", "26028140be3ee69b82b1d1450179ab71bb1121b9", 4096, 36, 151645, 16, 5),
}


def test_preparer_exists():
    assert SCRIPT.is_file(), "Pinned Qwen17 pair preparer is not implemented"


@pytest.fixture
def assets(monkeypatch):
    assert SCRIPT.is_file(), "Pinned Qwen17 pair preparer is not implemented"
    monkeypatch.syspath_prepend(str(SCRIPT.parent))

    def forbidden(*args, **kwargs):
        pytest.fail("CPU fixtures must not access network or external processes")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    spec = importlib.util.spec_from_file_location("qwen17_pair_assets_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.legacy.shutil, "which", lambda name: None)
    monkeypatch.setattr(module.legacy.time, "sleep", lambda seconds: None)
    return module


def test_official_pins_and_interface(assets):
    assert set(assets.MODELS) == set(IDENTITIES)
    assert assets.ROOT == assets.legacy.ROOT
    for key, (repo, revision, hidden, layers, eos, count, shards) in IDENTITIES.items():
        model = assets.MODELS[key]
        assert model["repo"] == repo and model["revision"] == revision
        assert model["path"] == assets.ROOT / "models" / repo.split("/")[1]
        assert isinstance(model["path"], Path) and model["path"].is_absolute()
        assert (model["hidden_size"], model["num_hidden_layers"], model["eos_token_id"]) == (hidden, layers, eos)
        assert model["generation_eos_token_id"] == (151643 if eos == 151643 else [151645, 151643])
        assert model["max_position_embeddings"] == (32768 if eos == 151643 else 40960)
        assert len(model["files"]) == count
        assert len([name for name in model["files"] if name.endswith(".safetensors")]) == shards
        for name, record in model["files"].items():
            assert name == record["Path"] and record["Type"] == "blob"
            assert len(record["Revision"]) == 40 and len(record["Sha256"]) == 64
            assert type(record["Size"]) is int and record["Size"] > 0
    base = assets.MODELS["base_student"]["files"]
    assert "model.safetensors.index.json" not in base
    assert base["model.safetensors"]["Sha256"] == "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"
    for name, record in assets.MODELS["base_teacher"]["files"].items():
        assert record["Sha256"] == assets.legacy.FILE_SHA256[name]
        assert record["Size"] == assets.legacy.FILE_SIZES[name]
        assert record["Revision"] == assets.legacy.FILE_REVISIONS[name]


class Response(io.BytesIO):
    def __init__(self, content, offset=0):
        super().__init__(content[offset:])
        self.status = 206 if offset else 200
        self.headers = {"Content-Length": str(len(content) - offset)}
        if offset:
            self.headers["Content-Range"] = f"bytes {offset}-{len(content) - 1}/{len(content)}"

    def getcode(self):
        return self.status


@pytest.fixture
def fixture_factory(assets, tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "MODELS", copy.deepcopy(assets.MODELS))
    for model in assets.MODELS.values():
        model["path"] = tmp_path / "models" / model["repo"].split("/")[1]

    def make(key="base_student"):
        model = assets.MODELS[key]
        contents = {name: (name + "\n").encode() for name in model["files"]}
        eos = model["eos_token_id"]
        eos_token = "<|endoftext|>" if eos == 151643 else "<|im_end|>"
        documents = {
            "config.json": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                            **{name: model[name] for name in (
                                "hidden_size", "num_hidden_layers", "eos_token_id", "max_position_embeddings")}},
            "generation_config.json": {"eos_token_id": model["generation_eos_token_id"]},
            "tokenizer_config.json": {"eos_token": eos_token},
            "tokenizer.json": {"added_tokens": [
                {"id": 151643, "content": "<|endoftext|>", "special": True},
                {"id": 151645, "content": "<|im_end|>", "special": True}]},
        }
        if "model.safetensors.index.json" in contents:
            documents["model.safetensors.index.json"] = {"weight_map": {
                f"layer.{i}": name for i, name in enumerate(contents) if name.endswith(".safetensors")}}
        contents.update({name: json.dumps(doc).encode() for name, doc in documents.items()})
        for name, content in contents.items():
            model["files"][name].update(Size=len(content), Sha256=hashlib.sha256(content).hexdigest())
        source = {"Success": True, "Data": {"LatestCommitter": {"Id": "", "ShortId": model["revision"][:8]},
                  "Files": copy.deepcopy(list(model["files"].values()))}}
        requests = []

        def urlopen(request, **kwargs):
            url = request if isinstance(request, str) else request.full_url
            requests.append(url)
            parts = urllib.parse.urlsplit(url)
            query = urllib.parse.parse_qs(parts.query)
            assert parts.netloc == "modelscope.cn" and parts.scheme == "https"
            assert model["repo"] in parts.path and query["Revision"] == [model["revision"]]
            assert kwargs["timeout"] > 0
            if parts.path.endswith("/files"):
                assert query["Recursive"] == ["true"]
                return Response(json.dumps(source).encode())
            name = query["FilePath"][0]
            offset = int(request.get_header("Range", "bytes=0-")[6:-1])
            return Response(contents[name], offset)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        return SimpleNamespace(assets=assets, key=key, model=model, contents=contents,
                               source=source, requests=requests, directory=model["path"])

    return make


def snapshot(directory):
    return {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("key", ["base_student", "instruct_student", "instruct_teacher"])
def test_download_and_offline_verification_cover_all_files(fixture_factory, key):
    f = fixture_factory(key)
    protected = f.assets.ensure_asset(key, download=True)
    assert len(f.requests) == 1 + len(f.contents)
    assert set(protected) == {str(path) for path in f.directory.iterdir()}
    for path, digest in protected.items():
        assert Path(path).is_absolute() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    for name in ("source_metadata.json", "asset_manifest.json", "preparation_manifest.json", "SOURCE_REVISION"):
        assert str(f.directory / name) in protected
    before = snapshot(f.directory)
    assert f.assets.ensure_asset(key) == protected
    assert f.assets.ensure_asset(key, download=True) == protected
    assert len(f.requests) == 1 + len(f.contents) and snapshot(f.directory) == before


@pytest.mark.parametrize("download", [False, True])
def test_unknown_directory_is_never_adopted(fixture_factory, download):
    f = fixture_factory()
    f.directory.mkdir(parents=True)
    (f.directory / "config.json").write_bytes(f.contents["config.json"])
    before = snapshot(f.directory)
    with pytest.raises(ValueError):
        f.assets.ensure_asset(f.key, download=download)
    assert not f.requests and snapshot(f.directory) == before


def test_default_offline_does_not_create_or_fetch(fixture_factory):
    f = fixture_factory()
    with pytest.raises(ValueError):
        f.assets.ensure_asset(f.key)
    assert not f.requests and not f.directory.exists()


@pytest.mark.parametrize("fault", ["revision", "short_id", "full_id", "hash", "file_revision", "size",
                                  "boolean_size", "missing", "extra", "duplicate", "traversal", "tree", "failure"])
def test_bad_source_is_rejected_before_directory_creation(fixture_factory, fault):
    f = fixture_factory()
    data, rows = f.source["Data"], f.source["Data"]["Files"]
    if fault == "revision":
        data["Revision"] = "master"
    elif fault == "short_id":
        data["LatestCommitter"]["ShortId"] = "00000000"
    elif fault == "full_id":
        data["LatestCommitter"]["Id"] = "0" * 40
    elif fault in ("hash", "file_revision", "size", "boolean_size"):
        field, value = {"hash": ("Sha256", "0" * 64), "file_revision": ("Revision", "0" * 40),
                        "size": ("Size", 999), "boolean_size": ("Size", True)}[fault]
        rows[0][field] = value
    elif fault == "missing":
        rows.pop()
    elif fault == "extra":
        rows.append({**rows[0], "Path": "extra.py"})
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "traversal":
        rows[0]["Path"] = "../config.json"
    elif fault == "tree":
        rows[0]["Type"] = "tree"
    else:
        f.source["Success"] = False
    with pytest.raises(ValueError):
        f.assets.ensure_asset(f.key, download=True)
    assert not f.directory.exists() and len(f.requests) == 1


@pytest.mark.parametrize("name", ["model.safetensors", "config.json", "SOURCE_REVISION",
                                  "source_metadata.json", "preparation_manifest.json", "asset_manifest.json"])
@pytest.mark.parametrize("download", [False, True])
def test_corruption_never_repairs_or_overwrites(fixture_factory, name, download):
    f = fixture_factory()
    f.assets.ensure_asset(f.key, download=True)
    path = f.directory / name
    value = path.read_bytes()
    path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
    before = snapshot(f.directory)
    f.requests.clear()
    with pytest.raises(ValueError):
        f.assets.ensure_asset(f.key, download=download)
    assert not f.requests and snapshot(f.directory) == before


@pytest.mark.parametrize("fault", ["file_symlink", "hardlink", "extra", "directory_symlink", "parent_symlink", "wrong_path"])
def test_unsafe_paths_fail_closed(fixture_factory, tmp_path, fault):
    f = fixture_factory()
    if fault in ("file_symlink", "hardlink", "extra"):
        f.assets.ensure_asset(f.key, download=True)
        path = f.directory / "config.json"
        if fault == "extra":
            (f.directory / "extra").mkdir()
        else:
            moved = tmp_path / "original"
            path.rename(moved)
            if fault == "file_symlink":
                path.symlink_to(moved)
            else:
                os.link(moved, path)
    elif fault in ("directory_symlink", "parent_symlink"):
        outside = tmp_path / "outside"
        outside.mkdir()
        target = f.directory if fault == "directory_symlink" else f.directory.parent
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(outside, target_is_directory=True)
    else:
        f.model["path"] = tmp_path / "old_assets"
    f.requests.clear()
    with pytest.raises(ValueError):
        f.assets.ensure_asset(f.key, download=True)
    assert not f.requests


def interrupt(f, monkeypatch):
    def transfer(url, partial, size):
        if not partial.exists():
            partial.write_bytes(f.contents[partial.name.removesuffix(".partial")][:1])
        raise OSError("temporary transfer failure")

    monkeypatch.setattr(f.assets.legacy, "_urlopen_transfer", transfer)
    with pytest.raises(RuntimeError, match="attempts"):
        f.assets.ensure_asset(f.key, download=True)
    return f.directory / ".gitattributes.partial"


def test_partial_resume_uses_pins_and_keeps_metadata(fixture_factory, monkeypatch):
    f = fixture_factory()
    original = f.assets.legacy._urlopen_transfer
    partial = interrupt(f, monkeypatch)
    assert partial.read_bytes() == b"."
    before = snapshot(f.directory)
    monkeypatch.setattr(f.assets.legacy, "_urlopen_transfer", original)
    f.requests.clear()
    f.assets.ensure_asset(f.key, download=True)
    assert not partial.exists() and len(f.requests) == len(f.contents)
    for name, value in before.items():
        if not name.endswith(".partial"):
            assert snapshot(f.directory)[name] == value


@pytest.mark.parametrize("fault", ["oversize", "full_wrong_hash", "missing_owner", "wrong_owner", "wrong_revision", "finalized_corruption"])
def test_unsafe_resume_is_read_only(fixture_factory, monkeypatch, fault):
    f = fixture_factory()
    partial = interrupt(f, monkeypatch)
    if fault == "oversize":
        partial.write_bytes(b"x" * (len(f.contents[".gitattributes"]) + 1))
    elif fault == "full_wrong_hash":
        partial.write_bytes(b"x" * len(f.contents[".gitattributes"]))
    elif fault == "missing_owner":
        (f.directory / "preparation_manifest.json").unlink()
    elif fault == "wrong_owner":
        path = f.directory / "preparation_manifest.json"
        doc = json.loads(path.read_bytes())
        doc["owner"] = "some_other_helper"
        path.write_text(json.dumps(doc))
    elif fault == "wrong_revision":
        (f.directory / "SOURCE_REVISION").write_text("0" * 40 + "\n")
    else:
        partial.rename(f.directory / ".gitattributes")
    before = snapshot(f.directory)
    f.requests.clear()
    with pytest.raises(ValueError):
        f.assets.ensure_asset(f.key, download=True)
    assert not f.requests and snapshot(f.directory) == before


@pytest.mark.parametrize("fault", ["hidden_size", "num_hidden_layers", "eos_token_id", "max_position_embeddings",
                                  "generation_eos", "tokenizer_eos", "tokenizer_mapping", "index"])
def test_semantic_checks_precede_final_manifest(fixture_factory, fault):
    f = fixture_factory("instruct_student")
    name = "config.json"
    if fault == "generation_eos":
        name, value = "generation_config.json", {"eos_token_id": 151645}
    elif fault == "tokenizer_eos":
        name, value = "tokenizer_config.json", {"eos_token": "<|endoftext|>"}
    elif fault == "tokenizer_mapping":
        name, value = "tokenizer.json", {"added_tokens": []}
    elif fault == "index":
        name, value = "model.safetensors.index.json", {"weight_map": {"layer": "../old.safetensors"}}
    else:
        value = json.loads(f.contents[name])
        value[fault] = -1
    f.contents[name] = json.dumps(value).encode()
    record = f.model["files"][name]
    record.update(Size=len(f.contents[name]), Sha256=hashlib.sha256(f.contents[name]).hexdigest())
    next(row for row in f.source["Data"]["Files"] if row["Path"] == name).update(record)
    with pytest.raises(ValueError):
        f.assets.ensure_asset(f.key, download=True)
    assert not (f.directory / "asset_manifest.json").exists()


def test_lock_refuses_competing_preparer(fixture_factory, monkeypatch):
    f = fixture_factory()
    interrupt(f, monkeypatch)
    before = snapshot(f.directory)
    f.requests.clear()
    with (f.directory / ".download.lock").open("r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="progress"):
            f.assets.ensure_asset(f.key, download=True)
    assert not f.requests and snapshot(f.directory) == before


@pytest.mark.parametrize("download", [False, True])
def test_existing_base_teacher_uses_only_read_only_legacy_verifier(fixture_factory, monkeypatch, download):
    f = fixture_factory("base_teacher")
    f.directory.mkdir(parents=True)
    for name, content in f.contents.items():
        (f.directory / name).write_bytes(content)
    for name in ("source_metadata.json", "asset_manifest.json"):
        (f.directory / name).write_text("{}\n")
    protected = {name: hashlib.sha256(value[0]).hexdigest() for name, value in snapshot(f.directory).items()}
    calls = []

    def verify():
        calls.append("verify")
        return dict(protected)

    def forbidden(*args, **kwargs):
        pytest.fail("Existing 8B-Base must never be downloaded or adopted")

    monkeypatch.setattr(f.assets.legacy, "TEACHER", f.directory)
    monkeypatch.setattr(f.assets.legacy, "verify_assets", verify)
    monkeypatch.setattr(f.assets.legacy, "download", forbidden)
    before = snapshot(f.directory)
    legacy_globals = dict(vars(f.assets.legacy))
    assert f.assets.ensure_asset(f.key, download=download) == protected
    assert dict(vars(f.assets.legacy)) == legacy_globals
    assert calls == ["verify"] and not f.requests and snapshot(f.directory) == before


def test_curl_resume_is_bounded_and_pinned(fixture_factory, monkeypatch):
    f = fixture_factory()
    partial = interrupt(f, monkeypatch)
    monkeypatch.setattr(f.assets.legacy.shutil, "which", lambda name: "/usr/bin/curl")
    commands = []

    def curl(command, **kwargs):
        commands.append(command)
        assert command[:2] == ["/usr/bin/curl", "--disable"]
        for flag, value in (("--continue-at", "-"), ("--retry", "0"),
                            ("--proto", "=https"), ("--proto-redir", "=https")):
            assert command[command.index(flag) + 1] == value
        assert 0 < int(command[command.index("--max-time") + 1]) <= 1800
        assert kwargs["check"] is True and 0 < kwargs["timeout"] <= 1830
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(command[command.index("--url") + 1]).query)
        assert query["Revision"] == [f.model["revision"]]
        content = f.contents[query["FilePath"][0]]
        assert command[command.index("--max-filesize") + 1] == str(len(content))
        output = Path(command[command.index("--output") + 1])
        offset = output.stat().st_size if output.exists() else 0
        with output.open("ab") as stream:
            stream.write(content[offset:])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", curl)
    f.assets.ensure_asset(f.key, download=True)
    assert len(commands) == len(f.contents) and not partial.exists()


def test_curl_failure_retries_only_three_times(fixture_factory, monkeypatch):
    f = fixture_factory()
    partial = interrupt(f, monkeypatch)
    monkeypatch.setattr(f.assets.legacy.shutil, "which", lambda name: "/usr/bin/curl")
    calls = []

    def curl(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(28, command)

    monkeypatch.setattr(subprocess, "run", curl)
    with pytest.raises(RuntimeError, match="3 attempts"):
        f.assets.ensure_asset(f.key, download=True)
    assert len(calls) == 3 and partial.read_bytes() == b"."


def test_asset_manifest_rejects_boolean_size(fixture_factory):
    f = fixture_factory()
    f.assets.ensure_asset(f.key, download=True)
    path = f.directory / "asset_manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["size_bytes"][str(f.directory / ".download.lock")] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        f.assets.ensure_asset(f.key)


def test_preparation_manifest_rejects_float_size(fixture_factory, monkeypatch):
    f = fixture_factory()
    interrupt(f, monkeypatch)
    path = f.directory / "preparation_manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["files"]["config.json"]["Size"] = float(manifest["files"]["config.json"]["Size"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        f.assets.ensure_asset(f.key, download=True)


def test_resume_refuses_conflicting_manifest_partial_before_transfer(fixture_factory, monkeypatch):
    f = fixture_factory()
    interrupt(f, monkeypatch)
    (f.directory / "asset_manifest.json.partial").write_text("{}\n")
    before = snapshot(f.directory)
    with pytest.raises(ValueError, match="manifest"):
        f.assets.ensure_asset(f.key, download=True)
    assert snapshot(f.directory) == before


def test_complete_weight_partial_is_published_without_request(fixture_factory, monkeypatch):
    f = fixture_factory()
    original = f.assets.legacy._urlopen_transfer
    partial = interrupt(f, monkeypatch)
    partial.write_bytes(f.contents[".gitattributes"])
    monkeypatch.setattr(f.assets.legacy, "_urlopen_transfer", original)
    f.requests.clear()
    f.assets.ensure_asset(f.key, download=True)
    assert len(f.requests) == len(f.contents) - 1 and not partial.exists()


def test_new_download_does_not_mutate_legacy_globals(fixture_factory):
    f = fixture_factory()
    before = dict(vars(f.assets.legacy))
    f.assets.ensure_asset(f.key, download=True)
    assert dict(vars(f.assets.legacy)) == before


def test_nfs_fallback_uses_existing_exclusive_publication(fixture_factory, monkeypatch):
    f = fixture_factory()
    monkeypatch.setattr(f.assets.legacy, "_rename_noreplace", lambda *args: False)
    f.assets.ensure_asset(f.key, download=True)
    assert not list(f.directory.glob("*.partial"))
    assert all(path.stat().st_nlink == 1 for path in f.directory.iterdir())


def test_publication_race_never_overwrites(fixture_factory, monkeypatch):
    f = fixture_factory()
    original = f.assets.legacy._publish

    def publish(partial, target):
        if target.name == ".gitattributes":
            target.write_bytes(b"race winner")
        original(partial, target)

    monkeypatch.setattr(f.assets.legacy, "_publish", publish)
    with pytest.raises(FileExistsError):
        f.assets.ensure_asset(f.key, download=True)
    assert (f.directory / ".gitattributes").read_bytes() == b"race winner"
    assert (f.directory / ".gitattributes.partial").read_bytes() == f.contents[".gitattributes"]
    assert not (f.directory / "asset_manifest.json").exists()


def test_source_retries_are_bounded_and_create_nothing(fixture_factory, monkeypatch):
    f = fixture_factory()
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args)
        raise OSError("source unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="3 attempts"):
        f.assets.ensure_asset(f.key, download=True)
    assert len(calls) == 3 and not f.directory.exists()


def test_unknown_model_key_is_rejected(assets):
    with pytest.raises(ValueError, match="Unknown model"):
        assets.ensure_asset("other", download=True)


@pytest.mark.parametrize("download", [False, True])
def test_cli_is_explicit(assets, monkeypatch, capsys, download):
    calls = []

    def ensure(key, download=False):
        calls.append((key, download))
        return {"/verified/file": "a" * 64}

    monkeypatch.setattr(assets, "ensure_asset", ensure)
    assets.main(["--model", "instruct_student"] + (["--download"] if download else []))
    assert calls == [("instruct_student", download)]
    assert json.loads(capsys.readouterr().out) == {
        "passed": True, "files_verified": 1, "provider": "modelscope", "model": "instruct_student"}


def test_import_has_no_network_process_or_gpu_side_effects(tmp_path):
    assert SCRIPT.is_file()
    code = """
import importlib.util, pathlib, socket, subprocess, sys
def forbidden(*args, **kwargs):
    raise AssertionError('import side effect')
socket.socket.connect = forbidden
socket.create_connection = forbidden
subprocess.Popen = forbidden
pathlib.Path.mkdir = forbidden
spec = importlib.util.spec_from_file_location('asset_import_check', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert set(module.MODELS) == {'base_student', 'base_teacher', 'instruct_student', 'instruct_teacher'}
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(SCRIPT)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

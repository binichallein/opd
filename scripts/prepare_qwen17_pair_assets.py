#!/usr/bin/env python3
"""Prepare pinned official Qwen8 -> Qwen1.7 pair assets; verify offline by default.

Only --download may create/resume one of the three new model directories.
base_teacher always uses the previous helper's read-only verification, even
with --download. Existing unowned directories and corrupt files fail closed.
"""

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_qwen8_teacher_assets as legacy

ROOT = legacy.ROOT
sha256 = legacy.sha256
OWNER = "prepare_qwen17_pair_assets/v1"
LOCAL_FILES = ("SOURCE_REVISION", "source_metadata.json", "preparation_manifest.json", ".download.lock")

# All blobs from the official full-revision /repo/files responses, 2026-09-21.
# Tuples contain size, last-modifying commit (NOT snapshot commit), and SHA256.
_PINNED_FILES = {
    "base_student": {
        ".gitattributes": (2127, "43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3",
            "088e25d6bceaa8943f4f292006210358292349cc36e7578f80c62fce75535f46"),
        "LICENSE": (11343, "b0786a09cd6ee101cd8c90e30a5727beb8230544",
            "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
        "README.md": (2941, "29fbdbba9557ba36358d09a5c8aa57d20f7bbcd9",
            "579b9d7052fa567f439114f01feceed2b06f47d830d06c8ac829f389fd2e39a4"),
        "config.json": (727, "43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3",
            "1bb33a92c3548fbc68b889b490e810440435253598835bd71dff0396060c12db"),
        "configuration.json": (73, "43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3",
            "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9"),
        "generation_config.json": (138, "43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3",
            "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8"),
        "merges.txt": (1671853, "683fb60557a72bc8eccf686d65f705bc7646b30a",
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
        "model.safetensors": (3441185608, "8261121c7c4658f02d16a54e21bbed91cd391ba2",
            "6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841"),
        "tokenizer.json": (7031645, "43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3",
            "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
        "tokenizer_config.json": (9678, "aac0b06fbab45f0f7a52814e7dc0ab88e670e00a",
            "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5"),
        "vocab.json": (2776833, "43ab5b53bd7f37e7e8b2214adcc9178ee11fb0b3",
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    },
    "instruct_student": {
        ".gitattributes": (2127, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "088e25d6bceaa8943f4f292006210358292349cc36e7578f80c62fce75535f46"),
        "LICENSE": (11343, "4855588ea1a12789f2e965e5f52a9e4a24c94b2a",
            "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
        "README.md": (13963, "7b34c202491054b0463f4921e2d01346f00cebdc",
            "257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091"),
        "config.json": (726, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"),
        "configuration.json": (73, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9"),
        "generation_config.json": (239, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
        "merges.txt": (1671853, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
        "model-00001-of-00002.safetensors": (3441185608, "980712f58bdf09497308d37d0e30b535064cde04",
            "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"),
        "model-00002-of-00002.safetensors": (622329984, "980712f58bdf09497308d37d0e30b535064cde04",
            "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"),
        "model.safetensors.index.json": (25605, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d"),
        "tokenizer.json": (11422654, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        "tokenizer_config.json": (9732, "c230923b64efe6d30f13552ace594b9578fab252",
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
        "vocab.json": (2776833, "ba16748358494dc16019aa354b05b6a175b67ff5",
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    },
    "instruct_teacher": {
        ".gitattributes": (2127, "8188480f040c5f1606a1bb3556abf14b31975155",
            "088e25d6bceaa8943f4f292006210358292349cc36e7578f80c62fce75535f46"),
        "LICENSE": (11343, "26028140be3ee69b82b1d1450179ab71bb1121b9",
            "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
        "README.md": (16660, "7a760cb9412aceb31030659c473628312c656aae",
            "0f36caaff9c2516411a7738db384606263ba653c1e63e61d72f511606164d5a6"),
        "config.json": (728, "8188480f040c5f1606a1bb3556abf14b31975155",
            "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"),
        "configuration.json": (73, "8188480f040c5f1606a1bb3556abf14b31975155",
            "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9"),
        "generation_config.json": (239, "8188480f040c5f1606a1bb3556abf14b31975155",
            "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
        "merges.txt": (1671853, "8188480f040c5f1606a1bb3556abf14b31975155",
            "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
        "model-00001-of-00005.safetensors": (3996250744, "7c9709d23bd2136dac1d6ea1fe30f4107d681cd6",
            "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f"),
        "model-00002-of-00005.safetensors": (3993160032, "7c9709d23bd2136dac1d6ea1fe30f4107d681cd6",
            "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282"),
        "model-00003-of-00005.safetensors": (3959604768, "7c9709d23bd2136dac1d6ea1fe30f4107d681cd6",
            "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836"),
        "model-00004-of-00005.safetensors": (3187841392, "7c9709d23bd2136dac1d6ea1fe30f4107d681cd6",
            "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a"),
        "model-00005-of-00005.safetensors": (1244659840, "7c9709d23bd2136dac1d6ea1fe30f4107d681cd6",
            "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff"),
        "model.safetensors.index.json": (32878, "8188480f040c5f1606a1bb3556abf14b31975155",
            "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc"),
        "tokenizer.json": (11422654, "8188480f040c5f1606a1bb3556abf14b31975155",
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        "tokenizer_config.json": (9732, "9a8d25579640d91ddc45ce1b991ef2df939d441f",
            "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
        "vocab.json": (2776833, "8188480f040c5f1606a1bb3556abf14b31975155",
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    },
}

_PINNED_FILES["base_teacher"] = {
    name: (legacy.FILE_SIZES[name], legacy.FILE_REVISIONS[name], digest)
    for name, digest in legacy.FILE_SHA256.items()
}
MODELS = {
    "base_student": {
        "repo": "Qwen/Qwen3-1.7B-Base",
        "revision": "b0786a09cd6ee101cd8c90e30a5727beb8230544",
        "hidden_size": 2048, "num_hidden_layers": 28, "eos_token_id": 151643,
    },
    "base_teacher": {
        "repo": "Qwen/Qwen3-8B-Base",
        "revision": "932bc907a0f908fd665867dec24af47c2f57e719",
        "hidden_size": 4096, "num_hidden_layers": 36, "eos_token_id": 151643,
    },
    "instruct_student": {
        "repo": "Qwen/Qwen3-1.7B",
        "revision": "4855588ea1a12789f2e965e5f52a9e4a24c94b2a",
        "hidden_size": 2048, "num_hidden_layers": 28, "eos_token_id": 151645,
    },
    "instruct_teacher": {
        "repo": "Qwen/Qwen3-8B",
        "revision": "26028140be3ee69b82b1d1450179ab71bb1121b9",
        "hidden_size": 4096, "num_hidden_layers": 36, "eos_token_id": 151645,
    },
}
# Architecture, context, and native EOS were checked against hashed config bytes.
for _key, _model in MODELS.items():
    _base = _model["eos_token_id"] == 151643
    _model.update(
        path=ROOT / "models" / _model["repo"].split("/")[1],
        max_position_embeddings=32768 if _base else 40960,
        generation_eos_token_id=151643 if _base else [151645, 151643],
        files={
            name: {"Path": name, "Type": "blob", "Size": size, "Revision": revision, "Sha256": digest}
            for name, (size, revision, digest) in sorted(_PINNED_FILES[_key].items())
        },
    )


def _source_url(model):
    query = urllib.parse.urlencode({"Revision": model["revision"], "Recursive": "true"})
    return f"https://modelscope.cn/api/v1/models/{model['repo']}/repo/files?{query}"


def select_files(key, source):
    """Require exact file coverage, per-file commits, sizes and hashes."""
    model = MODELS[key]
    if not isinstance(source, dict) or source.get("Success") is not True:
        raise ValueError("ModelScope source query failed")
    data = source.get("Data")
    if not isinstance(data, dict) or not isinstance(data.get("Files"), list):
        raise ValueError("Missing ModelScope file records")
    latest = data.get("LatestCommitter")
    revision = model["revision"]
    if (data.get("Revision", revision) != revision or not isinstance(latest, dict)
            or latest.get("ShortId") != revision[:8]
            or latest.get("Id", "") not in ("", revision)):
        raise ValueError("ModelScope snapshot does not match pinned revision")
    records = {}
    for row in data["Files"]:
        if not isinstance(row, dict) or row.get("Type") != "blob":
            raise ValueError("Unexpected ModelScope file record")
        name = row.get("Path")
        if not isinstance(name, str) or name not in model["files"] or name in records:
            raise ValueError("Unexpected or duplicate ModelScope file path")
        expected = model["files"][name]
        if (type(row.get("Size")) is not int
                or any(row.get(field) != expected[field] for field in expected)):
            raise ValueError(f"Wrong pinned file revision/size/hash: {name}")
        records[name] = expected
    if set(records) != set(model["files"]):
        raise ValueError("Missing pinned model files")
    return {name: records[name] for name in sorted(records)}


def _destination(key):
    if key not in MODELS:
        raise ValueError(f"Unknown model key: {key}")
    model = MODELS[key]
    directory = model["path"]
    if not ROOT.is_absolute() or directory != ROOT / "models" / model["repo"].split("/")[1]:
        raise ValueError("Destination must be the pinned absolute ROOT/models/repo-suffix")
    for path in (directory, *directory.parents):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"Unsafe asset directory: {path}")
    if key == "base_teacher" and (
            directory != legacy.TEACHER or model["repo"] != legacy.REPO
            or model["revision"] != legacy.REVISION):
        raise ValueError("Existing Base teacher must use the pinned legacy location/source")
    return model


def _identity(key):
    model = MODELS[key]
    return {
        "owner": OWNER, "model": key, "provider": "modelscope", "developer": "Qwen",
        "repo": model["repo"], "revision": model["revision"],
        "path": str(model["path"]), "source_request_url": _source_url(model),
    }


def _json_bytes(value):
    # Canonical JSON preserves numeric types: Python equality accepts False == 0.
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _preparation_manifest(key):
    model = MODELS[key]
    source = model["path"] / "source_metadata.json"
    return {
        **_identity(key), "files": model["files"],
        "source_sha256": sha256(source), "source_size_bytes": legacy._regular(source),
    }


def _inventory(key, complete=False):
    model = MODELS[key]
    names = set(model["files"]) | set(LOCAL_FILES) | {"asset_manifest.json"}
    allowed = names if complete else names | {
        name + ".partial" for name in (*model["files"], "asset_manifest.json")}
    if not model["path"].is_dir():
        raise ValueError(f"Missing asset directory: {model['path']}")
    actual = set()
    for path in model["path"].iterdir():
        if path.name not in allowed:
            raise ValueError(f"Unexpected asset outside manifest: {path}")
        legacy._regular(path)
        actual.add(path.name)
    if complete and actual != names:
        raise ValueError("Incomplete asset manifest/file coverage")
    return actual


def _resume_records(key):
    model = MODELS[key]
    directory = model["path"]
    actual = _inventory(key)
    if not set(LOCAL_FILES).issubset(actual):
        raise ValueError("Missing own pinned preparation manifest; refusing to adopt directory")
    if (directory / "SOURCE_REVISION").read_bytes() != (model["revision"] + "\n").encode():
        raise ValueError("Wrong pinned revision marker")
    if legacy._regular(directory / ".download.lock") != 0:
        raise ValueError("Invalid download lock file")
    records = select_files(key, legacy._read_json(directory / "source_metadata.json"))
    ownership = legacy._read_json(directory / "preparation_manifest.json")
    if _json_bytes(ownership) != _json_bytes(_preparation_manifest(key)):
        raise ValueError("Wrong own pinned preparation manifest")
    if "asset_manifest.json.partial" in actual:
        partial = directory / "asset_manifest.json.partial"
        if partial.read_bytes() != _json_bytes(_manifest(key)):
            raise ValueError("Conflicting retained asset manifest partial")
    # Validate every finalized file before any new transfer or partial mutation.
    for name, record in records.items():
        if name in actual:
            legacy.verify_file(directory / name, record)
            if name + ".partial" in actual:
                raise ValueError(f"Both finalized and partial asset exist: {name}")
        elif name + ".partial" in actual:
            partial = directory / (name + ".partial")
            size = legacy._regular(partial)
            if size > record["Size"]:
                raise ValueError(f"Oversized partial asset: {partial}")
            if size == record["Size"]:
                legacy.verify_file(partial, record)
    return records


@contextmanager
def _download_lock(directory, create=False):
    path = directory / ".download.lock"
    flags = os.O_RDWR | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        if legacy._regular(path) != 0:
            raise ValueError("Invalid download lock file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another asset preparation is in progress") from error
        yield
    finally:
        os.close(fd)


def _fetch_source(key):
    with urllib.request.urlopen(_source_url(MODELS[key]), timeout=60) as response:
        source = json.load(response, object_pairs_hook=legacy._unique_object)
    select_files(key, source)
    return source


def _download_file(model, name, record):
    target = model["path"] / name
    partial = model["path"] / (name + ".partial")
    if target.exists() or target.is_symlink():
        legacy._regular(target)
        legacy.verify_file(target, record)
        return
    query = urllib.parse.urlencode({"Revision": model["revision"], "FilePath": name})
    url = f"https://modelscope.cn/api/v1/models/{model['repo']}/repo?{query}"
    curl = legacy.shutil.which("curl")
    offset = legacy._regular(partial) if partial.exists() or partial.is_symlink() else 0
    print(f"Downloading {model['repo']}@{model['revision']} {name} "
          f"({record['Size']} bytes; resume offset {offset})", flush=True)

    def transfer():
        if partial.exists() or partial.is_symlink():
            size = legacy._regular(partial)
            if size > record["Size"]:
                raise ValueError(f"Oversized partial: {partial}")
            if size == record["Size"]:
                legacy.verify_file(partial, record)
                return
        if curl:
            subprocess.run([
                curl, "--disable", "--fail", "--location", "--silent", "--show-error",
                "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "30",
                "--max-time", str(legacy.DOWNLOAD_TIMEOUT), "--speed-limit", "1024", "--speed-time", "60",
                "--retry", "0", "--continue-at", "-", "--max-filesize", str(record["Size"]),
                "--output", str(partial), "--url", url,
            ], check=True, timeout=legacy.DOWNLOAD_TIMEOUT + 30, capture_output=True, text=True)
        else:
            legacy._urlopen_transfer(url, partial, record["Size"])
        if legacy._regular(partial) < record["Size"]:
            raise OSError(f"Incomplete ModelScope transfer: {partial}")
        legacy.verify_file(partial, record)

    # These lower-level helpers accept paths/operations, unlike legacy.download().
    legacy._retry(transfer)
    legacy.verify_file(partial, record)
    legacy._publish(partial, target)


def _verify_model(model):
    directory = model["path"]
    weights = {name for name in model["files"] if name.endswith(".safetensors")}
    if "model.safetensors.index.json" in model["files"]:
        legacy._read_json(directory / "model.safetensors.index.json")
        legacy.verify_index(directory, weights)
    elif weights != {"model.safetensors"}:
        raise ValueError("Single-file snapshot must contain exactly model.safetensors")
    config = legacy._read_json(directory / "config.json")
    if config.get("architectures") != ["Qwen3ForCausalLM"] or config.get("model_type") != "qwen3":
        raise ValueError("Unexpected Qwen3 architecture")
    for field in ("hidden_size", "num_hidden_layers", "max_position_embeddings", "eos_token_id"):
        if type(config.get(field)) is not int or config[field] != model[field]:
            raise ValueError(f"Wrong pinned model config: {field}")
    eos = legacy._read_json(directory / "generation_config.json").get("eos_token_id")
    values = eos if isinstance(eos, list) else [eos]
    if any(type(value) is not int for value in values) or eos != model["generation_eos_token_id"]:
        raise ValueError("Wrong native generation EOS IDs")
    token = "<|endoftext|>" if model["eos_token_id"] == 151643 else "<|im_end|>"
    if legacy._read_json(directory / "tokenizer_config.json").get("eos_token") != token:
        raise ValueError("Wrong native tokenizer EOS token")
    additions = legacy._read_json(directory / "tokenizer.json").get("added_tokens")
    if not isinstance(additions, list):
        raise ValueError("Missing native tokenizer EOS mapping")
    for token_id in values:
        expected = "<|endoftext|>" if token_id == 151643 else "<|im_end|>"
        matches = [row for row in additions if isinstance(row, dict) and row.get("id") == token_id]
        if len(matches) != 1 or matches[0].get("content") != expected or matches[0].get("special") is not True:
            raise ValueError(f"Wrong native tokenizer EOS mapping: {token_id}")


def _manifest(key):
    model = MODELS[key]
    directory = model["path"]
    hashes = {str(directory / name): row["Sha256"] for name, row in model["files"].items()}
    sizes = {str(directory / name): row["Size"] for name, row in model["files"].items()}
    for name in LOCAL_FILES:
        path = directory / name
        sizes[str(path)] = legacy._regular(path)
        hashes[str(path)] = sha256(path)
    return {**_identity(key), "sha256": hashes, "size_bytes": sizes}


def _verify_assets(key):
    model = MODELS[key]
    _inventory(key, complete=True)
    _resume_records(key)
    manifest_path = model["path"] / "asset_manifest.json"
    manifest = legacy._read_json(manifest_path)
    if _json_bytes(manifest) != _json_bytes(_manifest(key)):
        raise ValueError("Wrong asset manifest source/hash/size/coverage")
    _verify_model(model)
    return {**manifest["sha256"], str(manifest_path): sha256(manifest_path)}


def _write_json(path, value):
    legacy._write_once(path, _json_bytes(value))


def ensure_asset(key, download=False):
    """Return protected absolute-path -> SHA256 hashes; never adopt old assets.

    Verification is offline and read-only. Download is opt-in, exclusively locked,
    bounded to three attempts per transfer, and resumes only our pinned manifests.
    base_teacher retains the legacy verifier's historical protected files as well.
    """
    model = _destination(key)
    if key == "base_teacher":
        protected = legacy.verify_assets()
        _verify_model(model)
        return protected
    if not download:
        return _verify_assets(key)
    directory = model["path"]
    manifest_path = directory / "asset_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        return _verify_assets(key)
    created = not directory.exists()
    if created:
        source = legacy._retry(lambda: _fetch_source(key))
        _destination(key)
        directory.mkdir(parents=True, exist_ok=False)
    else:
        _resume_records(key)
    with _download_lock(directory, create=created):
        if created:
            _write_json(directory / "source_metadata.json", source)
            legacy._write_once(directory / "SOURCE_REVISION", (model["revision"] + "\n").encode())
            _write_json(directory / "preparation_manifest.json", _preparation_manifest(key))
        if manifest_path.exists() or manifest_path.is_symlink():
            return _verify_assets(key)
        records = _resume_records(key)
        for name, record in records.items():
            _download_file(model, name, record)
        _verify_model(model)
        _write_json(manifest_path, _manifest(key))
        return _verify_assets(key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--download", action="store_true",
                        help="Explicitly download/resume this new snapshot; Base teacher remains read-only")
    args = parser.parse_args(argv)
    protected = ensure_asset(args.model, download=args.download)
    print(json.dumps({"passed": True, "files_verified": len(protected),
                      "provider": "modelscope", "model": args.model}), flush=True)


if __name__ == "__main__":
    main()

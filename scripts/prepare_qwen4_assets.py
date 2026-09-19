#!/usr/bin/env python3
"""Prepare a pinned ModelScope Qwen4 student; preserve the historical GRPO teacher."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_paired_opd_evals import load_sha256_manifest
from prepare_qwen06_assets import ROOT, TEACHER, REFERENCE, tokenizer_alignment

REPO = "Qwen/Qwen3-4B-Base"
REVISION = "bbd6fc8d23e8788d987b7b970cbb7bd31c826e38"
STUDENT = ROOT / "models/Qwen3-4B-Base"
HISTORICAL_MANIFEST = ROOT / (
    "runs/20260712v1_token_opd_replication_seed21_ml2/token_opd/artifact_hashes.sha256")
WEIGHTS = {
    "model-00001-of-00003.safetensors": "4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241",
    "model-00002-of-00003.safetensors": "f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826",
    "model-00003-of-00003.safetensors": "c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965",
}
COMMON_FILES = (
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "vocab.json", "merges.txt", "README.md", "LICENSE", "configuration.json",
    "model.safetensors.index.json",
)
# All downloaded file identities from the pinned ModelScope /repo/files response.
FILE_SHA256 = {
    **WEIGHTS,
    "config.json": "304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1",
    "configuration.json": "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9",
    "generation_config.json": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors.index.json": "d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c",
    "README.md": "9fd20ab531a1dc75ae18fcde658dd69d04173fdb93311091c38a7098e3d4b4a1",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
SOURCE_REQUEST_URL = (f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?"
                      + urllib.parse.urlencode({"Revision": REVISION, "Recursive": "true"}))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_files(source):
    if source.get("Success") is not True:
        raise ValueError("ModelScope source query failed")
    data = source.get("Data", {})
    rows = data.get("Files")
    if not isinstance(rows, list):
        raise ValueError("Missing ModelScope file records")
    revision = data.get("Revision")
    if revision is not None and revision != REVISION:
        raise ValueError("ModelScope response is not at the pinned revision")
    if data.get("LatestCommitter", {}).get("ShortId") != REVISION[:8]:
        raise ValueError("ModelScope snapshot commit does not match the pinned revision")
    records = {}
    for row in rows:
        if row.get("Type") != "blob":
            continue
        name = row.get("Path")
        if not isinstance(name, str) or name in records:
            raise ValueError("Invalid or duplicate ModelScope file record")
        records[name] = row
    names = set(COMMON_FILES) | set(WEIGHTS)
    if set(FILE_SHA256) != names:
        raise ValueError("Incomplete pinned file identities")
    if not names.issubset(records):
        raise ValueError("Missing original model files")
    for name in names:
        row = records[name]
        # Per-file Revision is its last modifying commit, not the requested snapshot.
        last_modified = row.get("Revision")
        if not isinstance(last_modified, str) or not re.fullmatch("[0-9a-f]{40}", last_modified):
            raise ValueError(f"Invalid last-modifying commit: {name}")
        size, digest = row.get("Size"), row.get("Sha256")
        if (type(size) is not int or size <= 0 or not isinstance(digest, str)
                or not re.fullmatch("[0-9a-f]{64}", digest)):
            raise ValueError(f"Invalid source size/hash: {name}")
        if name in WEIGHTS and digest != WEIGHTS[name]:
            raise ValueError(f"Unexpected weight identity: {name}")
        if digest != FILE_SHA256[name]:
            raise ValueError(f"Unexpected pinned file identity: {name}")
    return {name: records[name] for name in sorted(names)}


def verify_file(path, record):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["Size"]
            or sha256(path) != record["Sha256"]):
        raise ValueError(f"ModelScope size/hash mismatch: {path}")


def download():
    directory = STUDENT
    directory.mkdir(parents=True, exist_ok=False)
    base = f"https://modelscope.cn/api/v1/models/{REPO}"
    with urllib.request.urlopen(SOURCE_REQUEST_URL, timeout=60) as response:
        source = json.load(response)
    records = select_files(source)
    (directory / "source_metadata.json").write_text(json.dumps(source, indent=2) + "\n")
    for name, record in records.items():
        query = urllib.parse.urlencode({"Revision": REVISION, "FilePath": name})
        target = directory / name
        partial = target.with_suffix(target.suffix + ".partial")
        print(f"Downloading {REPO}@{REVISION} {name} ({record['Size']} bytes)", flush=True)
        with urllib.request.urlopen(f"{base}/repo?{query}", timeout=300) as response, partial.open("xb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        verify_file(partial, record)
        partial.rename(target)
    # This is a ModelScope commit, not a Hugging Face revision.
    (directory / "SOURCE_REVISION").write_text(REVISION + "\n")
    manifest = {
        "provider": "modelscope", "developer": "Qwen", "repo": REPO, "revision": REVISION,
        "source_request_url": SOURCE_REQUEST_URL,
        "sha256": {str(directory / name): sha256(directory / name)
                   for name in (*records, "SOURCE_REVISION", "source_metadata.json")},
    }
    (directory / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_index(directory, weights):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Missing weight map: {directory}")
    if (any(not isinstance(name, str) or Path(name).name != name for name in mapping.values())
            or set(mapping.values()) != set(weights)):
        raise ValueError(f"Weight index/shard coverage mismatch: {directory}")


def verify_teacher():
    entries = load_sha256_manifest(HISTORICAL_MANIFEST)
    protected = {str(path): digest for digest, path in entries if path.parent == TEACHER}
    names = {Path(path).name for path in protected}
    required = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "model.safetensors.index.json"}
    weights = {name for name in names if name.endswith(".safetensors")}
    if not weights or not required.issubset(names):
        raise ValueError("Historical teacher manifest is incomplete")
    # Match the historical launcher's file selection, including non-weight inputs.
    actual = {str(path) for path in TEACHER.iterdir()
              if path.is_file() and path.suffix in {".safetensors", ".bin", ".json", ".jinja", ".txt"}}
    if not actual.issubset(protected):
        raise ValueError("Teacher has files outside the historical manifest")
    for name, digest in protected.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Changed historical teacher asset: {path}")
    verify_index(TEACHER, weights)
    protected[str(HISTORICAL_MANIFEST)] = sha256(HISTORICAL_MANIFEST)
    return protected


def verify_assets():
    if not all(path.is_absolute() for path in (STUDENT, TEACHER, REFERENCE, HISTORICAL_MANIFEST)):
        raise ValueError("Asset and historical manifest paths must be absolute")
    manifest_path = STUDENT / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("provider") != "modelscope" or manifest.get("developer") != "Qwen"
            or manifest.get("repo") != REPO or manifest.get("revision") != REVISION
            or manifest.get("source_request_url") != SOURCE_REQUEST_URL):
        raise ValueError("Wrong asset source")
    source_path = STUDENT / "source_metadata.json"
    records = select_files(json.loads(source_path.read_text()))
    expected = {str(STUDENT / name) for name in (*records, "SOURCE_REVISION", "source_metadata.json")}
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != expected:
        raise ValueError("Incomplete asset manifest")
    actual = {str(path) for path in STUDENT.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != expected | {str(manifest_path)}:
        raise ValueError("Unexpected student files outside the asset manifest")
    if (STUDENT / "SOURCE_REVISION").read_text().strip() != REVISION:
        raise ValueError("Wrong pinned revision")
    for name, record in records.items():
        verify_file(STUDENT / name, record)
    for name, digest in hashes.items():
        path = Path(name)
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"Changed asset file: {name}")
    verify_index(STUDENT, WEIGHTS)
    config = json.loads((STUDENT / "config.json").read_text())
    if (config.get("architectures") != ["Qwen3ForCausalLM"]
            or config.get("max_position_embeddings") != 32768):
        raise ValueError("Unexpected Qwen3 architecture or 32k context")
    protected = dict(hashes)
    protected[str(manifest_path)] = sha256(manifest_path)
    protected.update(verify_teacher())
    tokenizer_alignment(STUDENT, TEACHER, REFERENCE)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        protected[str(REFERENCE / name)] = sha256(REFERENCE / name)
    return protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the student into a new directory only")
    args = parser.parse_args(argv)
    if args.download:
        download()
    protected = verify_assets()
    print(json.dumps({"passed": True, "files_verified": len(protected), "provider": "modelscope"}), flush=True)


if __name__ == "__main__":
    main()

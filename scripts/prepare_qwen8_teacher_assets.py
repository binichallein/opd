#!/usr/bin/env python3
"""Prepare only the pinned official ModelScope 8B Base teacher; verify offline by default.

download() explicitly downloads/resumes into ROOT/models/Qwen3-8B-Base. Both it
and verify_assets() return absolute-path SHA256 maps including current4B assets.
No existing historical asset helper, model, or runtime is written by this script.
"""

import argparse
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_qwen4_assets as current4BBase

ROOT = current4BBase.ROOT
REPO = "Qwen/Qwen3-8B-Base"
REVISION = "932bc907a0f908fd665867dec24af47c2f57e719"
TEACHER = ROOT / "models/Qwen3-8B-Base"
sha256 = current4BBase.sha256
verify_file = current4BBase.verify_file
verify_index = current4BBase.verify_index

# Full master commit and ALL 15 blobs confirmed against ModelScope's pinned
# /repo/files response on 2026-09-21. This snapshot has .gitattributes, no LICENSE.
FILE_SHA256 = {
    ".gitattributes": "088e25d6bceaa8943f4f292006210358292349cc36e7578f80c62fce75535f46",
    "README.md": "0eb9f991baa4f92c9ef708d1e9c7a01ae5b19d05516d38f8833549aa8b06a07e",
    "config.json": "3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a",
    "configuration.json": "f888421726665e8a84b738eed42a64875aed79de8be7daade851ac8bf4c0cef9",
    "generation_config.json": "8c970692323e3ea0e9b8b0a4dca79388d31226e41f83c9fd6014804280ebf6e8",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model-00001-of-00005.safetensors": "9983f1b9ef2f60e7c3730d9bc11ada914e6ec630639b5b020a89bd158cd0446b",
    "model-00002-of-00005.safetensors": "9aa12339835bf7a093d3d0b0a0d2d77f8538301cfc7f8e7ec7a585858ebf7a1f",
    "model-00003-of-00005.safetensors": "c7dd5a191c8da555def0714550375b3c0117751add408fcdda0c2a79ac0186bc",
    "model-00004-of-00005.safetensors": "ad8b708792105133e03fe19e2d56c89c709f677f01285f8b3ede8947d59fd9fe",
    "model-00005-of-00005.safetensors": "fbf24915d47ea030bb68ab0b9488f4515a907185baa6dc26837c9c3f2326a550",
    "model.safetensors.index.json": "cd39d5e44c5aadfd1954b598883aea1c4a98822a2f7723d214ef2278b7fec2e4",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
FILE_SIZES = {
    ".gitattributes": 2127, "README.md": 2938, "config.json": 729,
    "configuration.json": 73, "generation_config.json": 138, "merges.txt": 1671853,
    "model-00001-of-00005.safetensors": 3996250744,
    "model-00002-of-00005.safetensors": 3993160032,
    "model-00003-of-00005.safetensors": 3959604768,
    "model-00004-of-00005.safetensors": 3187841392,
    "model-00005-of-00005.safetensors": 1244659840,
    "model.safetensors.index.json": 32878, "tokenizer.json": 7031645,
    "tokenizer_config.json": 9678, "vocab.json": 2776833,
}
WEIGHTS = {name: digest for name, digest in FILE_SHA256.items() if name.endswith(".safetensors")}
FILE_REVISIONS = {
    name: ("9ed78453e256ac6407ffba7369d9815ef4697fcc" if name in WEIGHTS
           else "26fb095fa44e582c29507c6ec61cba1e2ed67bf6") for name in FILE_SHA256
}
FILE_REVISIONS.update({
    "README.md": REVISION,
    "merges.txt": "00b4a13ba84ba74b07cc974f4f3c6c591d36ec44",
    "tokenizer_config.json": "dfb1b602fb76fd49e0884d5b3f126145dbe75ac4",
})
SOURCE_REQUEST_URL = (f"https://modelscope.cn/api/v1/models/{REPO}/repo/files?"
                      + urllib.parse.urlencode({"Revision": REVISION, "Recursive": "true"}))
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
LOCAL_FILES = ("SOURCE_REVISION", "source_metadata.json", ".download.lock")
MAX_ATTEMPTS = 3
DOWNLOAD_TIMEOUT = 1800
CHUNK_SIZE = 8 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular(path):
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"Missing asset file: {path}") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Asset must be a regular, unlinked file: {path}")
    return info.st_size


def _read_json(path):
    _regular(path)
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"Invalid JSON asset/manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def select_files(source):
    """Fail closed on any missing, additional, or changed pinned snapshot blob."""
    if not isinstance(source, dict) or source.get("Success") is not True:
        raise ValueError("ModelScope source query failed")
    data = source.get("Data")
    if not isinstance(data, dict) or not isinstance(data.get("Files"), list):
        raise ValueError("Missing ModelScope file records")
    latest = data.get("LatestCommitter")
    if (data.get("Revision", REVISION) != REVISION or not isinstance(latest, dict)
            or latest.get("ShortId") != REVISION[:8]
            or latest.get("Id", "") not in ("", REVISION)):
        raise ValueError("ModelScope snapshot does not match the pinned revision")
    if not set(FILE_SHA256) == set(FILE_SIZES) == set(FILE_REVISIONS):
        raise ValueError("Incomplete pinned identities")
    records = {}
    for row in data["Files"]:
        if not isinstance(row, dict) or row.get("Type") != "blob":
            raise ValueError("Unexpected ModelScope file record")
        name = row.get("Path")
        if not isinstance(name, str) or name not in FILE_SHA256 or name in records:
            raise ValueError("Unexpected or duplicate ModelScope file path")
        # Per-file Revision is the last modifying commit, not the snapshot commit.
        if (row.get("Revision") != FILE_REVISIONS[name]
                or type(row.get("Size")) is not int or row["Size"] != FILE_SIZES[name]
                or row.get("Sha256") != FILE_SHA256[name]):
            raise ValueError(f"Unexpected pinned file revision/size/hash: {name}")
        records[name] = row
    if set(records) != set(FILE_SHA256):
        raise ValueError("Missing original model files")
    return {name: records[name] for name in sorted(records)}


def _destination():
    if not ROOT.is_absolute() or TEACHER != ROOT / "models/Qwen3-8B-Base":
        raise ValueError("Destination must be the new absolute ROOT/models/Qwen3-8B-Base")
    for path in (TEACHER, *TEACHER.parents):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"Unsafe asset directory: {path}")
    if TEACHER.resolve() in {current4BBase.STUDENT.resolve(), current4BBase.TEACHER.resolve(),
                             current4BBase.REFERENCE.resolve()}:
        raise ValueError("Refusing to write an existing historical asset directory")


def _inventory(complete=False):
    names = set(FILE_SHA256) | set(LOCAL_FILES)
    allowed = names | {"asset_manifest.json"}
    if not complete:
        allowed |= {name + ".partial" for name in (*FILE_SHA256, "asset_manifest.json")}
    actual = set()
    for path in TEACHER.iterdir():
        if path.name not in allowed:
            raise ValueError(f"Unexpected asset outside manifest: {path}")
        _regular(path)
        actual.add(path.name)
    if complete and actual != names | {"asset_manifest.json"}:
        raise ValueError("Incomplete asset manifest/file coverage")
    return actual


def _resume_records():
    actual = _inventory()
    if not set(LOCAL_FILES).issubset(actual):
        raise ValueError("Missing pinned resume metadata; refusing to adopt an existing directory")
    if (TEACHER / "SOURCE_REVISION").read_text() != REVISION + "\n":
        raise ValueError("Wrong pinned revision marker")
    if _regular(TEACHER / ".download.lock") != 0:
        raise ValueError("Invalid download lock file")
    records = select_files(_read_json(TEACHER / "source_metadata.json"))
    # Check all finalized files before issuing any new request or mutating a partial.
    for name, record in records.items():
        if name in actual:
            verify_file(TEACHER / name, record)
            if name + ".partial" in actual:
                raise ValueError(f"Both finalized and partial asset exist: {name}")
        elif name + ".partial" in actual:
            partial = TEACHER / (name + ".partial")
            size = _regular(partial)
            if size > record["Size"]:
                raise ValueError(f"Oversized partial asset: {partial}")
            if size == record["Size"]:
                verify_file(partial, record)
    return records


@contextmanager
def _download_lock():
    path = TEACHER / ".download.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        _regular(path)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another 8B asset preparation is in progress") from error
        yield
    finally:
        os.close(fd)


def _rename_noreplace(partial, target):
    """Return False only when the platform/filesystem cannot do no-replace rename."""
    if not sys.platform.startswith("linux"):
        return False
    rename = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if rename is None:
        return False
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    # AT_FDCWD=-100; RENAME_NOREPLACE=1. Ordinary rename() would overwrite a race winner.
    if rename(-100, os.fsencode(partial), -100, os.fsencode(target), 1) == 0:
        return True
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.EXDEV, errno.EPERM}:
        return False
    raise OSError(error, os.strerror(error), str(target))


def _publish(partial, target):
    """Publish without hard links or overwrite; caller holds the download lock."""
    size = _regular(partial)
    if _rename_noreplace(partial, target):
        return
    # NFS may not support rename flags either. Exclusive creation never replaces
    # an existing file/symlink. Failed copies retain the source and fail closed;
    # the final manifest is not accepted until all files pass size/SHA checks.
    digest = sha256(partial)
    with partial.open("rb") as source, target.open("xb") as output:
        shutil.copyfileobj(source, output, length=CHUNK_SIZE)
        output.flush()
        os.fsync(output.fileno())
    verify_file(target, {"Size": size, "Sha256": digest})
    partial.unlink()


def _write_once(path, content):
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        _regular(partial)
        if partial.read_bytes() != content:
            raise ValueError(f"Conflicting retained partial metadata: {partial}")
    else:
        with partial.open("xb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    _publish(partial, path)


def _retry(operation):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation()
        except (OSError, http.client.HTTPException, subprocess.SubprocessError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (408, 429):
                if error.code < 500:
                    raise ValueError(f"Non-retryable ModelScope HTTP status: {error.code}") from error
            if attempt + 1 == MAX_ATTEMPTS:
                raise RuntimeError(f"ModelScope transfer failed after {MAX_ATTEMPTS} attempts") from error
            time.sleep(min(2 ** attempt, 8))


def _fetch_source():
    with urllib.request.urlopen(SOURCE_REQUEST_URL, timeout=60) as response:
        source = json.load(response, object_pairs_hook=_unique_object)
    select_files(source)
    return source


def _urlopen_transfer(url, partial, size):
    offset = _regular(partial) if partial.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        status = response.getcode()
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Unexpected encoded ModelScope response")
        if status == 206:
            expected = f"bytes {offset}-{size - 1}/{size}"
            if response.headers.get("Content-Range") != expected:
                raise ValueError("Invalid ModelScope resume Content-Range")
        elif status != 200 or offset:
            raise ValueError("ModelScope ignored or rejected the resume Range")
        length = response.headers.get("Content-Length")
        if length is not None and length != str(size - offset):
            raise ValueError("Wrong ModelScope response size")
        with partial.open("ab" if partial.exists() else "xb") as output:
            while chunk := response.read(min(CHUNK_SIZE, size - output.tell() + 1)):
                if output.tell() + len(chunk) > size:
                    raise ValueError("Oversized ModelScope response")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
    if _regular(partial) != size:
        raise OSError("Incomplete ModelScope response; retained partial is resumable")


def _download_file(name, record):
    target = TEACHER / name
    partial = TEACHER / (name + ".partial")
    if target.exists():
        verify_file(target, record)
        print(f"Verified {name} ({record['Size']} bytes; reusing existing file)", flush=True)
        return
    query = urllib.parse.urlencode({"Revision": REVISION, "FilePath": name})
    url = f"https://modelscope.cn/api/v1/models/{REPO}/repo?{query}"
    curl = shutil.which("curl")
    offset = _regular(partial) if partial.exists() else 0
    print(f"Downloading {REPO}@{REVISION} {name} "
          f"({record['Size']} bytes; resume offset {offset})", flush=True)

    def transfer():
        if partial.exists():
            size = _regular(partial)
            if size > record["Size"]:
                raise ValueError(f"Oversized partial: {partial}")
            if size == record["Size"]:
                verify_file(partial, record)
                return
        if curl:
            subprocess.run([
                curl, "--disable", "--fail", "--location", "--silent", "--show-error",
                "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "30",
                "--max-time", str(DOWNLOAD_TIMEOUT), "--speed-limit", "1024", "--speed-time", "60",
                "--retry", "0", "--continue-at", "-", "--max-filesize", str(record["Size"]),
                "--output", str(partial), "--url", url,
            ], check=True, timeout=DOWNLOAD_TIMEOUT + 30, capture_output=True, text=True)
        else:
            _urlopen_transfer(url, partial, record["Size"])
        if _regular(partial) < record["Size"]:
            raise OSError(f"Incomplete ModelScope transfer: {partial}")
        verify_file(partial, record)

    _retry(transfer)
    verify_file(partial, record)
    _publish(partial, target)
    print(f"Verified {name} ({record['Size']} bytes; SHA256 {record['Sha256']})", flush=True)


def _verify_model_and_alignment(protected):
    verify_index(TEACHER, WEIGHTS)
    config = _read_json(TEACHER / "config.json")
    if (config.get("architectures") != ["Qwen3ForCausalLM"]
            or config.get("model_type") != "qwen3" or config.get("max_position_embeddings") != 32768):
        raise ValueError("Unexpected Qwen3 architecture or 32k context")
    for directory in (TEACHER, current4BBase.STUDENT):
        for name in ("config.json", "generation_config.json"):
            eos = _read_json(directory / name).get("eos_token_id")
            if type(eos) is not int or eos != 151643:
                raise ValueError(f"Expected Base EOS 151643: {directory / name}")
    for name in TOKENIZER_FILES:
        teacher, student = TEACHER / name, current4BBase.STUDENT / name
        if (str(student) not in protected or _regular(teacher) != _regular(student)
                or sha256(teacher) != protected[str(student)]):
            raise ValueError(f"Teacher/current4BBase tokenizer byte alignment failed: {name}")
    tokenizer = _read_json(TEACHER / "tokenizer.json")
    additions = tokenizer.get("added_tokens")
    if not isinstance(additions, list):
        raise ValueError("Missing Base tokenizer EOS mapping")
    eos = [token for token in additions if isinstance(token, dict) and token.get("id") == 151643]
    if (len(eos) != 1 or eos[0].get("content") != "<|endoftext|>"
            or eos[0].get("special") is not True
            or _read_json(TEACHER / "tokenizer_config.json").get("eos_token") != "<|endoftext|>"):
        raise ValueError("Expected tokenizer EOS <|endoftext|> at 151643")


def _manifest():
    paths = [TEACHER / name for name in (*sorted(FILE_SHA256), *LOCAL_FILES)]
    return {
        "provider": "modelscope", "developer": "Qwen", "repo": REPO, "revision": REVISION,
        "source_request_url": SOURCE_REQUEST_URL,
        "sha256": {str(path): sha256(path) for path in paths},
        "size_bytes": {str(path): _regular(path) for path in paths},
    }


def verify_assets():
    """Offline verification; return all protected 8B and existing current4B hashes."""
    _destination()
    manifest_path = TEACHER / "asset_manifest.json"
    manifest = _read_json(manifest_path)
    _inventory(complete=True)
    for key, expected in {"provider": "modelscope", "developer": "Qwen", "repo": REPO,
                          "revision": REVISION, "source_request_url": SOURCE_REQUEST_URL}.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Wrong asset manifest source: {key}")
    _resume_records()
    expected = {str(TEACHER / name) for name in (*FILE_SHA256, *LOCAL_FILES)}
    hashes, sizes = manifest.get("sha256"), manifest.get("size_bytes")
    if (not isinstance(hashes, dict) or set(hashes) != expected
            or not isinstance(sizes, dict) or set(sizes) != expected):
        raise ValueError("Incomplete asset manifest hash/size coverage")
    for name, digest in hashes.items():
        if (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
                or type(sizes[name]) is not int or sizes[name] < 0):
            raise ValueError(f"Invalid asset manifest size/hash: {name}")
        verify_file(Path(name), {"Size": sizes[name], "Sha256": digest})
    historical = current4BBase.verify_assets()
    _verify_model_and_alignment(historical)
    protected = dict(historical)
    protected.update(hashes)
    protected[str(manifest_path)] = sha256(manifest_path)
    return protected


def download():
    """Explicit, bounded, resumable download; never repair a corrupt finalized file."""
    _destination()
    manifest_path = TEACHER / "asset_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        return verify_assets()
    if TEACHER.exists():
        _resume_records()
    historical = current4BBase.verify_assets()
    if not TEACHER.exists():
        source = _retry(_fetch_source)
        TEACHER.mkdir(parents=True, exist_ok=False)
        with _download_lock():
            _write_once(TEACHER / "source_metadata.json",
                        (json.dumps(source, indent=2) + "\n").encode())
            _write_once(TEACHER / "SOURCE_REVISION", (REVISION + "\n").encode())
    with _download_lock():
        # A completed snapshot may have appeared while the initial lock was released.
        if manifest_path.exists():
            return verify_assets()
        records = _resume_records()
        for name, record in records.items():
            _download_file(name, record)
        _verify_model_and_alignment(historical)
        _write_once(manifest_path, (json.dumps(_manifest(), indent=2) + "\n").encode())
        protected = verify_assets()
        if not historical.items() <= protected.items():
            raise ValueError("Existing protected assets changed during teacher preparation")
        return protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="Download/resume only the new official 8B Base teacher")
    args = parser.parse_args(argv)
    protected = download() if args.download else verify_assets()
    print(json.dumps({"passed": True, "files_verified": len(protected), "provider": "modelscope"}),
          flush=True)


if __name__ == "__main__":
    main()

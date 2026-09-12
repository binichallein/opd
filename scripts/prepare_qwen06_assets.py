#!/usr/bin/env python3
"""Download only the approved public Qwen06 snapshot and verify token alignment."""

import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path("/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd")
REPO = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
WEIGHT_SHA = "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
STUDENT = ROOT / "models/Qwen3-0.6B-Base"
TEACHER = ROOT / "models/Qwen3-4B-Base-GRPO"
REFERENCE = Path("/limx_embap/tos/user/Yaleon/opd_paper_sft_then_opd_qwen3_1p7b_to_4b_20260606/models/Qwen3-1.7B-Base")
FILES = ("LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt",
         "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_source(source):
    if source.get("id") != REPO or source.get("sha") != REVISION:
        raise ValueError("Unexpected official model or revision")
    weights = [f for f in source["siblings"] if f["rfilename"] == "model.safetensors"]
    if len(weights) != 1 or weights[0].get("lfs", {}).get("sha256") != WEIGHT_SHA:
        raise ValueError("Unexpected weight identity")


def verify_file(path, record):
    if path.stat().st_size != record["size"]:
        raise ValueError(f"Wrong size: {path}")
    if "lfs" in record:
        actual, expected = sha256(path), record["lfs"]["sha256"]
    else:
        digest = hashlib.sha1(b"blob " + str(record["size"]).encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual, expected = digest.hexdigest(), record["blobId"]
    if actual != expected:
        raise ValueError(f"Wrong source hash: {path}")


def tokenizer_alignment(student, teacher, reference):
    values = [json.loads((p / "tokenizer.json").read_text()) for p in (student, teacher, reference)]
    for key in ("model", "added_tokens", "normalizer", "pre_tokenizer", "decoder", "post_processor"):
        if any(value.get(key) != values[0].get(key) for value in values[1:]):
            raise ValueError(f"Incompatible tokenizer component: {key}")
    configs = [json.loads((p / "tokenizer_config.json").read_text()) for p in (student, reference)]
    for key in ("chat_template", "bos_token", "eos_token", "pad_token"):
        if configs[0].get(key) != configs[1].get(key):
            raise ValueError(f"Changed student prompt template/special token: {key}")


def main():
    from huggingface_hub import snapshot_download

    if STUDENT.exists():
        raise FileExistsError("Student directory already exists; inspect instead of overwriting/retrying")
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        source = json.load(response)
    validate_source(source)
    records = {item["rfilename"]: item for item in source["siblings"]}
    if not set(FILES).issubset(records):
        raise ValueError("Incomplete source snapshot")
    STUDENT.mkdir()
    (STUDENT / "source_metadata.json").write_text(json.dumps(source, indent=2) + "\n")
    snapshot_download(REPO, revision=REVISION, local_dir=STUDENT, allow_patterns=list(FILES), max_workers=2)
    for name in FILES:
        verify_file(STUDENT / name, records[name])
    tokenizer_alignment(STUDENT, TEACHER, REFERENCE)
    (STUDENT / "HF_REVISION").write_text(REVISION + "\n")
    hashes = {name: sha256(STUDENT / name) for name in (*FILES, "HF_REVISION", "source_metadata.json")}
    (STUDENT / "asset_manifest.json").write_text(json.dumps({
        "repo": REPO, "revision": REVISION, "tokenizer_alignment": True, "sha256": hashes,
    }, indent=2) + "\n")
    print(json.dumps({"student": str(STUDENT), "revision": REVISION, "verified": True}), flush=True)


if __name__ == "__main__":
    main()

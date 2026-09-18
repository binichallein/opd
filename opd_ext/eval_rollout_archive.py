"""Post-generation evaluation metadata and exclusive, durable JSONL archives.

Schema v1 extends the evaluator row, leaving ``prompt`` (source user text) and
``response`` (exact engine text used by the grader) unchanged. ``rendered_prompt``
is the submitted chat template; ``prompt_raw`` and ``response_raw`` decode the
engine's native token IDs with special tokens and tokenization spaces preserved.
``sampling`` snapshots the available SAMPLING_FIELDS from the submitted params,
including engine decoding defaults, without modifying them or requesting scores.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, TextIO

SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "min_tokens",
    "n", "best_of", "presence_penalty", "frequency_penalty", "repetition_penalty",
    "stop", "stop_token_ids", "ignore_eos", "logprobs", "prompt_logprobs",
    "detokenize", "skip_special_tokens", "spaces_between_special_tokens",
    "include_stop_str_in_output", "truncate_prompt_tokens",
)


def eval_rollout_record(
    output: Any,
    tokenizer: Any,
    sampling: Any,
    *,
    rendered_prompt: str,
    expected_prompt_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Snapshot one native engine result; never reconstruct IDs from decoded text."""
    if output.prompt_token_ids is None:
        raise ValueError("Missing engine prompt token IDs")
    prompt_ids = list(output.prompt_token_ids)
    if expected_prompt_ids is not None and prompt_ids != list(expected_prompt_ids):
        raise ValueError("Engine evaluation prompt IDs differ from the submitted prompt")
    engine_prompt = getattr(output, "prompt", None)
    if engine_prompt is not None and engine_prompt != rendered_prompt:
        raise ValueError("Engine evaluation prompt text differs from the submitted prompt")
    if len(output.outputs) != 1:
        raise ValueError("Expected exactly one engine sample per evaluation prompt")
    sample = output.outputs[0]
    if sample.token_ids is None:
        raise ValueError("Missing engine sample token IDs")
    response_ids = list(sample.token_ids)
    decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    return {
        "schema_version": 1,
        "prompt_token_ids": prompt_ids,
        "response_token_ids": response_ids,
        "rendered_prompt": rendered_prompt,
        "prompt_raw": tokenizer.decode(prompt_ids, **decode_kwargs),
        "response_raw": tokenizer.decode(response_ids, **decode_kwargs),
        "num_generated_tokens": len(response_ids),
        "finish_reason": sample.finish_reason,
        "stop_reason": sample.stop_reason,
        "sampling": {key: deepcopy(getattr(sampling, key)) for key in SAMPLING_FIELDS
                     if hasattr(sampling, key)},
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
    }


@contextmanager
def open_rollout_archive(
    directory: str | Path, task_name: str, rollout_id: int,
) -> Iterator[TextIO]:
    """Reserve a rollout before generation; retain partial files on any failure."""
    if not task_name or task_name in (".", "..") or Path(task_name).name != task_name:
        raise ValueError("Archive task must be a simple directory name")
    if not isinstance(rollout_id, int) or rollout_id < 0:
        raise ValueError("Archive rollout ID must be a nonnegative integer")
    root = Path(directory)
    folder = root / task_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rollout_{rollout_id}.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        try:
            yield handle
        finally:
            handle.flush()
            os.fsync(handle.fileno())
            # Persist the new directory entries as well as the JSONL contents.
            for directory_path in (folder, root, root.parent):
                fd = os.open(directory_path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)


def _read_records(path: Path, hashes: dict[str, str]) -> Iterator[dict[str, Any]]:
    if path.is_symlink():
        raise ValueError(f"Archive audit refuses symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object in {path}")
                yield row
    hashes[str(path.resolve())] = digest.hexdigest()


def _record_digest(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_record(
    row: dict[str, Any], task: str, config: dict[str, Any],
) -> tuple[str, int, int]:
    required = {
        "schema_version", "task", "example_id", "rollout_id", "seed", "model_path",
        "enable_thinking", "source", "problem", "prompt", "answer", "response",
        "rendered_prompt", "prompt_raw", "response_raw", "prompt_token_ids",
        "response_token_ids", "num_generated_tokens", "finish_reason", "stop_reason",
        "sampling", "eos_token_id", "tokenizer_name_or_path",
    }
    if not required.issubset(row):
        raise ValueError(f"{task}: missing native archive fields: {sorted(required - row.keys())}")
    if (type(row["schema_version"]) is not int or row["schema_version"] != 1
            or row["task"] != task or row["model_path"] != config["model_path"]
            or type(row["enable_thinking"]) is not bool
            or row["enable_thinking"] != config["enable_thinking"]):
        raise ValueError(f"{task}: archive schema/model/task/thinking identity mismatch")
    rollout_id, seed = row["rollout_id"], row["seed"]
    if (type(rollout_id) is not int or not 0 <= rollout_id < config["n"]
            or type(seed) is not int or seed != config["eval_seed"] + rollout_id
            or type(row["example_id"]) not in (str, int)):
        raise ValueError(f"{task}: invalid example/rollout/seed identity")
    for name in ("prompt_token_ids", "response_token_ids"):
        ids = row[name]
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError(f"{task}: invalid native {name}")
    if (not row["prompt_token_ids"] or type(row["num_generated_tokens"]) is not int
            or row["num_generated_tokens"] != len(row["response_token_ids"])
            or row["num_generated_tokens"] > config["max_tokens"]):
        raise ValueError(f"{task}: invalid native token count")
    for name in ("prompt", "response", "rendered_prompt", "prompt_raw", "response_raw"):
        if not isinstance(row[name], str):
            raise ValueError(f"{task}: invalid native text field {name}")
    if not row["rendered_prompt"] or not row["prompt_raw"]:
        raise ValueError(f"{task}: empty native prompt text")
    if (row["finish_reason"] not in ("stop", "length")
            or type(row["stop_reason"]) not in (str, int, type(None))):
        raise ValueError(f"{task}: invalid engine finish/stop reason")
    if (row["eos_token_id"] is not None
            and (type(row["eos_token_id"]) is not int or row["eos_token_id"] < 0)):
        raise ValueError(f"{task}: invalid native EOS token ID")
    if row["tokenizer_name_or_path"] is not None and not isinstance(
        row["tokenizer_name_or_path"], str,
    ):
        raise ValueError(f"{task}: invalid tokenizer identity")
    sampling = row["sampling"]
    expected = {key: config[key] for key in ("temperature", "top_p", "max_tokens")}
    expected.update(seed=seed, n=1, ignore_eos=False, logprobs=None, prompt_logprobs=None,
                    detokenize=True, skip_special_tokens=True)
    if not isinstance(sampling, dict) or any(
        key not in sampling or sampling[key] != value for key, value in expected.items()
    ):
        raise ValueError(f"{task}: sampling configuration mismatch")
    stops = sampling.get("stop_token_ids")
    if not isinstance(stops, list) or any(type(token) is not int or token < 0 for token in stops):
        raise ValueError(f"{task}: invalid native stop token IDs")
    return str(row["example_id"]), rollout_id, seed


def audit_archive(output_dir: str | Path, eval_config: dict[str, Any]) -> dict[str, Any]:
    """Read-only keyed equality/coverage audit of finalized raw and archived rows.

    Returns passed/schema_version, num_examples/num_rollouts/num_archive_files,
    per_task counts, and sha256 keyed by absolute raw/archive file paths. Reads
    JSONL incrementally and stores only record hashes. Dataset identity/counts
    and grading are audited by the caller; no tokenizer or engine is loaded here.
    """
    output = Path(output_dir)
    root = output / "rollout_archive"
    tasks, n = eval_config["tasks"], eval_config["n"]
    if (eval_config.get("retain_rollouts") is not True or type(n) is not int or n < 1
            or not tasks or len(set(tasks)) != len(tasks)):
        raise ValueError("Invalid retained evaluation tasks/rollout configuration")
    if root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != set(tasks):
        raise ValueError("Archive task coverage mismatch")
    result: dict[str, Any] = {
        "passed": True, "schema_version": 1, "num_examples": 0, "num_rollouts": 0,
        "num_archive_files": 0, "per_task": {}, "sha256": {},
    }
    for task in tasks:
        folder = root / task
        names = {f"rollout_{rollout_id}.jsonl" for rollout_id in range(n)}
        if (folder.is_symlink() or not folder.is_dir()
                or {p.name for p in folder.iterdir()} != names):
            raise ValueError(f"{task}: archive rollout file coverage mismatch")
        raw = output / (
            f"{task}_t{float(eval_config['temperature'])}_p{float(eval_config['top_p'])}"
            f"_n{n}-MNT{eval_config['max_tokens']}.jsonl"
        )
        expected = {}
        examples: dict[str, set[int]] = {}
        for row in _read_records(raw, result["sha256"]):
            key = _validate_record(row, task, eval_config)
            if key in expected:
                raise ValueError(f"{task}: duplicate finalized raw rollout key {key}")
            expected[key] = _record_digest(row)
            examples.setdefault(key[0], set()).add(key[1])
        if not examples or any(ids != set(range(n)) for ids in examples.values()):
            raise ValueError(f"{task}: incomplete finalized raw rollout coverage")
        num_rollouts = len(expected)
        for rollout_id in range(n):
            path = folder / f"rollout_{rollout_id}.jsonl"
            for row in _read_records(path, result["sha256"]):
                key = _validate_record(row, task, eval_config)
                if key[1] != rollout_id or key not in expected:
                    raise ValueError(f"{task}: duplicate/unexpected archive rollout key {key}")
                if expected.pop(key) != _record_digest(row):
                    raise ValueError(f"{task}: archive differs from finalized raw record {key}")
        if expected:
            raise ValueError(f"{task}: missing archived rollouts: {len(expected)}")
        counts = {"num_examples": len(examples), "num_rollouts": num_rollouts,
                  "num_archive_files": n}
        result["per_task"][task] = counts
        for name, count in counts.items():
            result[name] += count
    return result

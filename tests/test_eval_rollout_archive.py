import copy
import hashlib
import importlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval_qwen3_math_vllm.py"
ROWS = [
    {"id": i, "source": "MATH", "problem": f"Question {i}",
     "prompt": f"Question {i}\n\nPut the answer in \\boxed{{}}.", "answer": "42"}
    for i in range(2)
]
ENGINE_TEXT = "  \\boxed{42}\nraw\u2028separator  "
SAMPLING_DEFAULTS = {
    "n": 1, "best_of": None, "top_k": -1, "min_p": 0.,
    "presence_penalty": 0., "frequency_penalty": 0., "repetition_penalty": 1.,
    "stop": [], "ignore_eos": False, "min_tokens": 0,
    "logprobs": None, "prompt_logprobs": None, "detokenize": True,
    "skip_special_tokens": True, "spaces_between_special_tokens": True,
    "include_stop_str_in_output": False, "truncate_prompt_tokens": None,
}


class Tokenizer:
    def __init__(self, family):
        self.family = family
        self.name_or_path = f"native-{family}"
        self.eos_token_id = 128009 if family == "llama" else 151645
        self.decode_calls = []
        self.encode_calls = []
        self.template_calls = []

    def get_vocab(self):
        if self.family == "llama":
            return {"<|begin_of_text|>": 128000, "<|end_of_text|>": 128001,
                    "<|eom_id|>": 128008, "<|eot_id|>": 128009}
        return {}

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((copy.deepcopy(messages), kwargs))
        content = messages[0]["content"]
        if self.family == "llama":
            assert kwargs["date_string"] == "18 Sep 2026"
            return ("<|begin_of_text|>" + content +
                    "<|start_header_id|>assistant<|end_header_id|>\n\n")
        return ("<|im_start|>user\n" + content + "<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        assert kwargs == {"add_special_tokens": False}
        if self.family == "llama":
            return [128000, 42 + int("Question 1" in text)]
        return {"<|im_end|>": [151645], "<|endoftext|>": [151643]}[text]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False,
                          "clean_up_tokenization_spaces": False}
        self.decode_calls.append(list(ids))
        if ids[0] == 7:
            return ENGINE_TEXT + ("<|eot_id|>" if self.family == "llama" else "<|im_end|>")
        return f"native prompt {list(ids)}"


def load_evaluator(monkeypatch, source=None):
    # Isolate GPU dependencies only; execute the real worker, protocol and writer.
    for name, attributes in {
        "torch": {"cuda": SimpleNamespace(empty_cache=lambda: None)},
        "transformers": {"AutoTokenizer": object},
        "vllm": {"LLM": object, "SamplingParams": object},
    }.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location("archive_evaluator_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    if source is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(source, "historical-9b3b8b-evaluator", "exec"), module.__dict__)
    monkeypatch.setattr(module, "destroy_model_parallel", None)
    monkeypatch.setattr(module, "destroy_distributed_environment", None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return module


def engine(module, family="qwen", before_generate=None, change_outputs=None):
    tokenizer = Tokenizer(family)
    calls = {"llm": [], "sampling": [], "generate": []}

    def sampling_params(**kwargs):
        calls["sampling"].append(copy.deepcopy(kwargs))
        return SimpleNamespace(**copy.deepcopy(SAMPLING_DEFAULTS), **kwargs)

    class LLM:
        def __init__(self, **kwargs):
            calls["llm"].append(kwargs)

        def get_tokenizer(self):
            return tokenizer

        def generate(self, inputs, sampling, **kwargs):
            if before_generate:
                before_generate(len(calls["generate"]))
            calls["generate"].append((copy.deepcopy(inputs), vars(sampling), kwargs))
            outputs = []
            for i, value in enumerate(inputs):
                ids = value["prompt_token_ids"] if isinstance(value, dict) else [151644, i, 91]
                sample = SimpleNamespace(token_ids=(7, 8, tokenizer.eos_token_id),
                                         text=ENGINE_TEXT, finish_reason="stop",
                                         stop_reason=tokenizer.eos_token_id)
                outputs.append(SimpleNamespace(prompt_token_ids=tuple(ids),
                                               prompt=value if isinstance(value, str) else None,
                                               finished=True, outputs=[sample]))
            return change_outputs(outputs) if change_outputs else outputs

    module.LLM = LLM
    module.SamplingParams = sampling_params
    return tokenizer, calls


def worker_args(rollouts=(0, 4), task="math500"):
    return ("model", task, ROWS, list(rollouts), "0", 1., .9, 16384, False, 21)


def read_records(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.parametrize("family", ["qwen", "llama"])
def test_retention_is_lossless_without_changing_inputs_sampling_or_rng(
    monkeypatch, tmp_path, family,
):
    module = load_evaluator(monkeypatch)
    old_tokenizer, old_calls = engine(module, family)
    legacy = module.worker_generate(worker_args())
    tokenizer, calls = engine(module, family)
    rng_state = random.getstate()
    records = module.worker_generate(worker_args(), rollout_archive_dir=tmp_path)
    assert random.getstate() == rng_state
    assert calls == old_calls
    assert tokenizer.encode_calls == old_tokenizer.encode_calls
    assert tokenizer.template_calls == old_tokenizer.template_calls
    for index, record in enumerate(records):
        assert {key: record[key] for key in legacy[index]} == legacy[index]
        assert record["response"] == ENGINE_TEXT
        assert record["response_raw"] == tokenizer.decode(
            record["response_token_ids"], skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        assert record["prompt_raw"] == f"native prompt {record['prompt_token_ids']}"
        assert record["response_token_ids"] == [7, 8, tokenizer.eos_token_id]
        assert record["num_generated_tokens"] == 3
        assert record["finish_reason"] == "stop"
        assert record["stop_reason"] == tokenizer.eos_token_id
        assert record["sampling"] == calls["generate"][index // len(ROWS)][1]
        assert record["sampling"]["seed"] == record["seed"]
        assert record["model_path"] == "model"
        assert record["enable_thinking"] is False
        assert record["schema_version"] == 1
        assert record["eos_token_id"] == tokenizer.eos_token_id
        assert record["tokenizer_name_or_path"] == tokenizer.name_or_path
        if family == "qwen":
            assert record["prompt_token_ids"] == [151644, index % 2, 91]
            assert record["rendered_prompt"] == calls["generate"][index // 2][0][index % 2]
        else:
            assert record["prompt_token_ids"] == [128000, 42 + index % 2]
            assert record["rendered_prompt"].count("<|begin_of_text|>") == 1
    for rollout_id in (0, 4):
        assert read_records(tmp_path / "math500" / f"rollout_{rollout_id}.jsonl") == [
            record for record in records if record["rollout_id"] == rollout_id]


def test_qwen_generation_contract_matches_historical_9b3b8b(monkeypatch, tmp_path):
    source = subprocess.check_output(
        ["git", "show", "9b3b8b:scripts/eval_qwen3_math_vllm.py"], cwd=ROOT, text=True)
    historical = load_evaluator(monkeypatch, source)
    old_tokenizer, old_calls = engine(historical)
    old_records = historical.worker_generate(worker_args(range(8)))
    current = load_evaluator(monkeypatch)
    tokenizer, calls = engine(current)
    records = current.worker_generate(worker_args(range(8)), rollout_archive_dir=tmp_path)
    assert calls == old_calls
    assert tokenizer.template_calls == old_tokenizer.template_calls
    assert tokenizer.encode_calls == old_tokenizer.encode_calls
    assert all(isinstance(prompt, str) for prompt in calls["generate"][0][0])
    assert [record["sampling"]["seed"] for record in records[::2]] == list(range(21, 29))
    assert [{key: record[key] for key in old}
            for record, old in zip(records, old_records)] == old_records
    assert current.split_round_robin(list(range(8)), 4) == [[0, 4], [1, 5], [2, 6], [3, 7]]


def test_default_worker_preserves_positional_contract_and_no_decode(monkeypatch, tmp_path):
    module = load_evaluator(monkeypatch)
    tokenizer, _ = engine(module)
    monkeypatch.chdir(tmp_path)
    records = module.worker_generate(worker_args())
    assert tokenizer.decode_calls == []
    assert list(tmp_path.iterdir()) == []
    assert set(records[0]) == {"task", "example_id", "source", "problem", "prompt",
                               "answer", "rollout_id", "seed", "response"}


def test_batch_is_flushed_and_fsynced_before_next_generate_fails(monkeypatch, tmp_path):
    module = load_evaluator(monkeypatch)
    fsynced = []
    real_fsync = os.fsync

    def fsync(fd):
        fsynced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    path = tmp_path / "math500/rollout_0.jsonl"

    def before_generate(index):
        if index:
            assert len(read_records(path)) == len(ROWS)
            assert str(path) in fsynced
            raise RuntimeError("later generation failed")

    engine(module, before_generate=before_generate)
    with pytest.raises(RuntimeError, match="later generation failed"):
        module.worker_generate(worker_args(), rollout_archive_dir=tmp_path)
    assert len(read_records(path)) == len(ROWS)


def test_rollout_file_is_exclusive_and_reserved_before_generation(monkeypatch, tmp_path):
    module = load_evaluator(monkeypatch)
    _, calls = engine(module)
    module.worker_generate(worker_args((0,)), rollout_archive_dir=tmp_path)
    path = tmp_path / "math500/rollout_0.jsonl"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.worker_generate(worker_args((0,)), rollout_archive_dir=tmp_path)
    assert len(calls["generate"]) == 1
    assert path.read_bytes() == before
    module.worker_generate(worker_args((1,)), rollout_archive_dir=tmp_path)
    module.worker_generate(worker_args((0,), "aime24"), rollout_archive_dir=tmp_path)
    assert len(list(tmp_path.glob("*/*.jsonl"))) == 3


@pytest.mark.parametrize("failure", ["missing_output", "extra_output", "missing_ids",
                                     "multiple_samples", "wrong_llama_ids", "wrong_qwen_prompt"])
def test_invalid_engine_coverage_or_prompt_fails_closed(monkeypatch, tmp_path, failure):
    module = load_evaluator(monkeypatch)

    def change(outputs):
        if failure == "missing_output":
            outputs.pop()
        elif failure == "extra_output":
            outputs.append(outputs[0])
        elif failure == "missing_ids":
            outputs[0].prompt_token_ids = None
        elif failure == "multiple_samples":
            outputs[0].outputs.append(outputs[0].outputs[0])
        elif failure == "wrong_llama_ids":
            outputs[0].prompt_token_ids = (128000, 128000, 42)
        else:
            outputs[0].prompt = "different prompt"
        return outputs

    family = "llama" if failure == "wrong_llama_ids" else "qwen"
    engine(module, family, change_outputs=change)
    with pytest.raises(ValueError, match="coverage|prompt|sample"):
        module.worker_generate(worker_args(), rollout_archive_dir=tmp_path)


@pytest.mark.parametrize("finish_reason,stop_reason", [("length", None), ("stop", "END")])
def test_engine_reasons_are_not_inferred_from_length(
    monkeypatch, tmp_path, finish_reason, stop_reason,
):
    module = load_evaluator(monkeypatch)

    def change(outputs):
        for output in outputs:
            output.outputs[0].finish_reason = finish_reason
            output.outputs[0].stop_reason = stop_reason
        return outputs

    engine(module, change_outputs=change)
    records = module.worker_generate(worker_args((0,)), rollout_archive_dir=tmp_path)
    assert all(record["finish_reason"] == finish_reason for record in records)
    assert all(record["stop_reason"] == stop_reason for record in records)


def test_grader_receives_exact_engine_text_not_raw_decode(monkeypatch, tmp_path):
    module = load_evaluator(monkeypatch)
    engine(module)
    module.worker_generate(worker_args((0,)), rollout_archive_dir=tmp_path)
    responses = []
    monkeypatch.setattr(module, "load_grader", lambda _: lambda response, answer:
                        responses.append(response) or True)
    module.grade_outputs([tmp_path / "math500/rollout_0.jsonl"], tmp_path / "summary.json",
                         None, 1, "external", 21, False)
    assert responses == [ENGINE_TEXT, ENGINE_TEXT]


def run_cli(monkeypatch, tmp_path, retain=True, replace=False, n=2,
            tasks=("math500",), family="qwen"):
    module = load_evaluator(monkeypatch)
    engine(module, family)
    eval_dir = tmp_path / "input"
    eval_dir.mkdir(exist_ok=True)
    for task in tasks:
        (eval_dir / f"{task}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    out = tmp_path / "output"
    argv = [str(SCRIPT), "--model-path", "model", "--eval-jsonl-dir", str(eval_dir),
            "--output-dir", str(out), "--tasks", *tasks, "--n", str(n), "--gpus", "0,1"]
    if retain:
        argv.append("--retain-rollouts")
    if replace:
        argv.append("--replace")
    monkeypatch.setattr(sys, "argv", argv)
    submitted = []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, args, **kwargs):
            submitted.append((args, kwargs))
            return SimpleNamespace(result=lambda: function(args, **kwargs))

    monkeypatch.setattr(module.concurrent.futures, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(module.concurrent.futures, "as_completed", lambda futures: futures)
    monkeypatch.setattr(module, "load_grader", lambda _: lambda response, answer: True)
    module.main()
    return out, submitted


def test_cli_opt_in_wires_archive_without_changing_worker_tuples(monkeypatch, tmp_path):
    out, submitted = run_cli(monkeypatch, tmp_path)
    archive = out / "rollout_archive"
    assert len(list(archive.glob("math500/rollout_*.jsonl"))) == 2
    assert all(len(args) == 10 for args, _ in submitted)
    assert all(Path(kwargs["rollout_archive_dir"]) == archive for _, kwargs in submitted)
    config = json.loads((out / "eval_config.json").read_text())
    assert config["retain_rollouts"] is True
    assert config["rollout_archive_dir"] == str(archive)
    assert len(read_records(out / "math500_t1.0_p0.9_n2-MNT16384.jsonl")) == 4


def test_default_cli_does_not_add_metadata_or_worker_arguments(monkeypatch, tmp_path):
    out, submitted = run_cli(monkeypatch, tmp_path, retain=False)
    assert all(not kwargs for _, kwargs in submitted)
    assert not (out / "rollout_archive").exists()
    assert "retain_rollouts" not in json.loads((out / "eval_config.json").read_text())


@pytest.mark.parametrize("replace", [False, True])
def test_cli_never_reuses_existing_archive_even_with_replace(monkeypatch, tmp_path, replace):
    out, _ = run_cli(monkeypatch, tmp_path)
    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        run_cli(monkeypatch, tmp_path, replace=replace)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_cli_cannot_silently_skip_old_outputs_without_archive(monkeypatch, tmp_path):
    run_cli(monkeypatch, tmp_path, retain=False)
    with pytest.raises(FileExistsError):
        run_cli(monkeypatch, tmp_path)


def archive_helper():
    helper = importlib.import_module("opd_ext.eval_rollout_archive")
    assert hasattr(helper, "audit_archive"), "Read-only archive acceptance helper is missing"
    return helper


@pytest.mark.parametrize("family", ["qwen", "llama"])
def test_audit_four_tasks_n8_is_lossless_keyed_and_read_only(monkeypatch, tmp_path, family):
    helper = archive_helper()
    tasks = ("math500", "aime24", "aime25", "amc23")
    out, _ = run_cli(monkeypatch, tmp_path, n=8, tasks=tasks, family=family)
    config = json.loads((out / "eval_config.json").read_text())
    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}
    audit = helper.audit_archive(out, config)
    assert audit["passed"] is True
    assert audit["schema_version"] == 1
    assert audit["num_examples"] == 8
    assert audit["num_rollouts"] == 64
    assert audit["num_archive_files"] == 32
    assert audit["per_task"] == {
        task: {"num_examples": 2, "num_rollouts": 16, "num_archive_files": 8}
        for task in tasks}
    paths = list((out / "rollout_archive").glob("*/*.jsonl"))
    paths += list(out.glob("*_t*.jsonl"))
    assert audit["sha256"] == {
        str(path.resolve()): hashlib.sha256(before[path]).hexdigest() for path in paths}
    assert before == {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}


def rewrite_records(path, records):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                    encoding="utf-8")


@pytest.mark.parametrize("failure", ["missing_file", "extra_file", "extra_task", "duplicate",
                                     "missing_row", "wrong_partition", "symlink", "raw_duplicate",
                                     "raw_missing", "response_raw", "rendered_prompt", "response",
                                     "prompt_token_ids", "response_token_ids", "truncated_json"])
def test_audit_rejects_archive_coverage_and_exact_record_mismatches(monkeypatch, tmp_path, failure):
    helper = archive_helper()
    out, _ = run_cli(monkeypatch, tmp_path)
    config = json.loads((out / "eval_config.json").read_text())
    path = out / "rollout_archive/math500/rollout_0.jsonl"
    rows = read_records(path)
    if failure == "missing_file":
        path.unlink()
    elif failure == "extra_file":
        path.with_name("rollout_2.jsonl").write_bytes(path.read_bytes())
    elif failure == "extra_task":
        (out / "rollout_archive/other").mkdir()
    elif failure == "symlink":
        target = out / "external.jsonl"
        path.rename(target)
        path.symlink_to(target)
    elif failure.startswith("raw_"):
        path = out / "math500_t1.0_p0.9_n2-MNT16384.jsonl"
        rows = read_records(path)
        rewrite_records(path, rows + [rows[0]] if failure == "raw_duplicate" else rows[1:])
    elif failure == "truncated_json":
        path.write_text('{"task":', encoding="utf-8")
    else:
        if failure == "duplicate":
            rows.append(rows[0])
        elif failure == "missing_row":
            rows.pop()
        elif failure == "wrong_partition":
            rows = read_records(path.with_name("rollout_1.jsonl"))
        elif failure.endswith("token_ids"):
            rows[0][failure][0] += 1
        else:
            rows[0][failure] += "changed"
        rewrite_records(path, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        helper.audit_archive(out, config)


@pytest.mark.parametrize("field,value", [
    ("prompt_token_ids", None), ("prompt_token_ids", []), ("prompt_token_ids", [True]),
    ("response_token_ids", [-1]), ("response_token_ids", [1.5]),
    ("num_generated_tokens", 999), ("num_generated_tokens", True),
    ("finish_reason", None), ("finish_reason", "invented"), ("stop_reason", {}),
    ("response_raw", None), ("rendered_prompt", None), ("rendered_prompt", ""),
    ("seed", 123), ("rollout_id", -1), ("schema_version", 999),
    ("model_path", "other-model"), ("enable_thinking", True),
    ("sampling", {"seed": 21}),
])
def test_audit_rejects_invalid_native_fields_even_when_both_copies_match(
    monkeypatch, tmp_path, field, value,
):
    helper = archive_helper()
    out, _ = run_cli(monkeypatch, tmp_path)
    config = json.loads((out / "eval_config.json").read_text())
    for path in (out / "math500_t1.0_p0.9_n2-MNT16384.jsonl",
                 out / "rollout_archive/math500/rollout_0.jsonl"):
        rows = read_records(path)
        for row in rows:
            if row["rollout_id"] == 0:
                row[field] = value
        rewrite_records(path, rows)
    with pytest.raises(ValueError):
        helper.audit_archive(out, config)


def test_full_cap_response_ids_survive_archiving(monkeypatch, tmp_path):
    module = load_evaluator(monkeypatch)

    def change(outputs):
        for output in outputs:
            output.outputs[0].token_ids = [7] * 16383 + [151645]
        return outputs

    engine(module, change_outputs=change)
    records = module.worker_generate(worker_args((0,)), rollout_archive_dir=tmp_path)
    saved = read_records(tmp_path / "math500/rollout_0.jsonl")
    assert saved == records
    assert all(len(row["response_token_ids"]) == 16384 for row in saved)
    assert all(row["finish_reason"] == "stop" for row in saved)


def test_partial_write_is_durable_and_cannot_be_retried(monkeypatch, tmp_path):
    module = load_evaluator(monkeypatch)
    tokenizer, _ = engine(module)
    decode = tokenizer.decode

    def fail_on_second_prompt(ids, **kwargs):
        if list(ids) == [151644, 1, 91]:
            raise RuntimeError("decode failed")
        return decode(ids, **kwargs)

    tokenizer.decode = fail_on_second_prompt
    with pytest.raises(RuntimeError, match="decode failed"):
        module.worker_generate(worker_args((0,)), rollout_archive_dir=tmp_path)
    path = tmp_path / "math500/rollout_0.jsonl"
    assert len(read_records(path)) == 1
    with pytest.raises(FileExistsError):
        module.worker_generate(worker_args((0,)), rollout_archive_dir=tmp_path)

#!/usr/bin/env python3
"""Generate and grade Qwen3 math eval rollouts with vLLM.

Input data is the eval_jsonl directory produced by
prepare_dapo17k_revisiting_math.py. Each JSONL row must contain:
id, source, problem, prompt, answer.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import importlib.util
import json
import multiprocessing
import os
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
except Exception:  # pragma: no cover
    destroy_model_parallel = None
    destroy_distributed_environment = None


DEFAULT_GRADE_UTILS = (
    "/mnt/data/cpfs/Yaleon/opd_train_qwen3_1p7b_base_to_4b_grpo_20260605/"
    "eval_justrl_steps_20260606/scripts/utils.py"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def apply_template(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    if enable_thinking is False:
        from opd_ext.math_protocol import render_nonthinking
        return render_nonthinking(tokenizer, prompt)
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def split_round_robin(items: list[int], n: int) -> list[list[int]]:
    chunks = [[] for _ in range(n)]
    for i, item in enumerate(items):
        chunks[i % n].append(item)
    return chunks


def worker_generate(
    args_tuple: tuple[Any, ...], *, rollout_archive_dir: str | Path | None = None,
    prompt_protocol: str = 'legacy',
) -> list[dict[str, Any]]:
    (
        model_path,
        task_name,
        rows,
        rollout_ids,
        gpu_id,
        temperature,
        top_p,
        max_tokens,
        enable_thinking,
        eval_seed,
    ) = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = None
    results: list[dict[str, Any]] = []
    try:
        print(
            f"[GPU {gpu_id}] task={task_name} rollouts={rollout_ids} "
            f"model={model_path}",
            flush=True,
        )
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        tokenizer = llm.get_tokenizer()
        from opd_ext.math_protocol import evaluation_stop_ids, evaluation_inputs, native_eval_record
        stop_token_ids = evaluation_stop_ids(tokenizer)

        if prompt_protocol == 'qwen3_completion_boxed_v1':
            from opd_ext.math_protocol import completion_math_prompt, completion_input_ids
            if enable_thinking:
                raise ValueError('Completion evaluation cannot enable thinking')
            prompts = [completion_math_prompt(row['problem']) for row in rows]
            generation_inputs = [{'prompt_token_ids': completion_input_ids(tokenizer, row["problem"])} for row in rows]
            stop_token_ids = []  # Same model EOS151643 as completion training; no added ChatML stop.
        elif prompt_protocol == 'legacy':
            prompts = [apply_template(tokenizer, row["prompt"], enable_thinking) for row in rows]
            generation_inputs = evaluation_inputs(tokenizer, prompts)
        else:
            raise ValueError('Unknown evaluation prompt protocol')
        if rollout_archive_dir is not None:
            from opd_ext.eval_rollout_archive import eval_rollout_record, open_rollout_archive

        for rollout_id in rollout_ids:
            sampling = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids or None,
                seed=eval_seed + rollout_id,
            )
            archive_context = (
                open_rollout_archive(rollout_archive_dir, task_name, rollout_id)
                if rollout_archive_dir is not None else nullcontext()
            )
            with archive_context as archive:
                outputs = llm.generate(generation_inputs, sampling, use_tqdm=False)
                if archive is not None and len(outputs) != len(rows):
                    raise ValueError("Incomplete engine evaluation output coverage")
                for row, output, input_value, rendered in zip(
                    rows, outputs, generation_inputs, prompts,
                ):
                    if archive is not None:
                        extra = eval_rollout_record(
                            output, tokenizer, sampling, rendered_prompt=rendered,
                            expected_prompt_ids=(input_value['prompt_token_ids']
                                                 if isinstance(input_value, dict) else None),
                        )
                        extra.update(model_path=model_path, enable_thinking=enable_thinking)
                    else:
                        extra = (
                            native_eval_record(output, input_value['prompt_token_ids'])
                            if isinstance(input_value, dict) else {}
                        )
                    record = {
                        "task": task_name,
                        "example_id": row["id"],
                        "source": row.get("source", task_name),
                        "problem": row["problem"],
                        "prompt": row["prompt"],
                        "answer": row["answer"],
                        "rollout_id": rollout_id,
                        "seed": eval_seed + rollout_id,
                        "response": output.outputs[0].text,
                        **extra,
                    }
                    if prompt_protocol != 'legacy':
                        record['prompt_protocol'] = prompt_protocol
                    if archive is not None:
                        archive.write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                    results.append(record)
    finally:
        if llm is not None:
            del llm
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        if destroy_distributed_environment is not None:
            try:
                destroy_distributed_environment()
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()
    return results


def load_grader(grader_name: str):
    if grader_name == "verl":
        from verl.utils.reward_score.math import compute_score

        return lambda response, answer: bool(compute_score(response, answer))

    if grader_name != "external":
        raise ValueError(f"unsupported grader: {grader_name}")
    utils_path = Path(os.environ.get("EVAL_GRADE_UTILS_PATH", DEFAULT_GRADE_UTILS))
    if not utils_path.is_file():
        raise FileNotFoundError(
            f"external grader requested but EVAL_GRADE_UTILS_PATH is unavailable: {utils_path}"
        )
    spec = importlib.util.spec_from_file_location("qwen3_math_grade_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import external grader from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.grade_answer_verl


def grade_outputs(
    output_files: list[Path],
    summary_path: Path,
    length_tokenizer_path: str | None,
    n_expected: int,
    grader_name: str,
    eval_seed: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    grade_answer = load_grader(grader_name)
    length_tokenizer = None
    if length_tokenizer_path:
        length_tokenizer = AutoTokenizer.from_pretrained(length_tokenizer_path, local_files_only=True)

    by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in output_files:
        for row in load_jsonl(path):
            by_task[row["task"]][str(row["example_id"])].append(row)

    summary: dict[str, Any] = {
        "n_expected": n_expected,
        "grader": grader_name,
        "eval_seed": eval_seed,
        "rollout_seeds": [eval_seed + rollout_id for rollout_id in range(n_expected)],
        "enable_thinking": enable_thinking,
        "tasks": {},
    }
    for task, examples in sorted(by_task.items()):
        per_example_avg: list[float] = []
        per_example_best: list[float] = []
        lengths: list[int] = []
        format_errors = 0
        engine_length_stops = 0
        engine_finish_count = 0
        total_rollouts = 0
        graded_path = summary_path.with_name(f"{task.lower()}_graded.jsonl")
        with graded_path.open("w", encoding="utf-8") as f:
            for example_id, rows in sorted(examples.items()):
                scores: list[bool] = []
                for row in rows:
                    response = str(row["response"])
                    score = bool(grade_answer(response, str(row["answer"])))
                    if "\\boxed" not in response:
                        format_errors += 1
                    if 'num_generated_tokens' in row:
                        lengths.append(row['num_generated_tokens'])
                        engine_finish_count += 1
                        engine_length_stops += int(row['finish_reason'] == 'length')
                    elif length_tokenizer is not None:
                        lengths.append(len(length_tokenizer.encode(response)))
                    else:
                        lengths.append(len(response))
                    total_rollouts += 1
                    scores.append(score)
                    out = dict(row)
                    out["correct"] = score
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                if scores:
                    per_example_avg.append(sum(scores) / len(scores))
                    per_example_best.append(1.0 if any(scores) else 0.0)

        summary["tasks"][task] = {
            "num_examples": len(examples),
            "total_rollouts": total_rollouts,
            "avg_at_n": sum(per_example_avg) / max(1, len(per_example_avg)),
            "pass_at_n": sum(per_example_best) / max(1, len(per_example_best)),
            "solve_none": sum(1 for x in per_example_avg if x == 0),
            "solve_all": sum(1 for x in per_example_avg if x == 1),
            "format_error_rollouts": format_errors,
            "avg_response_length_tokens" if length_tokenizer is not None else "avg_response_length_chars": (
                sum(lengths) / max(1, len(lengths))
            ),
            "graded_jsonl": str(graded_path),
        }
        if engine_finish_count:
            if engine_finish_count != total_rollouts:
                raise ValueError('Incomplete engine finish-reason coverage')
            summary['tasks'][task]['engine_length_stop_rollouts'] = engine_length_stops
            summary['tasks'][task]['engine_truncation_ratio'] = engine_length_stops / total_rollouts

    task_values = list(summary["tasks"].values())
    summary["macro_avg_at_n"] = sum(t["avg_at_n"] for t in task_values) / max(1, len(task_values))
    summary["macro_pass_at_n"] = sum(t["pass_at_n"] for t in task_values) / max(1, len(task_values))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--eval-jsonl-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["math500", "aime24", "aime25", "amc23"])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--eval-seed", type=int, default=21)
    parser.add_argument("--grader", choices=("verl", "external"), default="verl")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument('--prompt-protocol', choices=['legacy', 'qwen3_completion_boxed_v1'], default='legacy')
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--retain-rollouts", action="store_true",
        help="Retain native tokens and raw decodes in a new exclusive rollout_archive directory",
    )
    parser.add_argument("--length-tokenizer-path", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(args.eval_jsonl_dir)
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must include at least one GPU id")

    rollout_archive_dir = None
    if args.retain_rollouts:
        for task in args.tasks:
            existing = out_dir / (
                f"{task}_t{args.temperature}_p{args.top_p}_n{args.n}-MNT{args.max_tokens}.jsonl"
            )
            if existing.exists():
                raise FileExistsError(existing)
        rollout_archive_dir = out_dir / "rollout_archive"
        rollout_archive_dir.mkdir(exist_ok=False)

    metadata = {
        "model_path": args.model_path,
        "eval_jsonl_dir": str(eval_dir),
        "tasks": args.tasks,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "gpus": gpu_ids,
        "eval_seed": args.eval_seed,
        "rollout_seeds": [args.eval_seed + rollout_id for rollout_id in range(args.n)],
        "grader": args.grader,
        "enable_thinking": args.enable_thinking,
        "prompt_protocol": args.prompt_protocol,
    }
    if args.retain_rollouts:
        metadata.update(retain_rollouts=True, rollout_archive_dir=str(rollout_archive_dir))
    (out_dir / "eval_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    output_files: list[Path] = []
    for task in args.tasks:
        task_path = eval_dir / f"{task}.jsonl"
        if not task_path.exists():
            raise FileNotFoundError(task_path)
        rows = load_jsonl(task_path)
        out_path = out_dir / (
            f"{task}_t{args.temperature}_p{args.top_p}_n{args.n}-MNT{args.max_tokens}.jsonl"
        )
        output_files.append(out_path)
        if out_path.exists() and not args.replace:
            print(f"skip existing {out_path}", flush=True)
            continue

        chunks = split_round_robin(list(range(args.n)), len(gpu_ids))
        work = [
            (
                args.model_path,
                task,
                rows,
                chunks[i],
                gpu_ids[i],
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.enable_thinking,
                args.eval_seed,
            )
            for i in range(len(gpu_ids))
            if chunks[i]
        ]
        all_rows: list[dict[str, Any]] = []
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(work), mp_context=ctx) as ex:
            worker_kwargs = (
                {"rollout_archive_dir": str(rollout_archive_dir)} if args.retain_rollouts else {}
            )
            if args.prompt_protocol != 'legacy':
                worker_kwargs['prompt_protocol'] = args.prompt_protocol
            futures = [ex.submit(worker_generate, item, **worker_kwargs) for item in work]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=task):
                all_rows.extend(fut.result())
        all_rows.sort(key=lambda row: (str(row["example_id"]), int(row["seed"])))
        with out_path.open("x" if args.retain_rollouts else "w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(all_rows)} generations to {out_path}", flush=True)

    summary = grade_outputs(
        output_files,
        out_dir / "summary.json",
        args.length_tokenizer_path,
        args.n,
        args.grader,
        args.eval_seed,
        args.enable_thinking,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "regrade_opd_eval_external.py"
TASK_COUNTS = {"math500": 500, "aime24": 30, "aime25": 30, "amc23": 83}


def load_module():
    spec = importlib.util.spec_from_file_location("regrade_opd_eval_external", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_eval_fixture(tmp_path: Path, *, incomplete: bool = False):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "eval_step_50_n8" / "outputs"
    write_json(
        output_dir / "eval_config.json",
        {
            "tasks": list(TASK_COUNTS),
            "n": 8,
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 16384,
            "eval_seed": 21,
            "rollout_seeds": list(range(21, 29)),
            "grader": "verl",
            "enable_thinking": False,
        },
    )
    primary_summary = {
        "n_expected": 8,
        "grader": "verl",
        "eval_seed": 21,
        "rollout_seeds": list(range(21, 29)),
        "enable_thinking": False,
        "tasks": {},
        "macro_avg_at_n": 0.0,
        "macro_pass_at_n": 0.0,
    }
    for task, count in TASK_COUNTS.items():
        rows = []
        for example_index in range(count):
            for rollout_id in range(8):
                rows.append(
                    {
                        "task": task,
                        "example_id": f"{task}-{example_index}",
                        "rollout_id": rollout_id,
                        "seed": 21 + rollout_id,
                        "response": "1" if rollout_id == 0 else "0",
                        "answer": "1",
                    }
                )
        if incomplete and task == "math500":
            rows.pop()
        raw_path = output_dir / f"{task}_t1.0_p0.9_n8-MNT16384.jsonl"
        raw_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        primary_summary["tasks"][task] = {
            "num_examples": count,
            "total_rollouts": len(rows),
            "avg_at_n": 0.0,
            "pass_at_n": 0.0,
            "solve_none": count,
            "solve_all": 0,
            "format_error_rollouts": len(rows),
            "avg_response_length_tokens": 1.0,
        }
    write_json(output_dir / "summary.json", primary_summary)

    grader = tmp_path / "utils.py"
    grader.write_text(
        "def grade_answer_verl(response, answer):\n"
        "    return response.strip() == answer.strip()\n",
        encoding="utf-8",
    )
    return run_dir, grader


def test_regrade_writes_exact_external_view_and_input_manifest(tmp_path):
    module = load_module()
    run_dir, grader = make_eval_fixture(tmp_path)
    grader_sha = hashlib.sha256(grader.read_bytes()).hexdigest()

    result = module.regrade_run(
        run_dir,
        grader,
        steps=(50,),
        expected_grader_sha256=grader_sha,
    )

    summary = result[50]
    assert summary["grader"] == "external"
    assert summary["macro_avg_at_n"] == pytest.approx(0.125)
    assert summary["macro_pass_at_n"] == pytest.approx(1.0)
    assert sum(task["total_rollouts"] for task in summary["tasks"].values()) == 5144
    view = run_dir / "eval_step_50_n8" / "historical_external_grader"
    assert len((view / "math500_graded.jsonl").read_text().splitlines()) == 4000
    manifest = (view / "input_hashes.sha256").read_text()
    assert manifest.count("\n") == 5
    assert grader_sha in manifest
    assert str(run_dir / "grading" / "historical_utils_sha04f7.py") in manifest
    assert str(
        run_dir
        / "eval_step_50_n8"
        / "outputs"
        / "math500_t1.0_p0.9_n8-MNT16384.jsonl"
    ) in manifest
    output_manifest = (view / "output_hashes.sha256").read_text().splitlines()
    assert len(output_manifest) == 5
    for line in output_manifest:
        digest, recorded_path = line.split(maxsplit=1)
        assert hashlib.sha256(Path(recorded_path).read_bytes()).hexdigest() == digest


def test_regrade_rejects_wrong_grader_hash_before_writing(tmp_path):
    module = load_module()
    run_dir, grader = make_eval_fixture(tmp_path)

    with pytest.raises(ValueError, match="grader SHA"):
        module.regrade_run(
            run_dir,
            grader,
            steps=(50,),
            expected_grader_sha256="0" * 64,
        )

    assert not (run_dir / "grading" / "historical_utils_sha04f7.py").exists()
    assert not (
        run_dir / "eval_step_50_n8" / "historical_external_grader"
    ).exists()


def test_regrade_rejects_incomplete_rollout_group(tmp_path):
    module = load_module()
    run_dir, grader = make_eval_fixture(tmp_path, incomplete=True)
    grader_sha = hashlib.sha256(grader.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="exactly 8 rollouts"):
        module.regrade_run(
            run_dir,
            grader,
            steps=(50,),
            expected_grader_sha256=grader_sha,
        )


def test_regrade_refuses_to_overwrite_evidence_without_replace(tmp_path):
    module = load_module()
    run_dir, grader = make_eval_fixture(tmp_path)
    grader_sha = hashlib.sha256(grader.read_bytes()).hexdigest()
    module.regrade_run(
        run_dir,
        grader,
        steps=(50,),
        expected_grader_sha256=grader_sha,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        module.regrade_run(
            run_dir,
            grader,
            steps=(50,),
            expected_grader_sha256=grader_sha,
        )

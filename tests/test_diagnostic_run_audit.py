import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_block10_run.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_opd_run", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_step_list_accepts_sorted_unique_steps():
    audit = load_audit_module()

    assert audit.parse_step_list("200,50,100,50") == [50, 100, 200]


def test_acceptance_json_is_published_with_atomic_replace(tmp_path, monkeypatch):
    audit = load_audit_module()
    output = tmp_path / "acceptance.json"
    replacements = []
    real_replace = audit.os.replace

    def recording_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(audit.os, "replace", recording_replace)
    audit.atomic_write_json(output, {"passed": True})

    assert json.loads(output.read_text()) == {"passed": True}
    assert len(replacements) == 1
    temporary, destination = replacements[0]
    assert destination == output
    assert temporary.parent == output.parent
    assert not temporary.exists()


def test_parse_step_list_rejects_nonpositive_steps():
    audit = load_audit_module()

    try:
        audit.parse_step_list("0,50")
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("expected nonpositive diagnostic step to be rejected")


def test_expected_run_card_can_target_block3():
    audit = load_audit_module()

    expected = audit.expected_run_card("block3_mean")

    assert expected["variant"] == "block3_mean"
    assert expected["rollout_gpu_memory_utilization"] == 0.6
    assert expected["ref_log_prob_micro_batch_size_per_gpu"] == 1
    assert expected["opd_diag_position_stride"] == 1
    assert expected["train_batch_size"] == 4
    assert expected["ppo_mini_batch_size"] == 32
    assert expected["rollout_group_size"] == 8
    assert expected["max_prompt_length"] == 2048
    assert expected["max_response_length"] == 16384
    assert expected["learning_rate"] == 2e-6
    assert expected["actor_ppo_micro_batch_size_per_gpu"] == 1
    assert expected["rollout_log_prob_micro_batch_size_per_gpu"] == 4
    assert expected["rollout_temperature"] == 1.0
    assert expected["rollout_top_p"] == 0.9


def test_checkpoint_audit_uses_requested_world_size(tmp_path):
    audit = load_audit_module()
    actor = tmp_path / "checkpoints" / "global_step_50" / "actor"
    actor.mkdir(parents=True)
    (actor.parent / "data.pt").write_bytes(b"data")
    for rank in range(2):
        for prefix in ("model", "optim", "extra_state"):
            (actor / f"{prefix}_world_size_2_rank_{rank}.pt").write_bytes(b"state")

    assert audit.checkpoint_issues(tmp_path, step=50, world_size=2) == []


def test_checkpoint_audit_rejects_duplicate_or_out_of_range_rank_names(tmp_path):
    audit = load_audit_module()
    actor = tmp_path / "checkpoints" / "global_step_50" / "actor"
    actor.mkdir(parents=True)
    (actor.parent / "data.pt").write_bytes(b"data")
    for prefix in ("model", "optim", "extra_state"):
        (actor / f"{prefix}_world_size_2_rank_0.pt").write_bytes(b"state")
        (actor / f"{prefix}_world_size_2_rank_2.pt").write_bytes(b"state")

    issues = audit.checkpoint_issues(tmp_path, step=50, world_size=2)

    assert any("ranks [0, 2]" in issue for issue in issues)


def test_numerical_audit_promotes_nonfinite_values_and_positive_error_counts_to_issues():
    audit = load_audit_module()
    records = [
        {
            "step": 5,
            "diagnostics/student_entropy": math.inf,
            "diagnostics/raw_token_advantage_nonfinite_count": 1.0,
            "diagnostics/post_update_block_ratio_overflow_count": 2.0,
        }
    ]

    issues = audit.numerical_scalar_issues(records)

    assert any("nonfinite scalar" in issue for issue in issues)
    assert any("raw_token_advantage_nonfinite_count=1.0" in issue for issue in issues)
    assert any("post_update_block_ratio_overflow_count=2.0" in issue for issue in issues)


@pytest.mark.parametrize("separator", ["", "\u0085", "\u2028", "\u2029"])
def test_eval_audit_requires_exact_dataset_size_config_and_rollout_seeds(tmp_path, separator):
    audit = load_audit_module()
    eval_dir = tmp_path / "eval_step_50_n8"
    outputs = eval_dir / "outputs"
    outputs.mkdir(parents=True)
    tasks = {
        task: {"num_examples": count, "total_rollouts": count * 8}
        for task, count in audit.EXPECTED_TASK_EXAMPLES.items()
    }
    (outputs / "summary.json").write_text(
        json.dumps(
            {
                "n_expected": 8,
                "grader": "verl",
                "eval_seed": 21,
                "rollout_seeds": list(range(21, 29)),
                "enable_thinking": False,
                "tasks": tasks,
            }
        )
    )
    (outputs / "eval_config.json").write_text(
        json.dumps(
            {
                "model_path": str(
                    tmp_path / "checkpoints" / "global_step_50" / "actor" / "huggingface"
                ),
                "eval_jsonl_dir": "/data/eval_jsonl",
                "tasks": ["math500", "aime24", "aime25", "amc23"],
                "n": 8,
                "temperature": 1.0,
                "top_p": 0.9,
                "max_tokens": 16384,
                "eval_seed": 21,
                "rollout_seeds": list(range(21, 29)),
                "grader": "verl",
                "enable_thinking": False,
            }
        )
    )
    (eval_dir / "exit_code.txt").write_text("0\n")
    (eval_dir / "eval_data_hashes.sha256").write_text(
        "".join(
            f"{digest}  /data/eval_jsonl/{task}.jsonl\n"
            for task, digest in audit.EXPECTED_EVAL_SHA256.items()
        )
    )
    for task, count in audit.EXPECTED_TASK_EXAMPLES.items():
        rows = []
        for example_id in range(count):
            for seed in range(21, 29):
                rows.append(json.dumps({"example_id": str(example_id), "seed": seed,
                                        "response": f"before{separator}after"}, ensure_ascii=False))
        (outputs / f"{task}_graded.jsonl").write_text("\n".join(rows) + "\n")

    assert audit.eval_issues(
        tmp_path, step=50, expected_eval_data_dir="/data/eval_jsonl"
    ) == []

    config = json.loads((outputs / "eval_config.json").read_text())
    config["top_p"] = 0.8
    (outputs / "eval_config.json").write_text(json.dumps(config))
    assert any(
        "top_p" in issue
        for issue in audit.eval_issues(
            tmp_path, step=50, expected_eval_data_dir="/data/eval_jsonl"
        )
    )


def test_artifact_manifest_checks_expected_train_sha(tmp_path):
    audit = load_audit_module()
    manifest = tmp_path / "artifact_hashes.sha256"
    manifest.write_text("abc123  /data/train.parquet\n")

    assert audit.artifact_hash_issues(manifest, "abc123") == []
    assert "train.parquet SHA-256" in audit.artifact_hash_issues(manifest, "wrong")[0]


def test_model_identity_audit_checks_suffixes_and_revisions():
    audit = load_audit_module()
    card = {
        "student_model": "/models/DeepSeek-R1-Distill-Qwen-1.5B",
        "teacher_model": "/models/JustRL-DeepSeek-1.5B",
        "student_model_revision": "student-revision",
        "teacher_model_revision": "teacher-revision",
    }

    assert audit.model_identity_issues(
        card,
        student_suffix="DeepSeek-R1-Distill-Qwen-1.5B",
        teacher_suffix="JustRL-DeepSeek-1.5B",
        student_revision="student-revision",
        teacher_revision="teacher-revision",
    ) == []

    issues = audit.model_identity_issues(
        card,
        student_revision="wrong-student-revision",
    )
    assert any("student_model_revision" in issue for issue in issues)


def test_position_snapshot_audit_requires_all_twelve_finite_metrics(tmp_path):
    audit = load_audit_module()
    path = tmp_path / "step_00005.npz"
    arrays = {"step": np.asarray(5)}
    for metric in audit.REQUIRED_POSITION_METRICS:
        arrays[f"{metric}__sum"] = np.zeros(16384)
        arrays[f"{metric}__squared_sum"] = np.zeros(16384)
        arrays[f"{metric}__valid_count"] = np.ones(16384, dtype=np.int64)
    np.savez_compressed(path, **arrays)

    assert audit.diagnostic_snapshot_issues(path, expected_step=5) == []

    arrays.pop("teacher_overlap_mass__sum")
    np.savez_compressed(path, **arrays)
    assert any(
        "teacher_overlap_mass__sum" in issue
        for issue in audit.diagnostic_snapshot_issues(path, expected_step=5)
    )


def test_scalar_record_audit_requires_every_monitored_metric():
    audit = load_audit_module()
    record = {"step": 5, **{name: 0.0 for name in audit.REQUIRED_SCALAR_METRICS}}

    assert audit.scalar_record_issues([record]) == []

    del record["actor/grad_norm"]
    assert "actor/grad_norm" in audit.scalar_record_issues([record])[0]

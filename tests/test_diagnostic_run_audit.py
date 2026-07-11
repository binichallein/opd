import importlib.util
import json
import math
from pathlib import Path


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


def test_eval_audit_requires_exact_dataset_size_config_and_rollout_seeds(tmp_path):
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
    for task, count in audit.EXPECTED_TASK_EXAMPLES.items():
        rows = []
        for example_id in range(count):
            for seed in range(21, 29):
                rows.append(json.dumps({"example_id": str(example_id), "seed": seed}))
        (outputs / f"{task}_graded.jsonl").write_text("\n".join(rows) + "\n")

    assert audit.eval_issues(tmp_path, step=50) == []

    config = json.loads((outputs / "eval_config.json").read_text())
    config["top_p"] = 0.8
    (outputs / "eval_config.json").write_text(json.dumps(config))
    assert any("top_p" in issue for issue in audit.eval_issues(tmp_path, step=50))


def test_artifact_manifest_checks_expected_train_sha(tmp_path):
    audit = load_audit_module()
    manifest = tmp_path / "artifact_hashes.sha256"
    manifest.write_text("abc123  /data/train.parquet\n")

    assert audit.artifact_hash_issues(manifest, "abc123") == []
    assert "train.parquet SHA-256" in audit.artifact_hash_issues(manifest, "wrong")[0]

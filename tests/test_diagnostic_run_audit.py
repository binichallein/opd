import importlib.util
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


def test_checkpoint_audit_uses_requested_world_size(tmp_path):
    audit = load_audit_module()
    actor = tmp_path / "checkpoints" / "global_step_50" / "actor"
    actor.mkdir(parents=True)
    (actor.parent / "data.pt").write_bytes(b"data")
    for rank in range(2):
        for prefix in ("model", "optim", "extra_state"):
            (actor / f"{prefix}_world_size_2_rank_{rank}.pt").write_bytes(b"state")

    assert audit.checkpoint_issues(tmp_path, step=50, world_size=2) == []

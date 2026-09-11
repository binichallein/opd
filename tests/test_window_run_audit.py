import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/audit_window_run.py"
ADAPTER = ROOT / "scripts/audit_window_control.py"


def load_audit():
    assert HELPER.is_file(), "window audit helper has not been implemented"
    spec = importlib.util.spec_from_file_location("window_run_audit", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture(params=["random3", "sliding3"])
def window_run(tmp_path, request):
    variant = request.param
    mode = {"random3": "random", "sliding3": "sliding"}[variant]
    card = {
        "variant": variant,
        "seed": 21,
        "opd_window_mode": mode,
        "opd_window_seed": 910021,
        "opd_block_size": 3,
        "opd_block_advantage_mode": "mean",
        "ppo_epochs": 1,
        "window_supervision_sha256": "a" * 64,
    }
    write_json(tmp_path / "run_card.json", card)
    (tmp_path / "script_hashes.sha256").write_text(
        f"{'a' * 64}  /runtime/opd_ext/window_supervision.py\n"
    )
    records = []
    for step in (1, 2):
        offset = step % 3 if mode == "random" else 0
        rng_state = {
            "bit_generator": "PCG64",
            "state": {"state": 12345, "inc": 6789},
            "has_uint32": 0,
            "uinteger": 0,
        }
        state = {
            "mode": mode,
            "block_size": 3,
            "seed": 910021,
            "last_step": step,
            "current_offset": offset,
            "rng_state": rng_state if mode == "random" else None,
        }
        write_json(tmp_path / f"checkpoints/global_step_{step}/window_state.json", state)
        records.append(
            {
                "step": step,
                "prompt_batch_sha256": f"{step:064x}",
                "prompt_schedule_sha256": f"{step + 100:064x}",
                "window_mode": mode,
                "window_offset": offset,
                "metrics": {"actor/pg_loss": 0.1},
            }
        )
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "diagnostics/window_steps.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return tmp_path, variant


def test_valid_json_window_state_and_resume_log_need_no_torch(window_run):
    run_dir, variant = window_run
    audit = load_audit()
    assert audit.window_issues(run_dir, variant, [1, 2]) == []
    assert "import torch" not in HELPER.read_text()
    assert "import opd_ext" not in HELPER.read_text()


@pytest.mark.parametrize(
    "field,value,issue",
    [
        ("mode", "fixed", "mode"),
        ("block_size", 10, "block_size"),
        ("seed", 910022, "seed"),
        ("last_step", 1, "last_step"),
        ("last_step", True, "last_step"),
        ("current_offset", 3, "current_offset"),
        ("current_offset", -1, "current_offset"),
        ("current_offset", True, "current_offset"),
    ],
)
def test_checkpoint_state_must_match_actual_step_and_variant(window_run, field, value, issue):
    run_dir, variant = window_run
    path = run_dir / "checkpoints/global_step_2/window_state.json"
    state = json.loads(path.read_text())
    state[field] = value
    write_json(path, state)
    assert any(issue in item for item in load_audit().window_issues(run_dir, variant, [1, 2]))


@pytest.mark.parametrize(
    "name",
    [
        "checkpoints/global_step_1/window_state.json",
        "diagnostics/window_steps.jsonl",
        "run_card.json",
        "script_hashes.sha256",
    ],
)
def test_missing_evidence_fails_closed(window_run, name):
    run_dir, variant = window_run
    (run_dir / name).unlink()
    assert load_audit().window_issues(run_dir, variant, [1, 2])


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "missing",
        "mode",
        "offset",
        "hash",
        "malformed",
        "extra",
        "step_type",
        "schedule_hash",
        "metrics",
    ],
)
def test_step_logs_must_be_contiguous_valid_and_agree_with_checkpoints(window_run, mutation):
    run_dir, variant = window_run
    path = run_dir / "diagnostics/window_steps.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "duplicate":
        records.append(records[-1])
    elif mutation == "missing":
        records.pop(0)
    elif mutation == "mode":
        records[-1]["window_mode"] = "fixed"
    elif mutation == "offset":
        records[-1]["window_offset"] = (records[-1]["window_offset"] + 1) % 3
    elif mutation == "hash":
        records[-1]["prompt_batch_sha256"] = "not-a-hash"
    elif mutation == "extra":
        records.append({**records[-1], "step": 3})
    elif mutation == "step_type":
        records[0]["step"] = True
    elif mutation == "schedule_hash":
        del records[0]["prompt_schedule_sha256"]
    elif mutation == "metrics":
        del records[0]["metrics"]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    if mutation == "malformed":
        path.write_text(path.read_text() + "{partial")
    assert load_audit().window_issues(run_dir, variant, [1, 2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("opd_window_mode", "fixed"),
        ("opd_window_seed", 1),
        ("opd_block_size", 10),
        ("opd_block_advantage_mode", "sum"),
        ("ppo_epochs", 2),
        ("window_supervision_sha256", "b" * 64),
    ],
)
def test_label_alone_does_not_establish_window_variant(window_run, field, value):
    run_dir, variant = window_run
    path = run_dir / "run_card.json"
    card = json.loads(path.read_text())
    card[field] = value
    write_json(path, card)
    assert load_audit().window_issues(run_dir, variant, [1, 2])


def test_rng_state_key_is_required_even_for_sliding(window_run):
    run_dir, variant = window_run
    path = run_dir / "checkpoints/global_step_2/window_state.json"
    state = json.loads(path.read_text())
    del state["rng_state"]
    write_json(path, state)
    assert any("rng_state" in item for item in load_audit().window_issues(run_dir, variant, [1, 2]))


def test_rng_state_must_match_mode_contract(window_run):
    run_dir, variant = window_run
    path = run_dir / "checkpoints/global_step_2/window_state.json"
    state = json.loads(path.read_text())
    state["rng_state"] = None if variant == "random3" else {"bit_generator": "PCG64"}
    write_json(path, state)
    assert any("rng_state" in item for item in load_audit().window_issues(run_dir, variant, [1, 2]))


@pytest.mark.parametrize("legacy_exit", [0, 7])
@pytest.mark.parametrize("missing_state", [True, False])
def test_adapter_always_runs_existing_audit_and_combines_acceptance(
    window_run,
    tmp_path,
    legacy_exit,
    missing_state,
):
    run_dir, variant = window_run
    assert ADAPTER.is_file(), "window control audit adapter has not been implemented"
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(HELPER, scripts / HELPER.name)
    shutil.copy(ADAPTER, scripts / ADAPTER.name)
    marker = tmp_path / "legacy_args.json"
    (scripts / "audit_block10_run.py").write_text(
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"pathlib.Path({str(run_dir / 'acceptance.json')!r}).write_text(json.dumps(\n"
        f"    {{'passed': {legacy_exit == 0!r},\n"
        f"      'issues': {[] if legacy_exit == 0 else ['legacy failure']!r},\n"
        "      'checkpoint_steps': [1, 2]}))\n"
        f"sys.exit({legacy_exit})\n"
    )
    state_path = run_dir / "checkpoints/global_step_2/window_state.json"
    if missing_state:
        state_path.unlink()
    args = [
        "--run-dir",
        str(run_dir),
        "--variant",
        variant,
        "--checkpoint-steps",
        "1,2",
        "--skip-eval",
        "--expected-source-commit",
        "b" * 40,
    ]
    result = subprocess.run(
        [sys.executable, str(scripts / ADAPTER.name), *args], capture_output=True, text=True
    )
    expected_pass = not legacy_exit and not missing_state
    assert (result.returncode == 0) == expected_pass
    assert json.loads(marker.read_text()) == args
    acceptance = json.loads((run_dir / "acceptance.json").read_text())
    assert acceptance["passed"] is expected_pass
    assert acceptance["checkpoint_steps"] == [1, 2]
    if legacy_exit:
        assert "legacy failure" in acceptance["issues"]
    elif missing_state:
        assert any("window_state.json" in item for item in acceptance["issues"])
    window_acceptance = json.loads((run_dir / "window_acceptance.json").read_text())
    assert acceptance["window_audit"] == window_acceptance
    assert window_acceptance["passed"] is not missing_state


def test_unsupported_audit_variant_is_rejected():
    with pytest.raises(ValueError, match="variant"):
        load_audit().window_issues(Path("unused"), "block3_mean", [1, 2])


@pytest.mark.parametrize("last_step,passed", [(None, True), (2, True), (3, False), (1, False)])
def test_standalone_cli_is_offline_and_does_not_overwrite_base_acceptance(
    window_run,
    last_step,
    passed,
):
    run_dir, variant = window_run
    base_acceptance = {"passed": True, "checkpoint_steps": [1, 2]}
    write_json(run_dir / "acceptance.json", base_acceptance)
    args = ["--run-dir", str(run_dir), "--variant", variant, "--checkpoint-steps", "1,2"]
    if last_step is not None:
        args += ["--expected-last-step", str(last_step)]
    result = subprocess.run(
        [sys.executable, "-S", str(HELPER), *args], capture_output=True, text=True
    )
    assert (result.returncode == 0) == passed, result.stderr
    assert json.loads((run_dir / "acceptance.json").read_text()) == base_acceptance
    assert json.loads((run_dir / "window_acceptance.json").read_text())["passed"] is passed


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"actor/pg_loss": float("nan")},
        {"actor/pg_loss": float("inf")},
        {"actor/pg_loss": "nan"},
    ],
)
def test_every_step_requires_nonempty_finite_numeric_metrics(window_run, metrics):
    run_dir, variant = window_run
    path = run_dir / "diagnostics/window_steps.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["metrics"] = metrics
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    assert any("metrics" in issue for issue in load_audit().window_issues(run_dir, variant, [1, 2]))

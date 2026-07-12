import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_final_acceptance.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_final_acceptance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_complete_run(run: Path) -> None:
    run.mkdir(parents=True)
    (run / "exit_code.txt").write_text("0\n", encoding="utf-8")
    for step in (50, 100, 200):
        eval_dir = run / f"eval_step_{step}_n8"
        eval_dir.mkdir()
        (eval_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (run / "acceptance.json").write_text(
        json.dumps(
            {
                "passed": True,
                "issues": [],
                "checkpoint_steps": [50, 100, 200],
                "eval_steps": [50, 100, 200],
            }
        ),
        encoding="utf-8",
    )


def test_complete_requires_acceptance_and_every_exit_code(tmp_path):
    module = load_module()
    run = tmp_path / "run"
    write_complete_run(run)

    assert module.acceptance_state(run) == ("complete", "all final artifacts passed")

    (run / "eval_step_100_n8" / "exit_code.txt").unlink()
    state, reason = module.acceptance_state(run)
    assert state == "pending"
    assert "missing" in reason


def test_nonzero_training_or_eval_exit_is_failed(tmp_path):
    module = load_module()
    run = tmp_path / "run"
    write_complete_run(run)
    (run / "eval_step_50_n8" / "exit_code.txt").write_text("7\n")

    state, reason = module.acceptance_state(run)

    assert state == "failed"
    assert "nonzero" in reason


def test_malformed_acceptance_is_failed_not_pending(tmp_path):
    module = load_module()
    run = tmp_path / "run"
    write_complete_run(run)
    (run / "acceptance.json").write_text("{broken", encoding="utf-8")

    state, reason = module.acceptance_state(run)

    assert state == "failed"
    assert "invalid" in reason


def test_passed_false_without_issues_is_failed(tmp_path):
    module = load_module()
    run = tmp_path / "run"
    write_complete_run(run)
    acceptance = json.loads((run / "acceptance.json").read_text())
    acceptance["passed"] = False
    (run / "acceptance.json").write_text(json.dumps(acceptance))

    state, reason = module.acceptance_state(run)

    assert state == "failed"
    assert "passed=false" in reason


def test_null_step_schema_is_failed_not_an_uncaught_pending_exit(tmp_path):
    module = load_module()
    run = tmp_path / "run"
    write_complete_run(run)
    acceptance = json.loads((run / "acceptance.json").read_text())
    acceptance["checkpoint_steps"] = None
    (run / "acceptance.json").write_text(json.dumps(acceptance))

    state, reason = module.acceptance_state(run)

    assert state == "failed"
    assert "schema" in reason


def test_unexpected_checker_exception_uses_dedicated_runtime_exit(monkeypatch, capsys):
    module = load_module()
    monkeypatch.setattr(
        module,
        "acceptance_state",
        lambda _: (_ for _ in ()).throw(RuntimeError("unexpected")),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "/run"])

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 3
    assert '"state": "runtime_error"' in capsys.readouterr().out

import importlib.util
import json
import signal
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_window_queue.py"


def module():
    assert SCRIPT.is_file(), "server-side window queue is not implemented"
    spec = importlib.util.spec_from_file_location("window_queue", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_queue_scope_is_only_two_seed21_variants():
    mod = module()
    assert mod.VARIANTS == ("random3", "sliding3")
    assert mod.STEPS == (50, 100, 200)
    assert mod.TRAINING_SEED == 21


def test_incomplete_or_failed_job_is_not_silently_restarted(tmp_path):
    mod = module()
    job = tmp_path / "job"
    job.mkdir()
    (job / "started_at.txt").write_text("started")
    with pytest.raises(RuntimeError, match="manual"):
        mod.completed_or_new(job)
    (job / "exit_code.txt").write_text("1")
    with pytest.raises(RuntimeError, match="manual"):
        mod.completed_or_new(job)
    (job / "exit_code.txt").write_text("0")
    assert mod.completed_or_new(job)


def test_eval_command_has_full_tasks_and_fixed_sampling(tmp_path):
    mod = module()
    argv = mod.eval_command(tmp_path, "/env/bin/python", tmp_path / "model", tmp_path / "data", tmp_path / "out", tmp_path / "student")
    assert argv[argv.index("--n") + 1] == "8"
    assert argv[argv.index("--max-tokens") + 1] == "16384"
    assert argv[argv.index("--eval-seed") + 1] == "21"
    assert "--enable-thinking" not in argv
    assert argv[argv.index("--tasks") + 1:argv.index("--tasks") + 5] == list(mod.TASKS)


def test_scope_validation_rejects_wrong_mode_or_seed(tmp_path):
    mod = module()
    card = {"variant": "random3", "seed": 21, "total_training_steps": 200, "opd_window_mode": "random", "opd_window_seed": 910021, "source_commit": "abc"}
    mod.validate_card(card, "random3", "abc")
    card["seed"] = 22
    with pytest.raises(ValueError):
        mod.validate_card(card, "random3", "abc")


def test_queue_interrupt_stops_only_its_child_process_group(tmp_path, monkeypatch):
    mod = module()
    calls = []

    class Process:
        pid = 999999
        attempt = 0

        def wait(self, timeout=None):
            self.attempt += 1
            if self.attempt == 1:
                raise KeyboardInterrupt
            return -signal.SIGTERM

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(mod.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        mod.run_job(["unused"], tmp_path / "job", tmp_path, tmp_path / "state.json", {})
    assert calls == [(999999, signal.SIGTERM)]
    assert (tmp_path / "job/exit_code.txt").exists()

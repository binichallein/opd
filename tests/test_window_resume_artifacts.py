import json

from opd_ext import window_supervision as windows


def test_resume_archives_future_diagnostics_without_changing_committed_prefix(tmp_path):
    assert hasattr(windows, "reconcile_diagnostic_resume")
    original = "".join(json.dumps({"step": step}) + "\n" for step in range(1, 4))
    (tmp_path / "window_steps.jsonl").write_text(original)
    (tmp_path / "scalars.jsonl").write_text(original)
    for step in range(1, 4):
        (tmp_path / f"step_{step:06d}.npz").write_bytes(b"fixture")
    archive = windows.reconcile_diagnostic_resume(tmp_path, 1)
    assert (archive / "window_steps.jsonl").read_text() == original
    assert [json.loads(line)["step"] for line in (tmp_path / "window_steps.jsonl").read_text().splitlines()] == [1]
    assert (tmp_path / "step_000001.npz").exists()
    assert not (tmp_path / "step_000002.npz").exists()
    assert (archive / "step_000002.npz").exists()
    assert windows.reconcile_diagnostic_resume(tmp_path, 1) is None

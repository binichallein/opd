import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "scripts" / "analysis_runtime_control.sh"


def test_analysis_control_is_syntax_valid_and_has_both_paired_targets():
    result = subprocess.run(
        ["bash", "-n", str(CONTROL)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    source = CONTROL.read_text(encoding="utf-8")
    assert "train-pair" in source
    assert "ml2-pair" in source
    assert "analysis_deployments/${ANALYSIS_COMMIT}" in source
    assert "FINALIZER_LOCK_PATH" in source
    assert "set -o noclobber" in source
    assert "flock -n" in source
    assert "trap cleanup_control_lock EXIT" in source


def test_analysis_control_deploys_only_committed_analysis_files():
    source = CONTROL.read_text(encoding="utf-8")

    assert 'archive "${ANALYSIS_COMMIT}"' in source
    assert 'cat-file -e "${ANALYSIS_COMMIT}^{commit}"' in source
    for path in (
        "scripts/regrade_opd_eval_external.py",
        "scripts/check_final_acceptance.py",
        "scripts/analysis_runtime_control.sh",
        "scripts/audit_block10_run.py",
        "scripts/compare_paired_opd_evals.py",
        "scripts/build_paired_validation_report.py",
        "scripts/analyze_single_opd_diagnostics.py",
        "scripts/analyze_block10_collapse_diagnostics.py",
        "opd_ext/analysis.py",
    ):
        assert path in source
    assert "refusing source sync while training" not in source
    assert "rm -rf" not in source
    assert 'diff -u "${hashes}" "${remote_hashes}"' in source


def test_analysis_control_runs_complete_non_destructive_finalization_contract():
    source = CONTROL.read_text(encoding="utf-8")

    assert "wait_for_final_acceptance" in source
    assert "WAIT_TIMEOUT_SECONDS" in source
    assert "check_final_acceptance.py" in source
    assert "historical_external_grader_audited" in source
    assert "--steps 50,100,200" in source
    assert 'write_comparison "${EXTERNAL_VIEW}"' in source
    assert "write_comparison outputs" in source
    assert "--bootstrap-replicates 10000" in source
    assert "analyze_single_opd_diagnostics.py" in source
    assert "LOCAL_PLOT_PYTHON" in source
    assert "SCP_BIN" in source
    assert '"${SCP_BIN}" -r' in source
    assert "partial diagnostic output exists; refusing overwrite" in source
    assert "build_paired_validation_report.py" in source
    assert "REPORT_TITLE_B64" in source
    assert "base64 --decode" in source
    assert "finalization_hashes.sha256" in source
    assert "input_hashes.sha256" in source
    assert "output_hashes.sha256" in source
    assert '"${run}/diagnostics"' in source
    assert '"${run}/acceptance.json"' in source
    assert "legacy" not in source.lower()


def test_analysis_control_rejects_unknown_target_without_ssh(tmp_path):
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "#!/bin/sh\nprintf called > \"$SSH_CALLED\"\nexit 99\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    called = tmp_path / "called"

    result = subprocess.run(
        ["bash", str(CONTROL), "unknown"],
        cwd=ROOT,
        env={**os.environ, "SSH_BIN": str(fake_ssh), "SSH_CALLED": str(called)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert not called.exists()

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "supervise_opd_validation.sh"


def test_supervisor_covers_train_pair_and_ml2_token_targets():
    script = SUPERVISOR.read_text(encoding="utf-8")

    assert "train-pair)" in script
    assert "ml2-token)" in script
    assert 'VARIANTS=(token_opd block3_mean)' in script
    assert 'VARIANTS=(token_opd)' in script
    assert "deepseek_justrl_train_control.sh" in script
    assert "token_opd_ml2_control.sh" in script


def test_supervisor_waits_for_success_and_runs_complete_eval_contract():
    script = SUPERVISOR.read_text(encoding="utf-8")

    assert 'EVAL_STEPS=(50 100 200)' in script
    assert 'wait_for_success "${run_dir}"' in script
    assert 'control_action "${variant}" audit-checkpoints' in script
    assert 'control_action "${variant}" eval "${step}"' in script
    assert 'control_action "${variant}" audit' in script
    assert 'kill -0' in script
    assert 'exit_code.txt' in script
    assert 'eval.pid' in script
    assert "cannot continue" in script


def test_supervisor_is_idempotent_and_never_deletes_evidence():
    script = SUPERVISOR.read_text(encoding="utf-8")

    assert 'if ! remote_dir_exists "${run_dir}"; then' in script
    assert 'if ! remote_dir_exists "${eval_dir}"; then' in script
    assert "flock -n 9" in script
    assert "rm -rf" not in script
    assert "rm -f" not in script

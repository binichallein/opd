import json
import os
from pathlib import Path
import subprocess
import threading


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "supervise_opd_validation.sh"


def make_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_supervisor_covers_train_pair_and_ml2_token_targets():
    script = SUPERVISOR.read_text(encoding="utf-8")

    assert "train-pair)" in script
    assert "ml2-token)" in script
    assert 'VARIANTS=(token_opd block3_mean)' in script
    assert 'VARIANTS=(token_opd)' in script
    assert "deepseek_justrl_train_control.sh" in script
    assert "token_opd_ml2_control.sh" in script


def test_supervisor_uses_remote_atomic_launch_lock_and_tristate_ssh():
    script = SUPERVISOR.read_text(encoding="utf-8")

    assert 'lock_path="${RUN_ROOT}/.supervisor-launch.lock"' in script
    assert "LOCK_OWNER_TOKEN" in script
    assert "set -o noclobber" in script
    assert 'owner="$(cat "${lock_path}"' in script
    assert "remote_dir_state" in script
    assert "SSH failure" in script
    assert "wait_for_idle" in script
    assert "flock -n 9" in script
    assert "trap cleanup_remote_launch_lock EXIT" in script
    assert "trap 'handle_signal 130' INT" in script
    assert "trap 'handle_signal 143' TERM" in script


def test_supervisor_validates_pid_identity_and_preserves_final_acceptance():
    script = SUPERVISOR.read_text(encoding="utf-8")

    assert 'kill -0 "${pid}"' in script
    assert '/proc/${pid}/cmdline' in script
    assert 'expected_command="${run_dir}/command.sh"' in script
    assert "checkpoint_acceptance.json" in script
    assert "final_acceptance_state" in script
    assert 'control_action "${variant}" audit-checkpoints' in script
    assert 'launch_if_missing "${eval_dir}" "${variant}" eval "${step}"' in script
    assert 'control_action "${variant}" audit' in script
    assert "rm -rf" not in script
    assert "rm -f" not in script


def test_ssh_transport_failure_never_invokes_control_driver(tmp_path):
    ssh = tmp_path / "ssh-fail"
    driver = tmp_path / "control"
    control_log = tmp_path / "control.log"
    make_executable(ssh, "#!/bin/sh\nexit 255\n")
    make_executable(
        driver,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CONTROL_LOG"\n',
    )
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "CONTROL_LOG": str(control_log),
        "RUN_ROOT_OVERRIDE": str(tmp_path / "runs"),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "SSH failure" in result.stdout + result.stderr
    assert not control_log.exists()


def test_completed_final_run_is_idempotent_and_does_not_reaudit(tmp_path):
    ssh = tmp_path / "ssh-local"
    driver = tmp_path / "control"
    control_log = tmp_path / "control.log"
    make_executable(ssh, '#!/bin/sh\nshift\nexec /bin/bash -c "$*"\n')
    make_executable(
        driver,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CONTROL_LOG"\n',
    )
    run_root = tmp_path / "runs"
    run_dir = run_root / "token_opd"
    run_dir.mkdir(parents=True)
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    for step in (50, 100, 200):
        eval_dir = run_dir / f"eval_step_{step}_n8"
        eval_dir.mkdir()
        (eval_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (run_dir / "acceptance.json").write_text(
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
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "CONTROL_LOG": str(control_log),
        "RUN_ROOT_OVERRIDE": str(run_root),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not control_log.exists()
    assert "already has complete final acceptance" in result.stdout


def test_stale_success_file_does_not_hide_an_active_matching_process(tmp_path):
    ssh = tmp_path / "ssh-local"
    driver = tmp_path / "control"
    control_log = tmp_path / "control.log"
    make_executable(ssh, '#!/bin/sh\nshift\nexec /bin/bash -c "$*"\n')
    make_executable(
        driver,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CONTROL_LOG"\n',
    )
    run_root = tmp_path / "runs"
    run_dir = run_root / "token_opd"
    run_dir.mkdir(parents=True)
    command = run_dir / "command.sh"
    make_executable(command, "#!/bin/sh\nsleep 0.2\n:\n")
    process = subprocess.Popen(["bash", str(command)])
    waiter = threading.Thread(target=process.wait)
    waiter.start()
    (run_dir / "train.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    for step in (50, 100, 200):
        eval_dir = run_dir / f"eval_step_{step}_n8"
        eval_dir.mkdir()
        (eval_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (run_dir / "acceptance.json").write_text(
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
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "CONTROL_LOG": str(control_log),
        "RUN_ROOT_OVERRIDE": str(run_root),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    waiter.join(timeout=2)
    assert not waiter.is_alive()

    assert result.returncode == 0, result.stdout + result.stderr
    assert "waiting run=" in result.stdout
    assert not control_log.exists()


def test_stale_success_file_does_not_hide_pid_reuse(tmp_path):
    ssh = tmp_path / "ssh-local"
    driver = tmp_path / "control"
    control_log = tmp_path / "control.log"
    make_executable(ssh, '#!/bin/sh\nshift\nexec /bin/bash -c "$*"\n')
    make_executable(
        driver,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CONTROL_LOG"\n',
    )
    run_root = tmp_path / "runs"
    run_dir = run_root / "token_opd"
    run_dir.mkdir(parents=True)
    process = subprocess.Popen(["sleep", "2"])
    (run_dir / "train.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "CONTROL_LOG": str(control_log),
        "RUN_ROOT_OVERRIDE": str(run_root),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    process.terminate()
    process.wait(timeout=2)

    assert result.returncode != 0
    assert "pid_mismatch" in result.stdout
    assert not control_log.exists()


def test_supervisor_fails_when_final_audit_does_not_produce_final_acceptance(tmp_path):
    ssh = tmp_path / "ssh-local"
    driver = tmp_path / "control"
    make_executable(ssh, '#!/bin/sh\nshift\nexec /bin/bash -c "$*"\n')
    make_executable(driver, "#!/bin/sh\nexit 0\n")
    run_root = tmp_path / "runs"
    run_dir = run_root / "token_opd"
    run_dir.mkdir(parents=True)
    (run_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    for step in (50, 100, 200):
        eval_dir = run_dir / f"eval_step_{step}_n8"
        eval_dir.mkdir()
        (eval_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
    (run_dir / "acceptance.json").write_text(
        json.dumps(
            {
                "passed": True,
                "issues": [],
                "checkpoint_steps": [50, 100, 200],
                "eval_steps": [],
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "RUN_ROOT_OVERRIDE": str(run_root),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "final audit did not produce complete acceptance" in result.stdout


def test_new_run_root_is_created_and_remote_lock_is_cleaned_on_launch_failure(tmp_path):
    ssh = tmp_path / "ssh-local"
    driver = tmp_path / "control"
    control_log = tmp_path / "control.log"
    make_executable(ssh, '#!/bin/sh\nshift\nexec /bin/bash -c "$*"\n')
    make_executable(
        driver,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CONTROL_LOG"\nexit 7\n',
    )
    run_root = tmp_path / "new" / "runs"
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "CONTROL_LOG": str(control_log),
        "RUN_ROOT_OVERRIDE": str(run_root),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert run_root.is_dir()
    assert not (run_root / ".supervisor-launch.lock").exists()
    assert control_log.read_text(encoding="utf-8").strip() == (
        "ml2-token token_opd formal"
    )


def test_uncertain_lock_acquisition_recovers_its_owner_token(tmp_path):
    ssh = tmp_path / "ssh-local"
    driver = tmp_path / "control"
    control_log = tmp_path / "control.log"
    make_executable(ssh, '#!/bin/sh\nshift\nexec /bin/bash -c "$*"\n')
    make_executable(
        driver,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CONTROL_LOG"\nexit 7\n',
    )
    run_root = tmp_path / "runs"
    run_root.mkdir()
    lock = run_root / ".supervisor-launch.lock"
    lock.write_text("owner-token\n", encoding="utf-8")
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "CONTROL_LOG": str(control_log),
        "RUN_ROOT_OVERRIDE": str(run_root),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "LOCK_OWNER_TOKEN_OVERRIDE": "owner-token",
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "recovered remote launch lock ownership" in result.stdout
    assert not lock.exists()
    assert control_log.read_text(encoding="utf-8").strip() == (
        "ml2-token token_opd formal"
    )


def test_foreign_remote_lock_is_never_removed(tmp_path):
    ssh = tmp_path / "ssh-local"
    driver = tmp_path / "control"
    control_log = tmp_path / "control.log"
    make_executable(ssh, '#!/bin/sh\nshift\nexec /bin/bash -c "$*"\n')
    make_executable(
        driver,
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CONTROL_LOG"\n',
    )
    run_root = tmp_path / "runs"
    run_root.mkdir()
    lock = run_root / ".supervisor-launch.lock"
    lock.write_text("foreign-owner\n", encoding="utf-8")
    env = {
        **os.environ,
        "SSH_BIN": str(ssh),
        "CONTROL_DRIVER": str(driver),
        "CONTROL_LOG": str(control_log),
        "RUN_ROOT_OVERRIDE": str(run_root),
        "LOCK_FILE_OVERRIDE": str(tmp_path / "supervisor.lock"),
        "LOCK_OWNER_TOKEN_OVERRIDE": "our-owner",
        "POLL_SECONDS": "0.01",
    }

    result = subprocess.run(
        ["bash", str(SUPERVISOR), "ml2-token"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert lock.read_text(encoding="utf-8") == "foreign-owner\n"
    assert not control_log.exists()

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_training_command_records_lifecycle_and_exit_code():
    launcher = (ROOT / "scripts" / "launch_revisiting_block_opd_formal_train.sh").read_text()

    assert "started_at.txt" in launcher
    assert "finished_at.txt" in launcher
    assert "exit_code.txt" in launcher
    assert "trap record_exit EXIT" in launcher

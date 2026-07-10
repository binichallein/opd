from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_training_command_records_lifecycle_and_exit_code():
    launcher = (ROOT / "scripts" / "launch_revisiting_block_opd_formal_train.sh").read_text()

    assert "started_at.txt" in launcher
    assert "finished_at.txt" in launcher
    assert "exit_code.txt" in launcher
    assert "trap record_exit EXIT" in launcher


def test_full_eval_has_explicit_reproducibility_controls():
    launcher = (ROOT / "scripts" / "launch_qwen3_math_eval.sh").read_text()
    evaluator = (ROOT / "scripts" / "eval_qwen3_math_vllm.py").read_text()

    assert 'GRADER="${GRADER:-verl}"' in launcher
    assert 'EVAL_SEED="${EVAL_SEED:-21}"' in launcher
    assert "--eval-seed '${EVAL_SEED}'" in launcher
    assert "--grader '${GRADER}'" in launcher
    assert 'seed=eval_seed + rollout_id' in evaluator
    assert 'choices=("verl", "external")' in evaluator
    assert '"enable_thinking": args.enable_thinking' in evaluator


def test_block10_diagnostics_use_memory_safe_teacher_micro_batch():
    control = (ROOT / "scripts" / "block10_diagnostics_control.sh").read_text()
    runner = (ROOT / "scripts" / "run_revisiting_sampled_block_opd_math.sh").read_text()

    assert "REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1" in control
    assert 'ref_log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"' in runner
    assert 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}"' in runner

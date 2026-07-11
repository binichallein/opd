from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_block3_control_fixes_the_approved_training_contract():
    control = (ROOT / "scripts" / "block3_replication_control.sh").read_text()

    expected_fragments = (
        'VARIANT="block3_mean"',
        'ENV_SEED=21',
        'TRAIN_BATCH_SIZE=4',
        'PPO_MINI_BATCH_SIZE=32',
        'ROLLOUT_GROUP_SIZE=8',
        'MAX_PROMPT_LENGTH=2048',
        'MAX_RESPONSE_LENGTH=16384',
        'LEARNING_RATE=2e-6',
        'TOTAL_TRAINING_STEPS="${total_steps}"',
        'ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}"',
        'REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"',
        'OPD_DIAG_INTERVAL=5',
        'OPD_DIAG_TOPK=16',
        'OPD_DIAG_POSITION_STRIDE=1',
        'launch_train formal 200 50,100,200 -1',
    )
    for fragment in expected_fragments:
        assert fragment in control


def test_block3_control_uses_tos_cache_and_complete_eval_contract():
    control = (ROOT / "scripts" / "block3_replication_control.sh").read_text()

    assert 'CACHE_ROOT="${REMOTE_ROOT}/cache/block3_replication_${DATE_TAG}"' in control
    assert "/tmp/opd" not in control
    assert 'N=8 TEMPERATURE=1.0 TOP_P=0.9 MAX_TOKENS=16384' in control
    assert 'EVAL_SEED=21 GRADER=verl ENABLE_THINKING=false' in control
    assert 'TASKS="math500 aime24 aime25 amc23"' in control
    assert '--variant block3_mean --checkpoint-steps 50,100,200 --eval-steps 50,100,200' in control


def test_block3_control_exposes_probe_resume_and_formal_actions():
    control = (ROOT / "scripts" / "block3_replication_control.sh").read_text()

    for action in ("sync", "probe1", "probe2", "formal", "status", "eval", "audit"):
        assert f"{action})" in control
    assert "launch_train probe 2 1 1" in control
    assert "launch_train probe 2 1,2 -1" in control


def test_block3_sync_includes_every_remote_runtime_dependency():
    sync = (ROOT / "scripts" / "sync_block3_replication_to_ml2.sh").read_text()

    for path in (
        "opd_ext/diagnostics.py",
        "scripts/block3_replication_control.sh",
        "scripts/run_revisiting_sampled_block_opd_math.sh",
        "scripts/launch_revisiting_block_opd_formal_train.sh",
        "scripts/launch_qwen3_math_eval.sh",
        "scripts/eval_qwen3_math_vllm.py",
        "scripts/audit_block10_run.py",
        "manifests/revisiting_opd_runtime.sha256",
    ):
        assert path in sync


def test_training_run_card_records_all_micro_batch_controls():
    launcher = (ROOT / "scripts" / "launch_revisiting_block_opd_formal_train.sh").read_text()
    runner = (ROOT / "scripts" / "run_revisiting_sampled_block_opd_math.sh").read_text()

    assert 'ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"' in launcher
    assert 'ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"' in launcher
    assert r'\"actor_ppo_micro_batch_size_per_gpu\": ${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU}' in launcher
    assert r'\"rollout_log_prob_micro_batch_size_per_gpu\": ${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}' in launcher
    assert 'actor_ppo_micro_batch_size_per_gpu="${ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"' in runner
    assert 'rollout_log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"' in runner

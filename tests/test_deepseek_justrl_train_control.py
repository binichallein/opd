from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "scripts" / "deepseek_justrl_train_control.sh"
SYNC = ROOT / "scripts" / "sync_block3_replication_to_ml2.sh"
LAUNCHER = ROOT / "scripts" / "launch_revisiting_block_opd_formal_train.sh"


def test_train_wrapper_locks_models_revisions_host_and_shared_contract():
    control = CONTROL.read_text(encoding="utf-8")

    expected = (
        'REMOTE="train"',
        'HOST_TAG="train"',
        'ASSET_ROOT="/mnt/data/cpfs/Yaleon/opd"',
        'RUNTIME_ROOT="${ASSET_ROOT}/deployments/${SOURCE_COMMIT}"',
        'REMOTE_ROOT="${RUNTIME_ROOT}"',
        'export VARIANT="${VARIANT}"',
        'STUDENT_MODEL="${ASSET_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B"',
        'MATH_TEACHER="${ASSET_ROOT}/models/JustRL-DeepSeek-1.5B"',
        'EXPECTED_STUDENT_MODEL_SUFFIX="DeepSeek-R1-Distill-Qwen-1.5B"',
        'EXPECTED_TEACHER_MODEL_SUFFIX="JustRL-DeepSeek-1.5B"',
        'EXPECTED_STUDENT_REVISION="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"',
        'EXPECTED_TEACHER_REVISION="0637e4096c789c67f9eecbe8355e0bdeddede1c2"',
        'DATA_DIR="${ASSET_ROOT}/data/math_opd_dapo17k_hf_full_eval4"',
        'DATE_TAG="20260712v1"',
        'RUN_TAG="deepseek_justrl_pair"',
        'ROLLOUT_GPU_MEMORY_UTILIZATION="0.6"',
        'ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="1"',
        'ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="4"',
        'REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="1"',
        'BASELINE_ALIGNMENT="DeepSeek-R1-Distill-Qwen-1.5B student and its post-RL JustRL-DeepSeek-1.5B teacher; identical tokenizer and architecture; DAPO-Math-17K"',
    )
    for fragment in expected:
        assert fragment in control


def test_train_wrapper_accepts_only_token_and_block3_variants():
    control = CONTROL.read_text(encoding="utf-8")

    assert 'token_opd|block3_mean)' in control
    assert 'usage: $0 {token_opd|block3_mean}' in control
    assert 'exec bash "${ROOT_DIR}/scripts/block3_replication_control.sh" "$@"' in control


def test_sync_script_can_target_train_without_changing_ml2_defaults():
    sync = SYNC.read_text(encoding="utf-8")

    assert 'SYNC_REMOTE="${SYNC_REMOTE:-ml2}"' in sync
    assert 'TARGET_ROOT="${TARGET_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"' in sync
    assert 'SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"' in sync
    assert 'ssh "${SYNC_REMOTE}"' in sync
    assert "ssh ml2" not in sync
    assert 'git -C "${ROOT_DIR}" cat-file -e "${SOURCE_COMMIT}^{commit}"' in sync
    assert 'git -C "${ROOT_DIR}" diff --quiet "${SOURCE_COMMIT}" HEAD -- "${FILES[@]}"' in sync
    assert 'git -C "${ROOT_DIR}" ls-tree "${SOURCE_COMMIT}" external/revisiting_opd' in sync
    assert 'test "${expected_submodule_commit}" = "${actual_submodule_commit}"' in sync
    assert 'sha256sum -c "${ROOT_DIR}/manifests/revisiting_opd_runtime.sha256"' in sync
    assert 'git -C "${ROOT_DIR}" status --porcelain -- "${FILES[@]}"' in sync
    assert "submodule_dirty=" not in sync


def test_formal_launcher_records_and_checks_pinned_model_revisions():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'STUDENT_MODEL_REVISION="${STUDENT_MODEL_REVISION:-}"' in launcher
    assert 'TEACHER_MODEL_REVISION="${TEACHER_MODEL_REVISION:-}"' in launcher
    assert '\\"student_model_revision\\": \\"${STUDENT_MODEL_REVISION}\\"' in launcher
    assert '\\"teacher_model_revision\\": \\"${TEACHER_MODEL_REVISION}\\"' in launcher
    assert "HF_REVISION" in launcher
    assert 'BASELINE_ALIGNMENT="${BASELINE_ALIGNMENT:-' in launcher
    assert '\\"baseline_alignment\\": \\"${BASELINE_ALIGNMENT}\\"' in launcher

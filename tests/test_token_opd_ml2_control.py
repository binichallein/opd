from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERIC_CONTROL = ROOT / "scripts" / "block3_replication_control.sh"
TOKEN_CONTROL = ROOT / "scripts" / "token_opd_ml2_control.sh"


def test_generic_control_keeps_block3_defaults_but_accepts_variant_overrides():
    control = GENERIC_CONTROL.read_text(encoding="utf-8")

    assert 'REMOTE="${REMOTE:-ml2}"' in control
    assert 'HOST_TAG="${HOST_TAG:-ml2}"' in control
    assert 'ASSET_ROOT="${ASSET_ROOT:-/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd}"' in control
    assert 'SOURCE_COMMIT="${SOURCE_COMMIT:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"' in control
    assert 'VARIANT="${VARIANT:-block3_mean}"' in control
    assert 'RUN_TAG="${RUN_TAG:-block3_replication}"' in control
    assert 'PROJECT_NAME="${PROJECT_NAME:-opd_block3_replication}"' in control
    assert 'EXP_PREFIX="${EXP_PREFIX:-block3-mean-replication-ml2}"' in control
    assert 'VARIANT="${VARIANT}"' in control
    assert '--variant "${VARIANT}"' in control
    assert '--expected-student-model-suffix "${EXPECTED_STUDENT_MODEL_SUFFIX}"' in control
    assert '--expected-teacher-model-suffix "${EXPECTED_TEACHER_MODEL_SUFFIX}"' in control
    assert '--expected-student-model-revision "${EXPECTED_STUDENT_REVISION}"' in control
    assert '--expected-teacher-model-revision "${EXPECTED_TEACHER_REVISION}"' in control


def test_token_control_locks_the_paired_ml2_contract_and_immutable_runtime():
    control = TOKEN_CONTROL.read_text(encoding="utf-8")

    expected_fragments = (
        'REMOTE="ml2"',
        'HOST_TAG="ml2"',
        'ASSET_ROOT="/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd"',
        'SOURCE_COMMIT="9b3b8b76bcdf02d0a4cfe4720cecb24bc7203977"',
        'RUNTIME_ROOT="${ASSET_ROOT}/deployments/${SOURCE_COMMIT}"',
        'REMOTE_ROOT="${RUNTIME_ROOT}"',
        'DATE_TAG="20260712v1"',
        'VARIANT="token_opd"',
        'RUN_TAG="token_opd_replication"',
        'PROJECT_NAME="opd_token_replication"',
        'EXP_PREFIX="token-opd-replication-ml2"',
        'CACHE_ROOT="/limx_embap/tos/tor/0712v1"',
        'ROLLOUT_GPU_MEMORY_UTILIZATION="0.6"',
        'ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU="1"',
        'ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="4"',
        'REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="1"',
        'exec bash "${ROOT_DIR}/scripts/block3_replication_control.sh" "$@"',
    )
    for fragment in expected_fragments:
        assert fragment in control

    assert "sync)" in control
    assert "immutable runtime is already deployed" in control
    assert "/tmp/" not in control


def test_token_control_inherits_full_diagnostics_checkpoints_and_fixed_eval():
    generic = GENERIC_CONTROL.read_text(encoding="utf-8")
    token = TOKEN_CONTROL.read_text(encoding="utf-8")

    assert 'DIAGNOSTIC_SAVE_STEPS="${milestones}"' in generic
    assert 'launch_train formal 200 50,100,200' in generic
    assert 'N=8 TEMPERATURE=1.0 TOP_P=0.9 MAX_TOKENS=16384' in generic
    assert 'EVAL_SEED=21 GRADER=verl ENABLE_THINKING=false' in generic
    assert 'TASKS="math500 aime24 aime25 amc23"' in generic
    assert 'checkpoint_policy' not in token

#!/usr/bin/env python3
"""Offline Qwen4 completion paper assets; no SSH, training, inference or regrading.

Commands (run from the repository root):
  python scripts/build_qwen4_paper_assets.py schema
  python scripts/build_qwen4_paper_assets.py compact-npz /local/step_00001.npz
  python scripts/build_qwen4_paper_assets.py export --artifact-root /local/mirror
  python scripts/build_qwen4_paper_assets.py build --input /local/qwen4.json
  python scripts/build_qwen4_paper_assets.py figures --input /local/qwen4.json \
    --figures-dir paper/iclr2027/figures/qwen4_partial --allow-partial \
    --only qwen4_accuracy qwen4_pass
  python scripts/build_qwen4_paper_assets.py merge-diagnostics \
    --input /local/qwen4_eval.json --diagnostics-from /local/qwen4_diagnostic_cache.json

Input schema v1:
  data_sha256: {math500: SHA256, aime24: SHA256, aime25: SHA256, amc23: SHA256}
  arms: {block3_mean: ARM, token_opd: ARM}; an absent arm is partial.
  ARM: training_root, evaluation_root, checkpoints: {"200": CHECKPOINT, ...},
       optional scalars: {source: SOURCE, records: [SCALAR]}, positions: [POSITION].
  CHECKPOINT: source: SOURCE for acceptance.json; result: original accepted JSON
       (at least passed/model/completion_protocol_verified/grader_sha256/per_task/
       rollout_archive/sha256); eval_card: original eval_card.json plus source;
       optional questions: {sources: [SOURCE], rows: [QUESTION]}.
       Question sources must be the four accepted outputs/TASK_graded.jsonl files;
       retain their hashes in result.sha256 as well as the four data hashes.
       The legacy Block3 eval card may omit variant; Token must declare token_opd.
  QUESTION: task, id (string), correct_count, missing_box_count, length_stop_count.
       Exactly one row per benchmark question, counts across all eight rollouts.
       No response text, token IDs, per-rollout logs or model arrays belong here.
  SCALAR: step plus any of SCALAR_METRICS, using the original scalar field names.
       Curate scalars.jsonl to these fields; duplicate steps are rejected.
  POSITION: compact-npz output, preserving metadata, bin sums and coverage.
       Override source.path with the original remote NPZ path after local export;
       retain the SHA256 of the unmodified source NPZ. Never average worker means.
  SOURCE: {path: original absolute artifact path, sha256: original artifact SHA256}.

Source hashes attest export lineage, not re-audits of unavailable remote files.
Every result's benchmark data hashes must match data_sha256. Eval cards must
match the frozen contract. Missing optional questions yield NO confidence bounds.
Bootstrap resamples paired questions within each benchmark, conditional on the
one seed21 training pair; it does not estimate training-run variance.
The export command reads only accepted metadata, selected diagnostic NPZs/scalars
and the accepted graded files for requested question steps (default: 200). It
streams graded files into question counts, checks their accepted hashes, and
prints a compact bundle to stdout; it never writes to the artifact store.
Diagnostics default to every training step 1..200 for both arms (400 NPZs),
not five snapshots. Missing requested NPZs fail the export. Use --question-steps
with no values for a one-time diagnostic cache without question-score scans;
use --no-diagnostics for later evaluation refreshes. merge-diagnostics combines
the fresh evaluation bundle with that cache locally and prints JSON to stdout.
It preserves fresh checkpoints and original diagnostic hashes, requires matching
fixed experiment roots/data hashes, and rejects conflicting existing diagnostics.
Both position and scalar coverage must be 1..200 unless explicitly overridden by
--allow-partial-diagnostics. Partial evaluation queues remain subject to build's
separate --allow-partial guard. Inspect schema for metric definitions.
The builder never reads raw rollouts. Outputs are created exclusively, never
replaced; rerun into fresh output directories. Matplotlib is needed for plots.
The figures command adds selected plots and qwen4_figures_provenance.json without
rewriting tables, bundles or prior figures. Publication accuracy/pass panels use
5.5 x 3.6 inches, and vertically paired heatmaps use 5.5 x 5 inches, all at 8 pt.
Training scalar panels use 5.5 x 5.5 inches at 8 pt. Diagnostic figure completeness
is determined by its own 200-step metric coverage, not evaluation-queue progress.
qwen4_coverage uses student_entropy.max_position_count: the maximum valid count
at any position within each bin, not summed token-position observations. Its
shared scale is 0..32; only unobserved bins/steps are gray, with no min-count cut.
For entropy heatmaps, --min-count 8 tests that same maximum, not the bin sum.
qwen4_scores is a larger internal 4 x 4 diagnostic grid, not a main-paper figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
from pathlib import Path

import numpy as np

ROOT = Path("/limx_embap/tos/user/Yaleon/opd_block_experiments_20260709/opd")
DATA = ROOT / "data/math_opd_dapo17k_hf_full_eval4/eval_jsonl"
TRAIN_COMMIT = "0f9161f02f08287fb07f0375ad0a6bda81133ff0"
GRADER_SHA = "04f7a0328be18409b55836f7a794dd31fbc982d5870bc542a8fe7c91f9490d7f"
TASKS = {"math500": 500, "aime24": 30, "aime25": 30, "amc23": 83}
TASK_LABELS = {"math500": "MATH500", "aime24": "AIME24", "aime25": "AIME25", "amc23": "AMC23"}
STEPS = (200, 150, 100, 50)
TRAINING_STEPS = tuple(range(1, 201))
VARIANTS = ("block3_mean", "token_opd")
LABELS = {"block3_mean": "Block3 Mean", "token_opd": "Token OPD"}
COLORS = {"block3_mean": "#0072B2", "token_opd": "#D55E00"}
TRAIN_ROOTS = {
    "block3_mean": ROOT / "runs/20260920v3_qwen4_completion_blockfirst_seed21_ml2/block3_mean",
    "token_opd": ROOT / "runs/20260920v1_qwen4_completion_token_seed21_ml2/token_opd",
}
EVAL_ROOTS = {
    "block3_mean": ROOT / "runs/20260920v1_qwen4_completion_block3_eval_seed21_ml2",
    "token_opd": ROOT / "runs/20260921v1_qwen4_completion_token_eval_seed21_ml2",
}
EVAL_CONTRACT = {
    "training_source_commit": TRAIN_COMMIT,
    "prompt_protocol": "qwen3_completion_boxed_v1",
    "n": 8,
    "temperature": 1.0,
    "top_p": 0.9,
    "max_tokens": 16384,
    "eval_seed": 21,
    "rollout_seeds": list(range(21, 29)),
    "grader": "external",
    "grader_sha256": GRADER_SHA,
    "enable_thinking": False,
    "retain_rollouts": True,
}
METRICS = ("avg_at_8", "pass_at_8", "format_error_rate", "engine_truncation_ratio")
SCALAR_METRICS = (
    "diagnostics/student_entropy",
    "diagnostics/teacher_entropy",
    "actor/grad_norm",
    "response_length/clip_ratio",
    "diagnostics/sign_flip_rate",
    "diagnostics/weighted_sign_flip_rate",
    "diagnostics/leakage_magnitude",
    "diagnostics/normalized_leakage",
    "diagnostics/raw_token_advantage_std",
    "diagnostics/block_advantage_std",
    "diagnostics/post_update_block_log_ratio_abs_mean",
    "diagnostics/post_update_block_ratio_outside_clip_fraction",
)
# These names and denominators match the frozen ray_trainer_multitask.py emitter.
POSITION_METRICS = {
    "student_entropy": "Student entropy (nats)",
    "teacher_entropy": "Teacher entropy on student prefixes (nats)",
    "entropy_gap_signed": "Teacher minus student entropy (nats)",
    "entropy_gap_absolute": "Absolute entropy gap (nats)",
    "sign_flip": "Sign-flip fraction among eligible positions",
    "leakage": "Absolute shared-mean advantage deviation",
    "overlap_ratio": "Top-16 overlap ratio",
    "student_overlap_mass": "Student probability mass on top-16 intersection",
    "teacher_overlap_mass": "Teacher probability mass on top-16 intersection",
    "overlap_token_advantage": "Advantage on valid top-16 overlap positions",
    "post_update_block_log_ratio_abs": "Post-update block absolute log-ratio",
    "post_update_block_outside_clip": "Post-update block outside-clip fraction",
}
SIGNED_METRICS = {"entropy_gap_signed", "overlap_token_advantage"}
RATE_POSITIONS = {
    "sign_flip",
    "overlap_ratio",
    "student_overlap_mass",
    "teacher_overlap_mass",
    "post_update_block_outside_clip",
}
UNCERTAINTY_SCOPE = (
    "95% percentile paired-question bootstrap, separately within each benchmark; "
    "Block3 minus Token in percentage points. Conditional on one seed21 training "
    "pair and eight recorded rollouts/question; not a training-run or multi-seed CI. "
    "Pointwise intervals, not simultaneous checkpoint or benchmark bounds."
)
DEFINITIONS = {
    "entropy": "Full-vocabulary Shannon entropy in nats: p_X(v|h)=softmax(z_X(h)/T)_v; "
    "H_X(h)=-sum_{v in V} p_X(v|h)*log(p_X(v|h)), X=student or teacher. "
    "Not sampled-token surprisal -log p_X(y|h), and not top-16 renormalized entropy.",
    "entropy_temperature": "Both diagnostic forwards divide logits by rollout temperature T=1.0. "
    "Generation top_p=0.9 affects the sampled student prefixes, not the entropy softmax: "
    "no nucleus truncation/renormalization is applied to diagnostic logits.",
    "entropy_timing": "The diagnostic entropy is pre-update, on that step's student rollout batch. "
    "Step200 is not a full-epoch average or post-update checkpoint200 validation entropy. "
    "The separate post_update_block_* ratio diagnostics are computed after that update.",
    "entropy_scalar": "diagnostics/student_entropy and diagnostics/teacher_entropy: "
    "sum_{i,t} m_it*H_X(h_it) / max(sum_{i,t} m_it,1). Token-weighted over the batch, "
    "not an equal-weight mean of trajectory means. The signed gap is teacher minus student; "
    "the absolute gap is mean(abs(H_teacher-H_student)), not abs(mean gap).",
    "teacher_entropy_limit": "Full-vocabulary teacher entropy is not missing from these Qwen4 "
    "diagnostic scalars/NPZs. It is conditional on each arm's own student-generated prefixes, "
    "not teacher-generated or fixed common validation states. Cross-arm differences alone "
    "do not identify teacher drift, OOD or causation. Stored entropy/statistics cannot recover "
    "the teacher's full probability vector or full-vocabulary KL at every prefix.",
    "response_mask": "m_it is the response suffix of attention_mask, using the unchanged "
    "historical EOS mask with EOS=151643: m_it=1 iff no EOS occurs at positions <t. "
    "The first EOS is included; subsequent positions and the prompt are excluded. "
    "At the 16384-position cap without EOS, all positions are valid. This uses the stored "
    "training tensor/mask, not a mask reconstructed from text or engine finish_reason.",
    "position_sufficient_statistics": "NPZ metric__sum[t]=sum_i m_it*x_it, "
    "metric__squared_sum[t]=sum_i m_it*x_it^2, metric__valid_count[t]=sum_i m_it; "
    "use the metric-specific mask (eligible for sign_flip; overlap-valid for overlap advantage). "
    "Arrays have 16384 absolute response positions. position_bin=128 metadata does not mean "
    "they were already binned. topk=16/position_stride=1 were checked in both real Step1 NPZs.",
    "block_credit": "Let a_it be the actual sampled-token OPD advantage. For fixed block B "
    "of size 3 (Token: size 1), Abar_iB=sum_{t in B} m_it*a_it / max(sum_{t in B} m_it,1). "
    "The incomplete final block divides by its valid-token count, not always 3. "
    "Abar is repeated onto each valid token for the credit diagnostics.",
    "sign_flip_formula": "diagnostics/sign_flip_rate; NPZ sign_flip. epsilon=1e-4; "
    "e_it=m_it*1[abs(a_it)>epsilon]*1[abs(Abar_iB)>epsilon]; "
    "SignFlipRate=sum e_it*1[sign(a_it)!=sign(Abar_iB)] / max(sum e_it,1). "
    "WeightedSignFlipRate replaces e_it weights with e_it*abs(a_it), denominator floor epsilon.",
    "leakage_formula": "diagnostics/leakage_magnitude; NPZ leakage. "
    "LeakageMagnitude=sum m_it*abs(Abar_iB-a_it)/max(sum m_it,1). "
    "diagnostics/normalized_leakage=sum m_it*abs(Abar_iB-a_it) / "
    "max(sum m_it*abs(a_it),1e-4). This is a credit-sharing deviation, "
    "not a directly measured parameter-gradient leakage or causal effect.",
    "advantage_scale_ratio": "There is no native field called scale_ratio in these logs. "
    "An explicitly derived advantage standard-deviation ratio is "
    "R_std=diagnostics/block_advantage_std / diagnostics/raw_token_advantage_std. "
    "Both stds use unbiased=False over finite valid response tokens; the shared block mean "
    "is repeated over its valid tokens before computing the block_advantage_std. "
    "If the denominator is zero, the ratio is undefined (NA), not epsilon-regularized. "
    "This is neither normalized_leakage, a parameter-gradient norm ratio, nor a PPO ratio. "
    "It alone does not establish reduction of parameter-gradient variance.",
    "post_update_block_formula": "L_iB=sum_{t in B} m_it*(log p_after(y_it|h_it) "
    "- log p_behavior(y_it|h_it)); r_iB=exp(L_iB) "
    "(implementation caps the exponent at 80 and separately counts overflow/underflow). "
    "NPZ post_update_block_log_ratio_abs stores abs(L_iB) expanded to valid tokens; "
    "post_update_block_outside_clip stores 1[r_iB<1-clip_low or r_iB>1+clip_high], "
    "also expanded. Nonfinite blocks are excluded; numerical counters are separate. "
    "Scalar *_mean/outside_clip_fraction average finite nonempty blocks, while binned "
    "position means weight the expanded valid tokens; these are different denominators.",
    "avg_at_8": "Mean fraction correct among eight rollouts per question.",
    "pass_at_8": "Fraction of questions with at least one correct rollout among eight.",
    "missing_box": "Rollout fraction without the literal \\boxed marker; "
    "not boxed-answer validity.",
    "eval_truncation": "Rollout fraction whose generation engine finish_reason is length.",
    "training_cap": "response_length/clip_ratio: masked response length equals tensor width, "
    "16384 in this frozen run; not an engine finish-reason measurement.",
    "grad_norm": "actor/grad_norm: global parameter gradient norm returned before clipping.",
    "position_mean": "Sum of metric observations / sum of metric-specific valid_count in each "
    "absolute output-position bin. Not a mean of per-position means.",
    "position_coverage": "Maximum per-position valid_count within a bin, not total token count "
    "or a count of distinct trajectories reaching every position in the bin.",
    "coverage_figure": "qwen4_coverage uses student_entropy.max_position_count with a shared "
    "0..32 scale. Zero-count/unobserved bins or absent steps are gray; positive counts "
    "are never thresholded. Metric heatmaps instead retain a bin when its metric-specific "
    "maximum per-position count is >= min_count (default 8); their mean uses all valid "
    "observations in that retained bin, not only positions meeting the threshold.",
    "overlap_advantage_formula": "NPZ overlap_token_advantage: O=top16(student) intersect "
    "top16(teacher); pbar and qbar renormalize each full-softmax distribution on O. "
    "The value is -KL(pbar||qbar)/|O|, with only response positions having |O|>0 valid. "
    "This is an overlap-restricted diagnostic, not the full-vocabulary KL.",
    "sign_flip": "Eligible: valid token and both token/shared-mean advantages exceed sign_epsilon "
    "in absolute value. Ratio uses eligible_count, not all response tokens.",
    "position_gradient_space": "No parameter-gradient heatmap is inferred: NPZ metadata says "
    "sampled_log_probs_not_model_parameters.",
    "lineage": "Original artifact hashes are supplied by the exporter; this offline build "
    "hashes the compact input and outputs but does not re-audit remote raw artifacts.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sha(value):
    require(
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        "invalid sha256",
    )


def check_source(value, expected=None):
    require(isinstance(value, dict), "source must contain path and sha256")
    check_sha(value.get("sha256"))
    path = value.get("path")
    require(isinstance(path, str) and Path(path).is_absolute(), "source path must be absolute")
    if expected is not None:
        require(path == str(expected), f"source identity mismatch: {expected}")


def finite_number(value, name, low=None, high=None):
    require(type(value) in (int, float) and math.isfinite(value), f"{name} must be finite numeric")
    require(low is None or value >= low, f"{name} below range")
    require(high is None or value <= high, f"{name} above range")
    return float(value)


def integer(value, name, low, high):
    require(type(value) is int and low <= value <= high, f"invalid {name}")
    return value


def reject_raw(value):
    forbidden = {
        "response",
        "response_text",
        "response_token_ids",
        "training_response_token_ids",
        "prompt_token_ids",
        "rendered_prompt",
        "rollout_log_probs",
        "state_dict",
        "logits",
    }
    if isinstance(value, dict):
        require(not forbidden.intersection(value), "raw payload is forbidden in paper inputs")
        for item in value.values():
            reject_raw(item)
    elif isinstance(value, list):
        for item in value:
            reject_raw(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "all compact numeric input must be finite")


def check_task(result, task):
    count = TASKS[task]
    integer(result["num_examples"], "num_examples", count, count)
    integer(result["num_rollouts"], "num_rollouts", count * 8, count * 8)
    for metric in METRICS:
        finite_number(result[metric], metric, 0, 1)
    avg, passed = result["avg_at_8"], result["pass_at_8"]
    require(avg <= passed + 1e-12 and passed <= 8 * avg + 1e-12, "inconsistent Avg/Pass summary")
    for value, denominator in ((avg, count * 8), (passed, count)):
        require(
            math.isclose(value * denominator, round(value * denominator), abs_tol=1e-8),
            "score is not supported by integer counts",
        )
    for key, rate in (
        ("format_error_rollouts", "format_error_rate"),
        ("engine_length_stop_rollouts", "engine_truncation_ratio"),
    ):
        n = integer(result[key], key, 0, count * 8)
        require(
            math.isclose(n / (count * 8), result[rate], abs_tol=1e-12, rel_tol=0),
            f"inconsistent {rate} count/denominator",
        )


def question_groups(questions, result, variant, step):
    require(
        isinstance(questions.get("sources"), list) and questions["sources"],
        "question scores require source hashes",
    )
    folder = EVAL_ROOTS[variant] / f"evaluations/{variant}_step{step}/outputs"
    expected_paths = {str(folder / f"{task}_graded.jsonl") for task in TASKS}
    require(
        len(questions["sources"]) == len(TASKS)
        and {item["path"] for item in questions["sources"]} == expected_paths,
        "question sources must cover the four accepted graded files",
    )
    for item in questions["sources"]:
        check_source(item)
        require(
            result["sha256"].get(item["path"]) == item["sha256"],
            "question source hash differs from accepted grading evidence",
        )
    summaries = result["per_task"]
    groups = {task: {} for task in TASKS}
    for row in questions["rows"]:
        require(
            set(row) == {"task", "id", "correct_count", "missing_box_count", "length_stop_count"},
            "question row must contain only compact counts and identity",
        )
        task, identity = row["task"], row["id"]
        require(
            task in TASKS and isinstance(identity, str) and identity, "invalid question identity"
        )
        require(identity not in groups[task], "duplicate question identity")
        for key in ("correct_count", "missing_box_count", "length_stop_count"):
            integer(row[key], key, 0, 8)
        groups[task][identity] = row
    for task, count in TASKS.items():
        rows = list(groups[task].values())
        require(len(rows) == count, f"incomplete question identities for {task}")
        actual = (
            sum(r["correct_count"] for r in rows) / (8 * count),
            sum(r["correct_count"] > 0 for r in rows) / count,
            sum(r["missing_box_count"] for r in rows) / (8 * count),
            sum(r["length_stop_count"] for r in rows) / (8 * count),
        )
        require(
            all(
                math.isclose(value, summaries[task][key], abs_tol=1e-12, rel_tol=0)
                for key, value in zip(METRICS, actual)
            ),
            f"question counts disagree with accepted summary: {task}",
        )
    return {task: list(rows.values()) for task, rows in groups.items()}


def paired_bootstrap(left, right, *, replicates=10000, seed=20260921):
    integer(replicates, "bootstrap replicates", 1, 1_000_000)
    integer(seed, "bootstrap seed", 0, 2**63 - 1)
    left_by_id, right_by_id = ({r["id"]: r for r in rows} for rows in (left, right))
    require(
        len(left_by_id) == len(left) and len(right_by_id) == len(right),
        "duplicate question identity",
    )
    require(left_by_id and set(left_by_id) == set(right_by_id), "paired question identities differ")
    ids = sorted(left_by_id)
    arrays = [
        np.asarray([integer(rows[k]["correct_count"], "correct_count", 0, 8) for k in ids])
        for rows in (left_by_id, right_by_id)
    ]
    differences = np.column_stack(
        ((arrays[0] - arrays[1]) / 8, (arrays[0] > 0).astype(float) - (arrays[1] > 0))
    )
    rng = np.random.default_rng(seed)
    sampled = np.empty((replicates, 2))
    # Batch the same paired indices for both metrics; never resample rollout rows.
    for start in range(0, replicates, 256):
        end = min(start + 256, replicates)
        indices = rng.integers(0, len(ids), size=(end - start, len(ids)))
        sampled[start:end] = differences[indices].mean(axis=1) * 100
    bounds = np.quantile(sampled, [0.025, 0.975], axis=0)
    return {
        metric: {
            "delta_pp": float(differences[:, i].mean() * 100),
            "lower_95_pp": float(bounds[0, i]),
            "upper_95_pp": float(bounds[1, i]),
        }
        for i, metric in enumerate(METRICS[:2])
    }


def check_stats(stats, length, *, binned=False, bin_size=1):
    keys = {"sum", "squared_sum", "valid_count"}
    if binned:
        keys.add("max_position_count")
    require(set(stats) == keys, "invalid position statistic names")
    arrays = {key: np.asarray(stats[key], dtype=float) for key in keys}
    require(
        all(a.shape == (length,) and np.isfinite(a).all() for a in arrays.values()),
        "position statistics must be aligned finite vectors",
    )
    sums, squares, counts = (arrays[k] for k in ("sum", "squared_sum", "valid_count"))
    require(np.all(counts >= 0) and np.all(counts == np.floor(counts)), "invalid position counts")
    require(np.all(squares >= 0), "negative squared sums")
    require(
        np.all(sums[counts == 0] == 0) and np.all(squares[counts == 0] == 0),
        "unobserved positions must have zero sufficient statistics",
    )
    require(
        np.all(sums**2 <= squares * counts + 1e-5 * np.maximum(1, squares * counts)),
        "inconsistent position second moments",
    )
    coverage = arrays["max_position_count"] if binned else counts
    require(
        np.all((coverage >= 0) & (coverage <= 32) & (coverage == np.floor(coverage))),
        "invalid per-position coverage for a 32-trajectory batch",
    )
    require(
        np.all((counts >= coverage) & (counts <= coverage * bin_size)),
        "bin count and per-position coverage disagree",
    )
    return arrays


def compact_npz(path, *, bin_size=128):
    """Reduce actual 16384-position NPZ sufficient statistics without loading pickle."""
    integer(bin_size, "position bin size", 1, 16384)
    require(16384 % bin_size == 0, "position bin size must divide 16384")
    path = Path(path)
    before = sha256(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = {
            key: data[key].item() for key in data.files if "__" not in key and key != "step"
        }
        step = data["step"].item()
        integer(step, "diagnostic step", 1, 200)
        names = {key.split("__", 1)[0] for key in data.files if "__" in key}
        require(
            names and names <= set(POSITION_METRICS),
            "unknown position metric; inspect emitter first",
        )
        metrics = {}
        for name in sorted(names):
            stats = {key: data[f"{name}__{key}"] for key in ("sum", "squared_sum", "valid_count")}
            arrays = check_stats(stats, 16384)
            metrics[name] = {
                key: array.reshape(-1, bin_size).sum(axis=1).tolist()
                for key, array in arrays.items()
            }
            metrics[name]["max_position_count"] = (
                arrays["valid_count"].reshape(-1, bin_size).max(axis=1).tolist()
            )
    require(before == sha256(path), "NPZ changed during compaction")
    item = {
        "step": step,
        "bin_size": bin_size,
        "metadata": metadata,
        "metrics": metrics,
        "source": {"path": str(path.resolve()), "sha256": before},
    }
    check_position(item)
    return item


def check_position(item):
    check_source(item["source"])
    integer(item["step"], "diagnostic step", 1, 200)
    size = integer(item["bin_size"], "position bin size", 1, 16384)
    require(16384 % size == 0, "position bin size must divide 16384")
    meta = item["metadata"]
    require(
        meta.get("topk") == 16 and meta.get("position_stride") == 1,
        "unexpected topk/position_stride; do not relabel unknown diagnostics",
    )
    require(
        meta.get("gradient_diagnostic_space") == "sampled_log_probs_not_model_parameters",
        "unknown gradient diagnostic space",
    )
    require(
        meta.get("block_ratio_definition") == "post_update_policy_vs_behavior_policy",
        "unknown block ratio definition",
    )
    check_sha(meta["prompt_batch_sha256"])
    require(meta.get("sign_epsilon") == 1e-4, "unknown sign-flip eligibility epsilon")
    require(
        item["metrics"] and set(item["metrics"]) <= set(POSITION_METRICS),
        "unknown or empty position metrics",
    )
    for name, stats in item["metrics"].items():
        arrays = check_stats(stats, 16384 // size, binned=True, bin_size=size)
        if name not in SIGNED_METRICS:
            require(np.all(arrays["sum"] >= -1e-8), f"negative {name}")
        if name in RATE_POSITIONS:
            require(np.all(arrays["sum"] <= arrays["valid_count"] + 1e-8), f"invalid {name} ratio")


def position_means(item, metric, *, min_count=8):
    integer(min_count, "min_count", 1, 32)
    stats = item["metrics"][metric]
    counts = np.asarray(stats["valid_count"], dtype=float)
    valid = (np.asarray(stats["max_position_count"]) >= min_count) & (counts > 0)
    values = np.full(counts.shape, np.nan)
    np.divide(stats["sum"], counts, out=values, where=valid)
    return values


def diagnostics(arm, variant):
    scalar_rows, positions = [], arm.get("positions", [])
    if "scalars" in arm:
        value = arm["scalars"]
        check_source(value["source"], TRAIN_ROOTS[variant] / "diagnostics/scalars.jsonl")
        seen = set()
        for row in value["records"]:
            step = integer(row["step"], "scalar step", 1, 200)
            require(step not in seen, "duplicate scalar step; curate resumed records explicitly")
            require(set(row) <= {"step", *SCALAR_METRICS}, "curate scalars to documented fields")
            require(len(row) > 1, "empty scalar record")
            seen.add(step)
            for key, value in row.items():
                if key != "step":
                    finite_number(value, key, 0, 1 if key == "response_length/clip_ratio" else None)
            scalar_rows.append(dict(row))
    seen = set()
    for item in positions:
        check_position(item)
        step = item["step"]
        require(step not in seen, "duplicate position step")
        seen.add(step)
        check_source(item["source"], TRAIN_ROOTS[variant] / f"diagnostics/step_{step:05d}.npz")
    return {
        "scalars": sorted(scalar_rows, key=lambda x: x["step"]),
        "positions": sorted(positions, key=lambda x: x["step"]),
    }


def analyze(bundle, *, allow_partial=False, bootstrap_replicates=10000, bootstrap_seed=20260921):
    """Validate all present evidence first. Partial only permits absent checkpoints."""
    try:
        return _analyze(bundle, allow_partial, bootstrap_replicates, bootstrap_seed)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed compact input: {error}") from error


def _analyze(bundle, allow_partial, replicates, seed):
    reject_raw(bundle)
    require(bundle["schema_version"] == 1, "unsupported schema_version")
    require(set(bundle["data_sha256"]) == set(TASKS), "all four data hashes required")
    for value in bundle["data_sha256"].values():
        check_sha(value)
    require(set(bundle["arms"]) <= set(VARIANTS), "unknown experiment arm")
    integer(replicates, "bootstrap replicates", 1, 1_000_000)
    integer(seed, "bootstrap seed", 0, 2**63 - 1)
    rows, missing, questions, diag = [], [], {}, {}
    for variant in VARIANTS:
        arm = bundle["arms"].get(variant)
        checkpoints = arm["checkpoints"] if arm is not None else {}
        if arm is not None:
            require(
                arm["training_root"] == str(TRAIN_ROOTS[variant])
                and arm["evaluation_root"] == str(EVAL_ROOTS[variant]),
                "wrong experiment roots",
            )
            require(set(checkpoints) <= {str(step) for step in STEPS}, "unauthorized checkpoint")
            diag[variant] = diagnostics(arm, variant)
        else:
            diag[variant] = {"scalars": [], "positions": []}
        validated = {}
        for step in STEPS:
            if str(step) not in checkpoints:
                missing.append(f"{variant}:{step}")
                continue
            cp = checkpoints[str(step)]
            folder = EVAL_ROOTS[variant] / f"evaluations/{variant}_step{step}"
            model = str(EVAL_ROOTS[variant] / f"merged/{variant}_step{step}")
            check_source(cp["source"], folder / "acceptance.json")
            card = cp["eval_card"]
            check_source(card["source"], folder / "eval_card.json")
            expected = {
                **EVAL_CONTRACT,
                "variant": variant,
                "checkpoint_step": step,
                "model": model,
            }
            # The accepted 645a2f3 Block3 controller predates the variant card field.
            if variant == "block3_mean" and "variant" not in card:
                del expected["variant"]
            for key, value in expected.items():
                require(
                    card.get(key) == value and (not isinstance(value, bool) or card[key] is value),
                    f"frozen eval contract mismatch: {variant}/{step}/{key}",
                )
            result = cp["result"]
            require(
                result.get("passed") is True and result.get("completion_protocol_verified") is True,
                "unaccepted completion result",
            )
            require(
                result.get("model") == model and result.get("grader_sha256") == GRADER_SHA,
                "accepted model/grader mismatch",
            )
            archive = result["rollout_archive"]
            require(
                archive.get("passed") is True
                and archive.get("num_examples") == 643
                and archive.get("num_rollouts") == 5144,
                "incomplete accepted archive",
            )
            require(set(result["per_task"]) == set(TASKS), "incomplete accepted benchmark set")
            for task in TASKS:
                require(
                    result["sha256"].get(str(DATA / f"{task}.jsonl"))
                    == bundle["data_sha256"][task],
                    f"benchmark data identity mismatch: {task}",
                )
                check_task(result["per_task"][task], task)
            validated[step] = result["per_task"]
            if "questions" in cp:
                questions[variant, step] = question_groups(cp["questions"], result, variant, step)
        for task in TASKS:
            for step in STEPS:
                present = step in validated
                values = validated[step][task] if present else {}
                rows.append(
                    {
                        "task": task,
                        "variant": variant,
                        "step": step,
                        "status": "accepted" if present else "missing",
                        **{metric: values.get(metric) for metric in METRICS},
                    }
                )
    require(allow_partial or not missing, "missing checkpoints: " + ", ".join(missing))
    require(any(r["status"] == "accepted" for r in rows), "no accepted checkpoints")
    sizes = {p["bin_size"] for value in diag.values() for p in value["positions"]}
    require(len(sizes) <= 1, "paired position snapshots must use the same bin size")
    left_positions = {p["step"]: p for p in diag[VARIANTS[0]]["positions"]}
    for item in diag[VARIANTS[1]]["positions"]:
        if item["step"] in left_positions:
            require(
                item["metadata"]["prompt_batch_sha256"]
                == left_positions[item["step"]]["metadata"]["prompt_batch_sha256"],
                "position snapshots use unpaired training prompts",
            )
    intervals, missing_scores = [], []
    for step in STEPS:
        if not all((variant, step) in questions for variant in VARIANTS):
            missing_scores.append(step)
            continue
        for task in TASKS:
            stable_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{task}:{step}".encode()).digest()[:8], "little"
            )
            # Keep a stable independent RNG stream for each benchmark/checkpoint.
            result = paired_bootstrap(
                questions[VARIANTS[0], step][task],
                questions[VARIANTS[1], step][task],
                replicates=replicates,
                seed=stable_seed % (2**63),
            )
            intervals.append({"task": task, "step": step, "num_questions": TASKS[task], **result})
    return {
        "schema_version": 1,
        "complete": not missing,
        "missing_checkpoints": missing,
        "rows": rows,
        "bootstrap": intervals,
        "bootstrap_unavailable_steps": missing_scores,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "uncertainty_scope": UNCERTAINTY_SCOPE,
        "definitions": DEFINITIONS,
        "diagnostics": diag,
    }


def tex_table(report, language):
    zh = language == "zh"
    status = "COMPLETE" if report["complete"] else "PARTIAL"
    headers = "Benchmark & Method & Step & Avg@8 & Pass@8 & No box & Trunc."
    caption = "Qwen3-4B completion, seed21, " + status + ". All scores in percent; NA = unavailable."
    note = "No box: no literal \\texttt{\\textbackslash boxed}; truncation: engine length stop."
    if zh:
        headers = (
            "\u57fa\u51c6 & \u65b9\u6cd5 & \u6b65\u6570 & Avg@8 & Pass@8 & \u7f3abox & \u622a\u65ad"
        )
        caption = (
            "Qwen3-4B completion\uff0cseed21\uff0c"
            + status
            + "\u3002\u5206\u6570\u5355\u4f4d\u4e3a\u767e\u5206\u6bd4\uff0c"
            "NA\u8868\u793a\u7f3a\u5931\u3002"
        )
        note = (
            "\u7f3abox\u4ec5\u6307\u65e0\u5b57\u9762\u6807\u8bb0 "
            "\\texttt{\\textbackslash boxed}\uff1b"
            "\u622a\u65ad\u6307\u5f15\u64ce length \u7ec8\u6b62\u3002"
        )
    lines = [
        "% Generated; do not edit. Requires booktabs; Chinese requires CJK support.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{" + caption + "}",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        headers + r" \\",
        "\\midrule",
    ]
    for task in TASKS:
        for row in (r for r in report["rows"] if r["task"] == task):
            values = ["NA" if row[k] is None else f"{100 * row[k]:.2f}" for k in METRICS]
            lines.append(
                " & ".join([TASK_LABELS[task], LABELS[row["variant"]], str(row["step"]), *values])
                + r" \\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    return "\n".join(
        [*lines, "\\end{tabular}", "\\par\\smallskip\\footnotesize " + note, "\\end{table*}", ""]
    )


def step200_table(report, language):
    """Compact main-text table; accepted point estimates do not require question scores."""
    status = "COMPLETE" if report["complete"] else "PARTIAL"
    caption = f"Qwen3-4B paired Step200 evaluation (seed21; queue {status})."
    note = (
        "Scores are percentages; $\\Delta$ is Block3 minus Token in percentage points. "
        "Brackets: pointwise 95\\% paired-question bootstrap within each benchmark, "
        "conditional on one training pair, not a multi-training-seed CI. NA: unavailable."
    )
    benchmark = "Benchmark"
    if language == "zh":
        caption = f"Qwen3-4B Step200 \u914d\u5bf9\u8bc4\u6d4b (seed21; \u961f\u5217 {status})\u3002"
        benchmark = "\u57fa\u51c6"
        note = (
            "\u5206\u6570\u4e3a\u767e\u5206\u6bd4\uff1b$\\Delta$\u4e3aBlock3\u51cfToken\u7684\u767e\u5206\u70b9\u5dee\u3002"
            "\u65b9\u62ec\u53f7\u4e3a\u5404\u57fa\u51c6\u5185\u6210\u5bf9\u95ee\u9898bootstrap\u7684\u9010\u70b995\\%\u533a\u95f4\uff0c"
            "\u975e\u591a\u8bad\u7ec3\u79cd\u5b50CI\uff1bNA\u8868\u793a\u7f3a\u5931\u3002"
        )
    rows = {(row["task"], row["variant"]): row for row in report["rows"] if row["step"] == 200}
    intervals = {row["task"]: row for row in report["bootstrap"] if row["step"] == 200}
    lines = [
        "% Generated from accepted results; requires booktabs.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{" + caption + "}",
        "\\label{tab:qwen4endpoint}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        r" & \multicolumn{3}{c}{Avg@8} & \multicolumn{3}{c}{Pass@8} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        benchmark
        + r" & Block3 & Token & $\Delta$ [95\% CI] & Block3 & Token & $\Delta$ [95\% CI] \\",
        "\\midrule",
    ]
    for task in TASKS:
        cells = [TASK_LABELS[task]]
        for metric in METRICS[:2]:
            left, right = (rows[task, variant][metric] for variant in VARIANTS)
            cells.extend("NA" if value is None else f"{100 * value:.2f}" for value in (left, right))
            if left is None or right is None:
                cells.append("NA")
            else:
                bounds = intervals.get(task, {}).get(metric)
                interval = (
                    f"[{bounds['lower_95_pp']:.2f}, {bounds['upper_95_pp']:.2f}]"
                    if bounds
                    else "[NA]"
                )
                cells.append(f"{100 * (left - right):+.2f} {interval}")
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(
        [
            *lines,
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\smallskip\\footnotesize " + note,
            "\\end{table*}",
            "",
        ]
    )


def compact_question_scores(path, task, expected_sha):
    """One pass, bounded memory: no generation text or rollout arrays leave this reader."""
    groups, identities, digest = {}, set(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row["example_id"])
            rollout = integer(row["rollout_id"], "rollout_id", 0, 7)
            require(
                type(row["seed"]) is int and row["seed"] == 21 + rollout,
                "graded rollout seed mismatch",
            )
            require((identity, rollout) not in identities, "duplicate graded rollout")
            identities.add((identity, rollout))
            require(type(row["correct"]) is bool, "graded correctness must be Boolean")
            require(row["finish_reason"] in ("length", "stop"), "unknown engine finish reason")
            require(isinstance(row["response"], str), "missing graded response")
            summary = groups.setdefault(
                identity,
                {
                    "task": task,
                    "id": identity,
                    "correct_count": 0,
                    "missing_box_count": 0,
                    "length_stop_count": 0,
                },
            )
            summary["correct_count"] += row["correct"]
            summary["missing_box_count"] += "\\boxed" not in row["response"]
            summary["length_stop_count"] += row["finish_reason"] == "length"
    require(digest.hexdigest() == expected_sha, f"accepted graded hash mismatch: {path}")
    require(
        len(groups) == TASKS[task] and len(identities) == TASKS[task] * 8,
        "incomplete question/rollout identities",
    )
    return [groups[key] for key in sorted(groups)]


def export_bundle(
    artifact_root=ROOT,
    *,
    include_diagnostics=True,
    position_steps=TRAINING_STEPS,
    question_steps=(200,),
):
    """Read a local artifact tree (or mirror) and return a compact validated snapshot."""
    artifact_root = Path(artifact_root)
    require(set(question_steps) <= set(STEPS), "unauthorized question step")
    require(len(position_steps) == len(set(position_steps)), "duplicate position step")
    for step in position_steps:
        integer(step, "position step", 1, 200)

    def local(original):
        return artifact_root / original.relative_to(ROOT)

    def read(original):
        raw = local(original).read_bytes()
        return json.loads(raw, object_pairs_hook=unique_json_object), {
            "path": str(original),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    bundle = {"schema_version": 1, "data_sha256": {}, "arms": {}}
    for variant in VARIANTS:
        root = EVAL_ROOTS[variant]
        arm = {
            "training_root": str(TRAIN_ROOTS[variant]),
            "evaluation_root": str(root),
            "checkpoints": {},
        }
        bundle["arms"][variant] = arm
        manifest_path = root / "queue_manifest.json"
        if local(manifest_path).exists():
            manifest, evidence = read(manifest_path)
            require(
                manifest["runtime_commit"] == TRAIN_COMMIT
                and manifest["prompt_protocol"] == EVAL_CONTRACT["prompt_protocol"],
                "export runtime/prompt manifest mismatch",
            )
            arm["queue_manifest"] = {"source": evidence, "value": manifest}
        for step in STEPS:
            folder = root / f"evaluations/{variant}_step{step}"
            if not local(folder / "acceptance.json").exists():
                continue
            require(
                local(folder / "exit_code.txt").read_text().strip() == "0",
                "accepted evaluation did not exit successfully",
            )
            result, source = read(folder / "acceptance.json")
            card, card_source = read(folder / "eval_card.json")
            card["source"] = card_source
            hashes = {task: result["sha256"][str(DATA / f"{task}.jsonl")] for task in TASKS}
            require(
                not bundle["data_sha256"] or hashes == bundle["data_sha256"],
                "exported benchmark data hashes differ",
            )
            bundle["data_sha256"] = hashes
            selected_hashes = {str(DATA / f"{task}.jsonl"): hashes[task] for task in TASKS}
            compact = {
                key: result[key]
                for key in (
                    "passed",
                    "model",
                    "per_task",
                    "grader_sha256",
                    "completion_protocol_verified",
                )
            }
            compact["rollout_archive"] = {
                key: result["rollout_archive"][key]
                for key in ("passed", "num_examples", "num_rollouts")
            }
            compact["sha256"] = selected_hashes
            checkpoint = {"source": source, "result": compact, "eval_card": card}
            if step in question_steps:
                questions = {"sources": [], "rows": []}
                for task in TASKS:
                    path = folder / f"outputs/{task}_graded.jsonl"
                    expected_sha = result["sha256"][str(path)]
                    questions["rows"].extend(
                        compact_question_scores(local(path), task, expected_sha)
                    )
                    questions["sources"].append({"path": str(path), "sha256": expected_sha})
                    selected_hashes[str(path)] = expected_sha
                checkpoint["questions"] = questions
            arm["checkpoints"][str(step)] = checkpoint
        if include_diagnostics:
            path = TRAIN_ROOTS[variant] / "diagnostics/scalars.jsonl"
            digest, records = hashlib.sha256(), []
            with local(path).open("rb") as stream:
                for line in stream:
                    digest.update(line)
                    if line.strip():
                        row = json.loads(line)
                        records.append(
                            {key: row[key] for key in ("step", *SCALAR_METRICS) if key in row}
                        )
            arm["scalars"] = {
                "source": {"path": str(path), "sha256": digest.hexdigest()},
                "records": records,
            }
            arm["positions"] = []
            for step in position_steps:
                path = TRAIN_ROOTS[variant] / f"diagnostics/step_{step:05d}.npz"
                item = compact_npz(local(path))
                item["source"]["path"] = str(path)
                arm["positions"].append(item)
    # This validates the snapshot, including question totals, without spending on final CIs.
    analyze(bundle, allow_partial=True, bootstrap_replicates=1)
    return bundle


def merge_diagnostics(evaluation_path, cache_path, *, allow_partial_diagnostics=False):
    """Reuse a validated diagnostic cache without changing either source file."""
    evaluation_path, cache_path = Path(evaluation_path), Path(cache_path)
    evaluation_raw, cache_raw = evaluation_path.read_bytes(), cache_path.read_bytes()
    evaluation = json.loads(evaluation_raw, object_pairs_hook=unique_json_object)
    cache = json.loads(cache_raw, object_pairs_hook=unique_json_object)
    for value in (evaluation, cache):
        analyze(value, allow_partial=True, bootstrap_replicates=1)
        require(set(value["arms"]) == set(VARIANTS), "diagnostic merge requires both arm records")
    require(evaluation["data_sha256"] == cache["data_sha256"], "diagnostic cache data mismatch")
    for variant in VARIANTS:
        target, source = evaluation["arms"][variant], cache["arms"][variant]
        require("positions" in source and "scalars" in source, "missing diagnostic cache fields")
        if not allow_partial_diagnostics:
            require(
                {item["step"] for item in source["positions"]} == set(TRAINING_STEPS)
                and {row["step"] for row in source["scalars"]["records"]} == set(TRAINING_STEPS),
                f"incomplete diagnostic coverage for {variant}; expected steps 1..200",
            )
        for key in ("scalars", "positions"):
            require(
                key not in target or target[key] == source[key],
                f"conflicting existing diagnostics: {variant}/{key}",
            )
            target[key] = source[key]
    evaluation["diagnostics_reuse"] = {
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": hashlib.sha256(evaluation_raw).hexdigest(),
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": hashlib.sha256(cache_raw).hexdigest(),
        "merger_sha256": sha256(Path(__file__)),
        "allow_partial_diagnostics": allow_partial_diagnostics,
    }
    analyze(evaluation, allow_partial=True, bootstrap_replicates=1)
    return evaluation


def csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"


def unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def check_destination(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "output paths must not contain traversal")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "output symlink refused")
    require(not path.is_relative_to(ROOT), "never write to experiment artifact roots")
    require(not path.exists() or path.is_dir(), "output destination must be a directory")
    return path


def figure_stems(report):
    stems = ["qwen4_accuracy", "qwen4_pass", "qwen4_scores"]
    if any(value["scalars"] for value in report["diagnostics"].values()):
        stems.append("qwen4_training")
    names = {
        name
        for value in report["diagnostics"].values()
        for item in value["positions"]
        for name in item["metrics"]
    }
    if "student_entropy" in names:
        stems.append("qwen4_coverage")
    return stems + [f"qwen4_position_{name}" for name in sorted(names)]


def plot_figures(report, output, *, min_count=8, only=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = set(figure_stems(report) if only is None else only)
    require(selected and selected <= set(figure_stems(report)), "unavailable figure selection")
    status = "" if report["complete"] else " (PARTIAL)"
    style = {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "figure.titlesize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
    }

    def save(fig, stem):
        try:
            for suffix in ("pdf", "png"):
                metadata = (
                    {"CreationDate": None, "ModDate": None}
                    if suffix == "pdf"
                    else {"Software": "qwen4-paper-assets"}
                )
                with (output / f"{stem}.{suffix}").open("xb") as handle:
                    fig.savefig(handle, format=suffix, dpi=200, metadata=metadata)
        finally:
            plt.close(fig)

    def score_axis(ax, task, metric):
        observed = []
        for variant in VARIANTS:
            by_step = {
                r["step"]: r[metric]
                for r in report["rows"]
                if r["task"] == task and r["variant"] == variant
            }
            values = [np.nan if by_step[s] is None else 100 * by_step[s] for s in sorted(STEPS)]
            observed.extend(value for value in values if np.isfinite(value))
            ax.plot(
                sorted(STEPS),
                values,
                "o-" if variant == VARIANTS[0] else "s--",
                color=COLORS[variant],
                label=LABELS[variant],
                markersize=3,
            )
        # Both arms share a task-specific scale; avoid hiding small checkpoint changes.
        limits = (0, 100)
        if observed:
            low, high = min(observed), max(observed)
            width = min(100, max(10, 1.3 * (high - low)))
            bottom = max(0, min((low + high - width) / 2, 100 - width))
            limits = (bottom, bottom + width)
        ax.set(xticks=sorted(STEPS), ylim=limits, xlabel="Checkpoint step")
        ax.grid(alpha=0.2)

    with plt.rc_context(style):
        for stem, metric, label in (
            ("qwen4_accuracy", "avg_at_8", "Avg@8 (%)"),
            ("qwen4_pass", "pass_at_8", "Pass@8 (%)"),
        ):
            if stem not in selected:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.6), constrained_layout=True)
            fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))
            for task, ax in zip(TASKS, axes.flat):
                score_axis(ax, task, metric)
                ax.set(title=TASK_LABELS[task], ylabel=label)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=2,
                frameon=False,
                title="Qwen3-4B " + label.removesuffix(" (%)") + status,
            )
            save(fig, stem)

        if "qwen4_scores" in selected:
            fig, axes = plt.subplots(4, 4, figsize=(10, 8), constrained_layout=True)
            for i, task in enumerate(TASKS):
                for j, (metric, label) in enumerate(
                    zip(
                        METRICS,
                        ("Avg@8 (%)", "Pass@8 (%)", "Missing-box marker (%)", "Length stop (%)"),
                    )
                ):
                    score_axis(axes[i, j], task, metric)
                    axes[i, j].set_title(TASK_LABELS[task] + ": " + label)
            axes[0, 0].legend(fontsize=6, frameon=False)
            fig.suptitle("Qwen3-4B paired completion evaluation" + status)
            save(fig, "qwen4_scores")

        if "qwen4_training" in selected:
            fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.5), constrained_layout=True)
            complete = all(
                {
                    row["step"]
                    for row in report["diagnostics"][variant]["scalars"]
                    if all(metric in row for metric in SCALAR_METRICS[:4])
                }
                == set(TRAINING_STEPS)
                for variant in VARIANTS
            )
            for variant in VARIANTS:
                rows = {r["step"]: r for r in report["diagnostics"][variant]["scalars"]}
                for metric, ax_index in zip(SCALAR_METRICS, (0, 0, 1, 2)):
                    values = [rows.get(step, {}).get(metric, np.nan) for step in range(1, 201)]
                    teacher = metric == "diagnostics/teacher_entropy"
                    label = (
                        LABELS[variant] + (" teacher" if teacher else " student")
                        if ax_index == 0
                        else LABELS[variant]
                    )
                    axes[ax_index].plot(
                        range(1, 201),
                        values,
                        "--" if teacher else "-",
                        color=COLORS[variant],
                        label=label,
                        marker=".",
                        markersize=2,
                    )
            for ax, title in zip(
                axes,
                (
                    "Entropy on student prefixes (nats)",
                    "Parameter gradient norm (pre-clip)",
                    "Training length-at-cap fraction",
                ),
            ):
                ax.set(title=title, xlabel="Training step", xlim=(1, 200))
                ax.legend(fontsize=8, frameon=False, ncol=2)
                ax.grid(alpha=0.2)
            axes[2].set_ylim(-0.02, 1.02)
            fig.suptitle("Qwen3-4B training diagnostics" + ("" if complete else " (PARTIAL)"))
            save(fig, "qwen4_training")

        for stem in sorted(selected):
            coverage = stem == "qwen4_coverage"
            if not coverage and not stem.startswith("qwen4_position_"):
                continue
            metric = "student_entropy" if coverage else stem.removeprefix("qwen4_position_")
            all_items = [
                item for value in report["diagnostics"].values() for item in value["positions"]
            ]
            size = all_items[0]["bin_size"]
            complete = all(
                {
                    item["step"]
                    for item in report["diagnostics"][variant]["positions"]
                    if metric in item["metrics"]
                }
                == set(TRAINING_STEPS)
                for variant in VARIANTS
            )
            matrices = {}
            for variant in VARIANTS:
                matrix = np.full((200, 16384 // size), np.nan)
                for item in report["diagnostics"][variant]["positions"]:
                    if metric in item["metrics"]:
                        if coverage:
                            counts = np.asarray(item["metrics"][metric]["max_position_count"])
                            matrix[item["step"] - 1] = np.where(counts > 0, counts, np.nan)
                        else:
                            matrix[item["step"] - 1] = position_means(
                                item, metric, min_count=min_count
                            )
                matrices[variant] = matrix
            finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()])
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            if coverage:
                lo, hi = 0, 32
            elif metric in SIGNED_METRICS:
                hi = max(abs(lo), abs(hi), 1e-8)
                lo = -hi
            elif metric in RATE_POSITIONS:
                lo, hi = 0, 1
            else:
                lo, hi = 0, max(hi, 1e-8)
            cmap = plt.get_cmap("RdBu_r" if metric in SIGNED_METRICS else "viridis").with_extremes(
                bad="#dddddd"
            )
            fig, axes = plt.subplots(2, 1, figsize=(5.5, 5), constrained_layout=True)
            for ax, variant in zip(axes, VARIANTS):
                im = ax.imshow(
                    matrices[variant],
                    aspect="auto",
                    origin="lower",
                    interpolation="nearest",
                    extent=(0, 16384, 0.5, 200.5),
                    cmap=cmap,
                    vmin=lo,
                    vmax=hi,
                )
                ax.set(
                    title=LABELS[variant],
                    xlabel="Absolute response-token position",
                    ylabel="Training step",
                )
            if coverage:
                fig.colorbar(
                    im,
                    ax=list(axes),
                    label="Max per-position valid count",
                    ticks=[0, 8, 16, 24, 32],
                    shrink=0.8,
                )
                title = (
                    "Qwen3-4B entropy-position coverage (count)\n"
                    f"Maximum per-position valid count in each {size}-token bin"
                )
            else:
                fig.colorbar(im, ax=list(axes), label=POSITION_METRICS[metric], shrink=0.8)
                title = f"Qwen3-4B; max per-position valid count >= {min_count}"
            fig.suptitle(title + ("" if complete else " (PARTIAL)"))
            save(fig, stem)


def render_figures(input_path, figures_dir, *, only, allow_partial=False, min_count=8):
    """Validate local evidence and add plots without rebuilding tables or CIs."""
    integer(min_count, "min_count", 1, 32)
    input_path = Path(input_path)
    raw = input_path.read_bytes()
    bundle = json.loads(raw, object_pairs_hook=unique_json_object)
    report = analyze(bundle, allow_partial=allow_partial, bootstrap_replicates=1)
    require(only and len(only) == len(set(only)), "empty or duplicate figure selection")
    require(set(only) <= set(figure_stems(report)), "unavailable figure selection")
    output = check_destination(figures_dir)
    paths = [output / f"{stem}.{suffix}" for stem in only for suffix in ("pdf", "png")]
    manifest = output / "qwen4_figures_provenance.json"
    for path in [*paths, manifest]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}; use a fresh directory")
        require(path != input_path.absolute(), "output would replace input")
    import matplotlib

    output.mkdir(parents=True, exist_ok=True)
    plot_figures(report, output, min_count=min_count, only=only)
    provenance = {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "builder_sha256": sha256(Path(__file__)),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "options": {"only": only, "allow_partial": allow_partial, "min_count": min_count},
        "complete": report["complete"],
        "missing_checkpoints": report["missing_checkpoints"],
        "output_sha256": {path.name: sha256(path) for path in paths},
    }
    with manifest.open("x", encoding="utf-8") as handle:
        handle.write(json_text(provenance))
    return report


def build(
    input_path,
    generated_dir,
    figures_dir,
    *,
    allow_partial=False,
    make_figures=True,
    bootstrap_replicates=10000,
    bootstrap_seed=20260921,
    min_count=8,
):
    integer(min_count, "min_count", 1, 32)
    input_path = Path(input_path)
    raw = input_path.read_bytes()
    bundle = json.loads(raw, object_pairs_hook=unique_json_object)
    report = analyze(
        bundle,
        allow_partial=allow_partial,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    generated, figures = check_destination(generated_dir), check_destination(figures_dir)
    require(
        generated != figures
        and not generated.is_relative_to(figures)
        and not figures.is_relative_to(generated),
        "output directories must be disjoint",
    )
    fields = ["task", "variant", "step", "status", *METRICS]
    ci_rows = [
        {
            "task": r["task"],
            "step": r["step"],
            "num_questions": r["num_questions"],
            "metric": metric,
            **r[metric],
        }
        for r in report["bootstrap"]
        for metric in METRICS[:2]
    ]
    caption_en = (
        "Shared limits across arms; gray heatmap cells are missing/low-coverage, not zero. "
        + UNCERTAINTY_SCOPE
    )
    caption_zh = (
        "\u4e24\u81c2\u5171\u4eab\u8272\u6807\uff1b\u7070\u8272\u8868\u793a\u7f3a\u5931\u6216\u4f4e\u8986\u76d6\uff0c\u975e\u96f6\u503c\u3002"
        "\u533a\u95f4\u4e3a\u5404\u57fa\u51c6\u5185\u6210\u5bf9\u95ee\u9898\u81ea\u52a9\u91cd\u91c7\u6837\u768495%\u767e\u5206\u4f4d\u533a\u95f4\uff0c"
        "\u5dee\u503c\u4e3aBlock3\u51cfToken\uff1b\u4ec5\u6761\u4ef6\u4e8eseed21\u8bad\u7ec3\u5bf9\uff0c\u975e\u591a\u8bad\u7ec3\u79cd\u5b50\u533a\u95f4\u3002"
    )
    texts = {
        "qwen4_results.csv": csv_text(report["rows"], fields),
        "qwen4_results_en.tex": tex_table(report, "en"),
        "qwen4_results_zh.tex": tex_table(report, "zh"),
        "qwen4_step200_en.tex": step200_table(report, "en"),
        "qwen4_step200_zh.tex": step200_table(report, "zh"),
        "qwen4_paired_ci.csv": csv_text(
            ci_rows,
            ["task", "step", "num_questions", "metric", "delta_pp", "lower_95_pp", "upper_95_pp"],
        ),
        "qwen4_summary.json": json_text(report),
        "qwen4_captions.json": json_text(
            {
                "en": caption_en,
                "zh": caption_zh,
                "complete": report["complete"],
                "definitions": DEFINITIONS,
            }
        ),
    }
    destinations = [generated / name for name in [*texts, "qwen4_provenance.json"]]
    if make_figures:
        destinations += [
            figures / f"{stem}.{suffix}"
            for stem in figure_stems(report)
            for suffix in ("pdf", "png")
        ]
        import matplotlib  # Fail before creating outputs when the optional dependency is absent.
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}; use fresh directories")
        require(path != input_path.absolute(), "output would replace input")
    generated.mkdir(parents=True, exist_ok=True)
    if make_figures:
        figures.mkdir(parents=True, exist_ok=True)
        plot_figures(report, figures, min_count=min_count)
    for name, value in texts.items():
        with (generated / name).open("x", encoding="utf-8", newline="") as handle:
            handle.write(value)
    provenance = {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "builder_sha256": sha256(Path(__file__)),
        "schema_version": 1,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__ if make_figures else None,
        },
        "options": {
            "allow_partial": allow_partial,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "min_count": min_count,
            "make_figures": make_figures,
        },
        "complete": report["complete"],
        "missing_checkpoints": report["missing_checkpoints"],
        "output_sha256": {
            path.name: sha256(path) for path in destinations if path.name != "qwen4_provenance.json"
        },
        "source_evidence": bundle,
        "lineage_scope": DEFINITIONS["lineage"],
    }
    with (generated / "qwen4_provenance.json").open("x", encoding="utf-8") as handle:
        handle.write(json_text(provenance))
    return report


def schema():
    return {
        "schema_version": 1,
        "documentation": __doc__,
        "checkpoint_order": STEPS,
        "default_diagnostic_steps": TRAINING_STEPS,
        "tasks": TASKS,
        "eval_contract": EVAL_CONTRACT,
        "scalar_metrics": SCALAR_METRICS,
        "position_metrics": POSITION_METRICS,
        "definitions": DEFINITIONS,
        "roots": {
            v: {"training_root": str(TRAIN_ROOTS[v]), "evaluation_root": str(EVAL_ROOTS[v])}
            for v in VARIANTS
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="Print schema documentation; contains no fabricated results")
    compact = commands.add_parser(
        "compact-npz", help="Print compact, traceable position statistics"
    )
    compact.add_argument("input", type=Path)
    compact.add_argument("--bin-size", type=int, default=128)
    export = commands.add_parser(
        "export", help="Read-only artifact snapshot to compact JSON stdout"
    )
    export.add_argument("--artifact-root", type=Path, default=ROOT)
    export.add_argument("--no-diagnostics", action="store_true")
    export.add_argument(
        "--position-steps",
        type=int,
        nargs="*",
        default=list(TRAINING_STEPS),
        help="Default: all training steps 1..200 for each arm",
    )
    export.add_argument("--question-steps", type=int, nargs="*", default=[200])
    merge = commands.add_parser(
        "merge-diagnostics", help="Merge cached diagnostics into fresh local evaluation JSON stdout"
    )
    merge.add_argument("--input", type=Path, required=True)
    merge.add_argument("--diagnostics-from", type=Path, required=True)
    merge.add_argument("--allow-partial-diagnostics", action="store_true")
    command = commands.add_parser(
        "build", help="Build paper assets exclusively from local evidence"
    )
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--generated-dir", type=Path, default=Path("paper/iclr2027/generated"))
    command.add_argument("--figures-dir", type=Path, default=Path("paper/iclr2027/figures"))
    command.add_argument("--allow-partial", action="store_true")
    command.add_argument("--no-figures", action="store_true")
    command.add_argument("--bootstrap-replicates", type=int, default=10000)
    command.add_argument("--bootstrap-seed", type=int, default=20260921)
    command.add_argument("--min-count", type=int, default=8)
    figures = commands.add_parser("figures", help="Add selected plots without rewriting tables")
    figures.add_argument("--input", type=Path, required=True)
    figures.add_argument("--figures-dir", type=Path, required=True)
    figures.add_argument("--only", nargs="+", required=True)
    figures.add_argument("--allow-partial", action="store_true")
    figures.add_argument("--min-count", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            print(json_text(schema()), end="")
        elif args.command == "compact-npz":
            print(json_text(compact_npz(args.input, bin_size=args.bin_size)), end="")
        elif args.command == "export":
            value = export_bundle(
                args.artifact_root,
                include_diagnostics=not args.no_diagnostics,
                position_steps=args.position_steps,
                question_steps=args.question_steps,
            )
            print(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
        elif args.command == "merge-diagnostics":
            value = merge_diagnostics(
                args.input,
                args.diagnostics_from,
                allow_partial_diagnostics=args.allow_partial_diagnostics,
            )
            print(json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")))
        else:
            if args.command == "figures":
                report = render_figures(
                    args.input,
                    args.figures_dir,
                    only=args.only,
                    allow_partial=args.allow_partial,
                    min_count=args.min_count,
                )
            else:
                report = build(
                    args.input,
                    args.generated_dir,
                    args.figures_dir,
                    allow_partial=args.allow_partial,
                    make_figures=not args.no_figures,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.bootstrap_seed,
                    min_count=args.min_count,
                )
            print(
                json_text(
                    {
                        "complete": report["complete"],
                        "missing_checkpoints": report["missing_checkpoints"],
                    }
                ),
                end="",
            )
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

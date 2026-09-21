"""Synthetic counts only; no paper artifacts are created by these tests."""

import copy
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import test_qwen4_paper_assets as fixtures

from scripts import build_qwen4_paper_assets as assets

MATH_SHA = "cc164b47b60771eeda257bcb9a2068940cc0706bb8bb79b6635af4c1d4534501"


def module():
    assert importlib.util.find_spec("scripts.build_overlap_sensitivity") is not None
    return importlib.import_module("scripts.build_overlap_sensitivity")


def ids_hash(ids):
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture
def evidence():
    bundle = fixtures.bundle.__wrapped__(assets)
    bundle["data_sha256"]["math500"] = MATH_SHA
    for variant, arm in bundle["arms"].items():
        for step, cp in arm["checkpoints"].items():
            cp["result"]["sha256"][str(assets.DATA / "math500.jsonl")] = MATH_SHA
            rows = fixtures.add_questions(assets, bundle, variant, int(step))
            for row in rows:
                if row["task"] == "math500":
                    row["id"] = f"math500/test/{row['id']}"
    # Deliberately different from the real audit: exclusions must come from the input.
    ids = [f"math500/test/{i}" for i in (7, 11, 22, 35)]
    views = {}
    for name, selected in (("exact", ids[:2]), ("whitespace_casefold", ids)):
        matches = [
            {
                "evaluation_id": identity,
                "eval_question_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                "comparison_key_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                "also_exact_match": identity in ids[:2],
            }
            for identity in selected
        ]
        views[name] = {
            "heldout_rows": 500,
            "heldout_unique_comparison_keys": 500,
            "overlap_heldout_rows": len(selected),
            "overlap_unique_comparison_keys": len(selected),
            "matches": matches,
            "overlap_heldout_ids_sha256": ids_hash(selected),
        }
    audit = {
        "schema_version": 1,
        "mode": "read_only_cpu_exact_string_set_membership",
        "train": {"path": str(assets.DATA.parent / "train.parquet")},
        "definitions": {
            "whitespace_casefold": "For both sides: ' '.join(question.split()).casefold()"
        },
        "per_benchmark": {
            "math500": {
                "jsonl": {"path": str(assets.DATA / "math500.jsonl"), "sha256": MATH_SHA},
                "heldout_question_field": "problem",
                "views": views,
                "normalized_only_additional_heldout_ids": ids[2:],
                "normalized_only_additional_heldout_rows": 2,
            }
        },
    }
    return bundle, audit


def test_module_available():
    module()


def test_subset_from_audit_ids_is_paired_and_preserves_official_inputs(evidence):
    m = module()
    bundle, audit = evidence
    for variant, arm in bundle["arms"].items():
        for cp in arm["checkpoints"].values():
            rows = [r for r in cp["questions"]["rows"] if r["task"] == "math500"]
            for row in rows:
                row["correct_count"] = 8 if variant == "block3_mean" else 0
                if row["id"] in {
                    "math500/test/7",
                    "math500/test/11",
                    "math500/test/22",
                    "math500/test/35",
                }:
                    row["correct_count"] = 0 if variant == "block3_mean" else 8
            cp["result"]["per_task"]["math500"]["avg_at_8"] = (
                sum(r["correct_count"] for r in rows) / 4000
            )
            cp["result"]["per_task"]["math500"]["pass_at_8"] = (
                sum(r["correct_count"] > 0 for r in rows) / 500
            )
    before = copy.deepcopy(evidence)
    report = m.analyze_sensitivity(bundle, audit, primary_ci=True, bootstrap_replicates=16)
    assert report["complete"] and report["subset_questions"] == 496
    assert set(report["excluded_ids"]) == {
        r["evaluation_id"]
        for r in audit["per_benchmark"]["math500"]["views"]["whitespace_casefold"]["matches"]
    }
    for row in report["rows"]:
        assert row["arms"]["block3_mean"]["avg_at_8"] == 1
        assert row["arms"]["token_opd"]["pass_at_8"] == 0
        assert row["delta_pp"] == {"avg_at_8": 100, "pass_at_8": 100}
        assert bool(row["paired_ci"]) == (row["step"] == 200)
        if row["paired_ci"]:
            assert row["paired_ci"]["avg_at_8"]["lower_95_pp"] == 100
    assert evidence == before
    assert "post-hoc" in report["scope"].lower() and "not a new benchmark" in report["scope"]
    assert "training-run" in report["uncertainty_scope"]


def test_partial_requires_explicit_flag_and_never_infers_unpaired_scores(evidence):
    m = module()
    bundle, audit = evidence
    for step in (150, 100, 50):
        del bundle["arms"]["token_opd"]["checkpoints"][str(step)]
        del bundle["arms"]["block3_mean"]["checkpoints"][str(step)]["questions"]
    with pytest.raises(ValueError, match="missing checkpoints"):
        m.analyze_sensitivity(bundle, audit)
    report = m.analyze_sensitivity(bundle, audit, allow_partial=True)
    assert not report["complete"] and report["unavailable_steps"] == [150, 100, 50]
    for row in report["rows"][1:]:
        assert row["arms"] is None and row["delta_pp"] is None
    assert report["rows"][0]["paired_ci"] is None
    del bundle["arms"]["token_opd"]["checkpoints"]["200"]["questions"]
    with pytest.raises(ValueError, match="no paired"):
        m.analyze_sensitivity(bundle, audit, allow_partial=True)


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "both_hashes",
        "path",
        "unknown_id",
        "duplicate",
        "count",
        "heldout_count",
        "id_hash",
        "exact_not_subset",
        "exact_flag",
        "additional_count",
        "additional_duplicate",
    ],
)
def test_bad_audit_fails_closed(evidence, problem):
    m = module()
    bundle, audit = evidence
    task = audit["per_benchmark"]["math500"]
    view = task["views"]["whitespace_casefold"]
    if problem in ("hash", "both_hashes"):
        task["jsonl"]["sha256"] = "e" * 64
        if problem == "both_hashes":
            bundle["data_sha256"]["math500"] = "e" * 64
            for arm in bundle["arms"].values():
                for cp in arm["checkpoints"].values():
                    cp["result"]["sha256"][str(assets.DATA / "math500.jsonl")] = "e" * 64
    elif problem == "path":
        task["jsonl"]["path"] = "/other/math500.jsonl"
    elif problem == "unknown_id":
        view["matches"][0]["evaluation_id"] = "math500/test/9000"
    elif problem == "duplicate":
        view["matches"][1] = copy.deepcopy(view["matches"][0])
    elif problem == "count":
        view["overlap_heldout_rows"] = 3
    elif problem == "heldout_count":
        view["heldout_rows"] = 499
    elif problem == "id_hash":
        view["overlap_heldout_ids_sha256"] = "f" * 64
    elif problem == "exact_not_subset":
        exact = task["views"]["exact"]
        exact["matches"][0]["evaluation_id"] = "math500/test/499"
        exact["overlap_heldout_ids_sha256"] = ids_hash(
            [r["evaluation_id"] for r in exact["matches"]]
        )
    elif problem == "exact_flag":
        view["matches"][0]["also_exact_match"] = False
    elif problem == "additional_count":
        task["normalized_only_additional_heldout_rows"] = 3
    else:
        task["normalized_only_additional_heldout_ids"] *= 2
    with pytest.raises(ValueError):
        m.analyze_sensitivity(bundle, audit, allow_partial=True)


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "unknown_id", "counts", "source_hash", "unaccepted"]
)
def test_bad_question_evidence_never_bypasses_shared_validation(evidence, problem):
    m = module()
    bundle, audit = evidence
    cp = bundle["arms"]["token_opd"]["checkpoints"]["200"]
    rows = cp["questions"]["rows"]
    if problem == "missing":
        rows.pop(0)
    elif problem == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif problem == "unknown_id":
        for variant in assets.VARIANTS:
            bundle["arms"][variant]["checkpoints"]["200"]["questions"]["rows"][0]["id"] = (
                "math500/test/9000"
            )
    elif problem == "counts":
        rows[0]["correct_count"] = 9
    elif problem == "source_hash":
        cp["questions"]["sources"][0]["sha256"] = "f" * 64
    else:
        cp["result"]["passed"] = False
    with pytest.raises(ValueError):
        m.analyze_sensitivity(bundle, audit, allow_partial=True)


def test_complete_checkpoints_still_require_question_counts_at_all_steps(evidence):
    m = module()
    bundle, audit = evidence
    del bundle["arms"]["token_opd"]["checkpoints"]["150"]["questions"]
    with pytest.raises(ValueError, match="question"):
        m.analyze_sensitivity(bundle, audit)


def test_outputs_are_small_bilingual_posthoc_exclusive_and_traceable(evidence, tmp_path):
    m = module()
    bundle, audit = evidence
    input_path, audit_path = tmp_path / "bundle.json", tmp_path / "audit.json"
    input_path.write_text(json.dumps(bundle))
    audit_path.write_text(json.dumps(audit))
    before = [p.read_bytes() for p in (input_path, audit_path)]
    output = tmp_path / "new"
    report = m.build(input_path, audit_path, output, primary_ci=True, bootstrap_replicates=16)
    assert {p.name for p in output.iterdir()} == {
        "qwen4_overlap_sensitivity.json",
        "qwen4_overlap_sensitivity_en.tex",
        "qwen4_overlap_sensitivity_zh.tex",
    }
    raw = (output / "qwen4_overlap_sensitivity.json").read_bytes()
    assert len(raw) < 12000
    assert report["provenance"]["audit_sha256"] == assets.sha256(audit_path)
    assert report["provenance"]["input_sha256"] == assets.sha256(input_path)
    en = (output / "qwen4_overlap_sensitivity_en.tex").read_text()
    zh = (output / "qwen4_overlap_sensitivity_zh.tex").read_text()
    assert "Post-hoc" in en and "not a new benchmark" in en
    assert "\u4e8b\u540e" in zh and "496" in zh
    assert "95\\%" in en and "primary" in en
    assert [p.read_bytes() for p in (input_path, audit_path)] == before
    with pytest.raises(FileExistsError):
        m.build(input_path, audit_path, output)


def test_invalid_input_and_symlink_write_nothing(evidence, tmp_path):
    m = module()
    bundle, audit = evidence
    input_path, audit_path = tmp_path / "bundle.json", tmp_path / "audit.json"
    input_path.write_text(json.dumps(bundle))
    audit_path.write_text(json.dumps(audit))
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        m.build(input_path, audit_path, alias)
    assert not list(target.iterdir())
    input_path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate"):
        m.build(input_path, audit_path, target)
    assert not list(target.iterdir())


def test_direct_cli_stdout_only_and_no_generated_files(evidence, tmp_path):
    module()
    bundle, audit = evidence
    paths = [tmp_path / "bundle.json", tmp_path / "audit.json"]
    for path, value in zip(paths, (bundle, audit)):
        path.write_text(json.dumps(value))
    script = Path(__file__).resolve().parents[1] / "scripts/build_overlap_sensitivity.py"
    result = subprocess.run(
        [sys.executable, "-B", str(script), "--input", str(paths[0]), "--audit", str(paths[1])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["subset_questions"] == 496
    assert set(tmp_path.iterdir()) == set(paths)

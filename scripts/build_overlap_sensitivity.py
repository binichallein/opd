#!/usr/bin/env python3
"""Offline, post-hoc Qwen4 MATH500 overlap sensitivity; never a replacement benchmark.

Inputs: a compact accepted Qwen4 bundle and internal/dataset_overlap_audit.json.
Exclusions come only from math500.views.whitespace_casefold.matches in that audit.
The frozen MATH500 file SHA, 500 canonical question IDs, audit counts/ID hashes,
exact/normalized consistency, accepted contracts and question totals are checked.
No raw responses, regrading, training, remote access or changes to official scores.

Example (NumPy required):
  python scripts/build_overlap_sensitivity.py --input /local/qwen4_final.json \
    --audit paper/iclr2027/internal/dataset_overlap_audit.json --primary-ci \
    --output-dir paper/iclr2027/generated/qwen4_overlap_final

Default: all four accepted checkpoint pairs AND their question scores are required.
--allow-partial permits unavailable pairs, never malformed evidence. --primary-ci
adds only Step200 paired-question percentile intervals on the 496 retained items.
Without --output-dir, print the small report to stdout and write nothing. With it,
create qwen4_overlap_sensitivity.json and bilingual _en.tex/_zh.tex exclusively.
For full tables, export question scores for steps 200 150 100 50, not only 200.
The audit is trusted evidence bound by hashes, not a repeated raw-dataset audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from . import build_qwen4_paper_assets as assets
else:
    import build_qwen4_paper_assets as assets


# Identity of the frozen file, not an exclusion list or a result estimate.
MATH500_SHA256 = "cc164b47b60771eeda257bcb9a2068940cc0706bb8bb79b6635af4c1d4534501"
MATH_IDS = frozenset(f"math500/test/{i}" for i in range(500))
SCOPE = (
    "Post-hoc overlap sensitivity on a 496-question MATH500 subset; not a new benchmark "
    "or a primary result. Official full-500 scores remain unchanged. Exclusions follow "
    "the audit's whitespace-collapse plus casefold matches, including exact matches. "
    "No semantic decontamination or causal effect of overlap is established."
)
UNCERTAINTY = (
    "Optional Step200 pointwise 95% percentile paired-question bootstrap of Block3 minus "
    "Token on the same 496 retained MATH500 questions, in percentage points. Conditional "
    "on one seed21 training pair; not training-run variance or a multi-training-seed CI."
)


def list_sha256(items):
    raw = json.dumps(items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def audit_exclusions(bundle, audit):
    assets.require(audit["schema_version"] == 1, "unsupported overlap audit schema")
    assets.require(
        audit["mode"] == "read_only_cpu_exact_string_set_membership", "unknown audit mode"
    )
    assets.require(
        audit["train"]["path"] == str(assets.DATA.parent / "train.parquet"),
        "overlap audit training data path mismatch",
    )
    task = audit["per_benchmark"]["math500"]
    assets.check_source(task["jsonl"], assets.DATA / "math500.jsonl")
    assets.require(
        task["jsonl"]["sha256"] == bundle["data_sha256"]["math500"] == MATH500_SHA256,
        "frozen MATH500 data SHA mismatch",
    )
    assets.require(task["heldout_question_field"] == "problem", "unexpected audited field")
    definition = audit["definitions"]["whitespace_casefold"]
    assets.require(
        isinstance(definition, str) and "' '.join(question.split()).casefold()" in definition,
        "unknown audit normalization definition",
    )
    ids, matches = {}, {}
    for name in ("exact", "whitespace_casefold"):
        view = task["views"][name]
        assets.integer(view["heldout_rows"], "audit heldout count", 500, 500)
        assets.integer(view["heldout_unique_comparison_keys"], "audit unique count", 500, 500)
        assets.require(isinstance(view["matches"], list), "invalid audit matches")
        ids[name] = [row["evaluation_id"] for row in view["matches"]]
        assets.require(
            all(isinstance(identity, str) and identity in MATH_IDS for identity in ids[name]),
            "unknown audit evaluation ID",
        )
        assets.require(len(set(ids[name])) == len(ids[name]), "duplicate audit evaluation ID")
        count = len(ids[name])
        assets.integer(view["overlap_heldout_rows"], "audit overlap count", count, count)
        for row in view["matches"]:
            assets.check_sha(row["eval_question_sha256"])
            assets.check_sha(row["comparison_key_sha256"])
        unique = len({row["comparison_key_sha256"] for row in view["matches"]})
        assets.integer(view["overlap_unique_comparison_keys"], "audit key count", unique, unique)
        assets.require(
            view["overlap_heldout_ids_sha256"] == list_sha256(ids[name]),
            "audit evaluation ID-list hash mismatch",
        )
        matches[name] = {row["evaluation_id"]: row for row in view["matches"]}
    exact, normalized = set(ids["exact"]), set(ids["whitespace_casefold"])
    assets.require(len(exact) == 2 and len(normalized) == 4, "expected audited 2/4 overlap counts")
    assets.require(exact <= normalized, "exact audit IDs are not a normalized subset")
    for name in matches:
        for identity, row in matches[name].items():
            assets.require(
                row["also_exact_match"] is (identity in exact), "audit exact flag mismatch"
            )
            if identity in exact:
                assets.require(
                    row["eval_question_sha256"]
                    == matches["exact"][identity]["eval_question_sha256"],
                    "audit question hash differs across views",
                )
    additional = task["normalized_only_additional_heldout_ids"]
    assets.require(
        isinstance(additional, list)
        and len(additional) == len(set(additional))
        and set(additional) == normalized - exact,
        "audit additional IDs mismatch",
    )
    assets.integer(
        task["normalized_only_additional_heldout_rows"],
        "audit additional count",
        len(additional),
        len(additional),
    )
    return ids["whitespace_casefold"], definition


def analyze_sensitivity(
    bundle,
    audit,
    *,
    allow_partial=False,
    primary_ci=False,
    bootstrap_replicates=10000,
    bootstrap_seed=20260921,
):
    try:
        return _analyze_sensitivity(
            bundle, audit, allow_partial, primary_ci, bootstrap_replicates, bootstrap_seed
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed overlap sensitivity input: {error}") from error


def _analyze_sensitivity(bundle, audit, allow_partial, primary_ci, replicates, seed):
    assets.integer(replicates, "bootstrap replicates", 1, 1_000_000)
    assets.integer(seed, "bootstrap seed", 0, 2**63 - 1)
    assets.analyze(bundle, allow_partial=allow_partial, bootstrap_replicates=1, bootstrap_seed=seed)
    excluded, definition = audit_exclusions(bundle, audit)
    retained = MATH_IDS - set(excluded)
    paired = {}
    for variant, arm in bundle["arms"].items():
        for step, cp in arm["checkpoints"].items():
            if "questions" not in cp:
                continue
            rows = assets.question_groups(cp["questions"], cp["result"], variant, int(step))[
                "math500"
            ]
            assets.require({row["id"] for row in rows} == MATH_IDS, "unknown MATH500 question IDs")
            paired[variant, int(step)] = [row for row in rows if row["id"] in retained]
    unavailable = [
        step for step in assets.STEPS if not all((v, step) in paired for v in assets.VARIANTS)
    ]
    assets.require(
        allow_partial or not unavailable, f"missing paired question scores: {unavailable}"
    )
    assets.require(len(unavailable) < len(assets.STEPS), "no paired subset question scores")
    assets.require(
        not primary_ci or 200 not in unavailable, "Step200 question scores required for CI"
    )
    rows = []
    for step in assets.STEPS:
        row = {"step": step, "arms": None, "delta_pp": None, "paired_ci": None}
        if step not in unavailable:
            arms = {}
            for variant in assets.VARIANTS:
                counts = [r["correct_count"] for r in paired[variant, step]]
                assets.require(len(counts) == 496, "post-hoc subset must contain 496 questions")
                arms[variant] = {
                    "avg_at_8": sum(counts) / (8 * len(counts)),
                    "pass_at_8": sum(value > 0 for value in counts) / len(counts),
                }
            row["arms"] = arms
            row["delta_pp"] = {
                metric: 100 * (arms[assets.VARIANTS[0]][metric] - arms[assets.VARIANTS[1]][metric])
                for metric in assets.METRICS[:2]
            }
            if primary_ci and step == 200:
                row["paired_ci"] = assets.paired_bootstrap(
                    paired[assets.VARIANTS[0], step],
                    paired[assets.VARIANTS[1], step],
                    replicates=replicates,
                    seed=seed,
                )
        rows.append(row)
    return {
        "schema_version": 1,
        "complete": not unavailable,
        "unavailable_steps": unavailable,
        "scope": SCOPE,
        "uncertainty_scope": UNCERTAINTY,
        "official_questions": 500,
        "subset_questions": 496,
        "data_sha256": MATH500_SHA256,
        "excluded_ids": excluded,
        "retained_ids_sha256": list_sha256(sorted(retained)),
        "exclusion_rule": "whitespace_casefold",
        "normalization_definition": definition,
        "units": {"arm_scores": "fraction", "delta_and_ci": "percentage points"},
        "options": {
            "allow_partial": allow_partial,
            "primary_ci": primary_ci,
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
        },
        "rows": rows,
    }


def tex_table(report, language):
    status = "COMPLETE" if report["complete"] else "PARTIAL"
    caption = f"Post-hoc overlap sensitivity: MATH500 subset (496 questions; {status})."
    step_label = "Step"
    note = (
        "Excludes four audit whitespace+casefold matches, including two exact matches. "
        "This is not a new benchmark or a primary result; official full-500 scores are unchanged. "
        "Scores are percentages; $\\Delta$ is Block3 minus Token in percentage points. "
        "NA: paired question scores unavailable."
    )
    ci_note = (
        " Brackets at Step200 only: pointwise 95\\% paired-question bootstrap on 496 questions; "
        "conditional on one training pair, not a multi-training-seed CI."
    )
    if language == "zh":
        caption = (
            "\u4e8b\u540e\u91cd\u53e0\u654f\u611f\u6027\u5206\u6790\uff1a"
            f"MATH500\u7684496\u9898\u5b50\u96c6 ({status})\u3002"
        )
        step_label = "\u6b65\u6570"
        note = (
            "\u5254\u9664\u5ba1\u8ba1\u4e2d\u7a7a\u767d\u6298\u53e0\u52a0casefold\u5339\u914d\u76844\u9898\uff0c\u5305\u542b2\u9898\u7cbe\u786e\u5339\u914d\u3002"
            "\u4ec5\u4e3a\u4e8b\u540e\u654f\u611f\u6027\u5206\u6790\uff0c\u4e0d\u662f\u65b0\u57fa\u51c6\u6216\u4e3b\u7ed3\u679c\uff1b\u5b98\u65b9\u5168500\u9898\u5206\u6570\u4e0d\u53d8\u3002"
            "\u5206\u6570\u4e3a\u767e\u5206\u6bd4\uff1b$\\Delta$\u4e3aBlock3\u51cfToken\u7684\u767e\u5206\u70b9\u5dee\uff1bNA\u8868\u793a\u7f3a\u5c11\u914d\u5bf9\u9898\u76ee\u5206\u6570\u3002"
        )
        ci_note = (
            " \u4ec5Step200\u65b9\u62ec\u53f7\u4e3a496\u9898\u5185\u914d\u5bf9\u95ee\u9898"
            "bootstrap\u7684\u9010\u70b995\\%\u533a\u95f4\uff1b"
            "\u6761\u4ef6\u4e8e\u4e00\u5bf9\u8bad\u7ec3\u8fd0\u884c\uff0c\u975e\u591a\u8bad\u7ec3\u79cd\u5b50CI\u3002"
        )
    lines = [
        "% Generated post-hoc sensitivity only; requires booktabs and CJK for Chinese.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        "\\caption{" + caption + "}",
        r"\label{tab:qwen4overlapsensitivity}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Avg@8} & \multicolumn{3}{c}{Pass@8} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        step_label + r" & Block3 & Token & $\Delta$ & Block3 & Token & $\Delta$ \\",
        r"\midrule",
    ]
    for row in report["rows"]:
        cells = [str(row["step"])]
        for metric in assets.METRICS[:2]:
            if row["arms"] is None:
                cells.extend(["NA"] * 3)
                continue
            cells.extend(f"{100 * row['arms'][v][metric]:.2f}" for v in assets.VARIANTS)
            delta = f"{row['delta_pp'][metric]:+.2f}"
            if row["paired_ci"] is not None:
                ci = row["paired_ci"][metric]
                delta += f" [{ci['lower_95_pp']:.2f}, {ci['upper_95_pp']:.2f}]"
            cells.append(delta)
        lines.append(" & ".join(cells) + r" \\")
    if report["options"]["primary_ci"]:
        note += ci_note
    return "\n".join(
        [
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip\footnotesize " + note,
            r"\end{table*}",
            "",
        ]
    )


def build(input_path, audit_path, output_dir=None, **options):
    input_path, audit_path = Path(input_path), Path(audit_path)
    raw, audit_raw = input_path.read_bytes(), audit_path.read_bytes()
    bundle = json.loads(raw, object_pairs_hook=assets.unique_json_object)
    audit = json.loads(audit_raw, object_pairs_hook=assets.unique_json_object)
    report = analyze_sensitivity(bundle, audit, **options)
    report["provenance"] = {
        "input_path": str(input_path.resolve()),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "audit_path": str(audit_path.resolve()),
        "audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "builder_sha256": assets.sha256(Path(__file__)),
        "shared_helpers_sha256": assets.sha256(Path(assets.__file__)),
        "numpy_version": assets.np.__version__,
    }
    if output_dir is not None:
        output = assets.check_destination(output_dir)
        texts = {"qwen4_overlap_sensitivity.json": assets.json_text(report)}
        texts.update(
            {
                f"qwen4_overlap_sensitivity_{lang}.tex": tex_table(report, lang)
                for lang in ("en", "zh")
            }
        )
        for name in texts:
            path = output / name
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"refusing to overwrite {path}; use a fresh directory")
        output.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            with (output / name).open("x", encoding="utf-8", newline="") as handle:
                handle.write(text)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--primary-ci", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    args = parser.parse_args(argv)
    try:
        report = build(
            args.input,
            args.audit,
            args.output_dir,
            allow_partial=args.allow_partial,
            primary_ci=args.primary_ci,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(assets.json_text(report), end="")
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()

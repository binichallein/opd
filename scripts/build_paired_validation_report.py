#!/usr/bin/env python3
"""Build a self-contained HTML verdict for a paired OPD validation."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


STEPS = (50, 100, 200)
TASKS = ("math500", "aime24", "aime25", "amc23")
DIAGNOSTIC_IMAGES = (
    "scalar_alignment.png",
    "scalar_credit.png",
    "scalar_optimization.png",
    "position_heatmaps.png",
    "entropy_segments.png",
)
VERDICTS = {
    "strong_single_seed_support": (
        "强复现（单训练 seed）",
        "Block3 在 Step 200 的 macro Avg@8 与 Pass@8 均为正，且两个 paired bootstrap 95% CI 均严格高于 0。",
        "positive",
    ),
    "directional_support": (
        "方向复现，但不构成强复现",
        "两个 Step 200 主终点的点估计均为正，但至少一个 95% CI 仍覆盖 0。",
        "caution",
    ),
    "not_supported": (
        "未复现",
        "Step 200 的 macro Avg@8 或 Pass@8 至少有一个点估计不大于 0。",
        "negative",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-json", type=Path, required=True)
    parser.add_argument("--builtin-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--token-diagnostics", type=Path, required=True)
    parser.add_argument("--block-diagnostics", type=Path, required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing comparison JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_comparisons(
    external: dict[str, Any], builtin: dict[str, Any]
) -> dict[str, Any]:
    if not str(external.get("grader_view", "")).startswith(
        "historical_external_grader"
    ):
        raise ValueError("primary comparison must use an audited external grader view")
    if builtin.get("grader_view") != "outputs":
        raise ValueError("sensitivity comparison must use builtin outputs view")
    if external.get("training_contract") != builtin.get("training_contract"):
        raise ValueError("builtin and external training contracts differ")
    for key in ("left", "right"):
        if external.get(key) != builtin.get(key):
            raise ValueError(f"builtin and external {key} runs differ")
    expected_steps = {str(step) for step in STEPS}
    if set(external.get("steps", {})) != expected_steps:
        raise ValueError("external comparison steps are incomplete")
    if set(builtin.get("steps", {})) != expected_steps:
        raise ValueError("builtin comparison steps are incomplete")
    decision = external.get("decision", {})
    if decision.get("status") not in VERDICTS or int(decision.get("step", -1)) != 200:
        raise ValueError("external comparison has no valid Step 200 decision")
    bootstrap = external.get("bootstrap", {})
    if int(bootstrap.get("replicates", -1)) != 10_000:
        raise ValueError("report requires 10,000 bootstrap replicates")
    return external["training_contract"]


def _relative_image(output: Path, directory: Path, filename: str) -> str:
    path = directory / filename
    if not path.is_file():
        raise ValueError(f"missing diagnostic image: {path}")
    return Path(os.path.relpath(path, output.parent)).as_posix()


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def _step_rows(result: dict[str, Any]) -> str:
    rows = []
    for step in STEPS:
        record = result["steps"][str(step)]
        avg = record["macro"]["avg_at_n"]
        passed = record["macro"]["pass_at_n"]
        avg_ci = record["bootstrap"]["avg_at_n"]
        pass_ci = record["bootstrap"]["pass_at_n"]
        rows.append(
            "<tr>"
            f"<th>Step {step}</th>"
            f"<td>{_fmt(avg['left'])}</td><td>{_fmt(avg['right'])}</td>"
            f"<td class='delta'>{_fmt(avg['delta'])}</td>"
            f"<td>[{_fmt(avg_ci['lower_95'])}, {_fmt(avg_ci['upper_95'])}]</td>"
            f"<td>{_fmt(passed['left'])}</td><td>{_fmt(passed['right'])}</td>"
            f"<td class='delta'>{_fmt(passed['delta'])}</td>"
            f"<td>[{_fmt(pass_ci['lower_95'])}, {_fmt(pass_ci['upper_95'])}]</td>"
            "</tr>"
        )
    return "".join(rows)


def _task_rows(result: dict[str, Any]) -> str:
    record = result["steps"]["200"]
    rows = []
    for task in TASKS:
        values = record["per_task"][task]
        avg = values["avg_at_n"]
        passed = values["pass_at_n"]
        rows.append(
            "<tr>"
            f"<th>{html.escape(task)}</th>"
            f"<td>{_fmt(avg['left'])}</td><td>{_fmt(avg['right'])}</td>"
            f"<td class='delta'>{_fmt(avg['delta'])}</td>"
            f"<td>{_fmt(passed['left'])}</td><td>{_fmt(passed['right'])}</td>"
            f"<td class='delta'>{_fmt(passed['delta'])}</td>"
            "</tr>"
        )
    return "".join(rows)


def _outcome_rows(result: dict[str, Any]) -> str:
    outcomes = result["steps"]["200"]["prompt_outcomes"]
    labels = (("avg_at_n", "Avg@8"), ("pass_at_n", "Pass@8"))
    return "".join(
        "<tr>"
        f"<th>{label}</th>"
        f"<td>{int(outcomes[key]['win'])}</td>"
        f"<td>{int(outcomes[key]['tie'])}</td>"
        f"<td>{int(outcomes[key]['loss'])}</td>"
        "</tr>"
        for key, label in labels
    )


def _diagnostic_sections(
    output: Path, token_directory: Path, block_directory: Path
) -> str:
    labels = (
        ("token_opd", token_directory),
        ("block3_mean", block_directory),
    )
    sections = []
    for label, directory in labels:
        images = "".join(
            "<figure>"
            f"<img src='{html.escape(_relative_image(output, directory, filename))}' "
            f"alt='{html.escape(label + ' ' + filename)}'>"
            f"<figcaption>{html.escape(filename.removesuffix('.png').replace('_', ' '))}</figcaption>"
            "</figure>"
            for filename in DIAGNOSTIC_IMAGES
        )
        sections.append(
            f"<section class='diagnostic-band'><h3>{html.escape(label)}</h3>"
            f"<div class='figure-grid'>{images}</div></section>"
        )
    return "".join(sections)


def write_report(
    external: dict[str, Any],
    builtin: dict[str, Any],
    output: Path,
    *,
    token_diagnostics: Path,
    block_diagnostics: Path,
    title: str,
) -> None:
    contract = validate_comparisons(external, builtin)
    status = external["decision"]["status"]
    verdict, explanation, tone = VERDICTS[status]
    output.parent.mkdir(parents=True, exist_ok=True)
    student = Path(str(contract["student_model"])).name
    teacher = Path(str(contract["teacher_model"])).name
    diagnostics = _diagnostic_sections(
        output, token_diagnostics, block_diagnostics
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ --ink:#172033; --muted:#5c6675; --line:#d6dce5; --panel:#f5f7fa; --blue:#155eef; --green:#18794e; --amber:#9a6700; --red:#b42318; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.55; letter-spacing:0; }}
main {{ width:min(1440px,calc(100% - 32px)); margin:28px auto 64px; }}
header {{ border-bottom:3px solid var(--ink); padding:0 0 20px; }}
h1 {{ margin:0 0 8px; font-size:clamp(25px,3vw,40px); }}
h2 {{ margin:34px 0 12px; font-size:22px; }} h3 {{ margin:0 0 12px; font-size:18px; }}
p {{ margin:0; color:var(--muted); }}
.verdict {{ margin:22px 0; padding:18px 20px; border-left:5px solid var(--green); background:#eef8f2; }}
.verdict.caution {{ border-color:var(--amber); background:#fff8e6; }} .verdict.negative {{ border-color:var(--red); background:#fff1f0; }}
.verdict strong {{ display:block; font-size:24px; margin-bottom:4px; }}
.facts {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); }}
.fact {{ min-width:0; padding:14px 16px; border-right:1px solid var(--line); }} .fact:last-child {{ border-right:0; }}
.fact span {{ display:block; color:var(--muted); font-size:12px; }} .fact strong {{ display:block; overflow-wrap:anywhere; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); }}
table {{ width:100%; min-width:880px; border-collapse:collapse; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
thead th {{ background:var(--panel); font-size:12px; color:var(--muted); }} tbody th {{ text-align:left; }} tbody tr:last-child th,tbody tr:last-child td {{ border-bottom:0; }}
.delta {{ color:var(--blue); font-weight:700; }}
.contract {{ width:100%; border-collapse:collapse; }} .contract th {{ width:25%; text-align:left; background:var(--panel); }} .contract td {{ text-align:left; white-space:normal; overflow-wrap:anywhere; }}
.diagnostic-band {{ margin:0 0 34px; }}
.figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
figure {{ margin:0; }} img {{ display:block; width:100%; height:auto; border:1px solid var(--line); }} figcaption {{ padding:7px 0; color:var(--muted); font-size:12px; }}
.note {{ margin-top:18px; padding:14px 16px; border:1px solid var(--line); background:var(--panel); color:var(--muted); }}
@media(max-width:800px) {{ main {{ width:calc(100% - 20px); margin-top:16px; }} .facts,.figure-grid {{ grid-template-columns:1fr; }} .fact {{ border-right:0; border-bottom:1px solid var(--line); }} }}
</style></head><body><main>
<header><h1>{html.escape(title)}</h1><p>sampled-token OPD 与 block3_mean 的严格配对验证报告</p></header>
<section class="verdict {tone}"><strong>{html.escape(verdict)}</strong><p>{html.escape(explanation)}</p></section>
<section class="facts">
<div class="fact"><span>Student</span><strong>{html.escape(student)}</strong></div>
<div class="fact"><span>Teacher</span><strong>{html.escape(teacher)}</strong></div>
<div class="fact"><span>Matched prompt batches</span><strong>{int(contract['prompt_batch_hashes_matched'])}</strong></div>
<div class="fact"><span>Bootstrap</span><strong>10,000 paired replicates</strong></div>
</section>
<h2>实验契约</h2><table class="contract"><tbody>
<tr><th>Student revision</th><td>{html.escape(str(contract.get('student_model_revision')))}</td></tr>
<tr><th>Teacher revision</th><td>{html.escape(str(contract.get('teacher_model_revision')))}</td></tr>
<tr><th>Training data SHA-256</th><td>{html.escape(str(contract['train_sha256']))}</td></tr>
<tr><th>Eval data SHA-256</th><td>{html.escape(str(contract['test_sha256']))}</td></tr>
<tr><th>Runtime commit</th><td>{html.escape(str(contract['source_commit']))}</td></tr>
<tr><th>Fixed training</th><td>DAPO-Math-17K pool; seed 21; 200 steps; batch 4; 8 rollouts; LR 2e-6; max response 16,384.</td></tr>
<tr><th>Fixed evaluation</th><td>Math500, AIME24, AIME25, AMC23; n=8; seeds 21-28; temperature 1.0; top-p 0.9; max 16,384; thinking off.</td></tr>
</tbody></table>
<h2>主结果：固定历史 grader</h2><div class="scroll"><table><thead><tr><th>Checkpoint</th><th>Token Avg@8</th><th>Block3 Avg@8</th><th>Delta</th><th>Avg 95% CI</th><th>Token Pass@8</th><th>Block3 Pass@8</th><th>Delta</th><th>Pass 95% CI</th></tr></thead><tbody>{_step_rows(external)}</tbody></table></div>
<h2>Step 200 分任务</h2><div class="scroll"><table><thead><tr><th>Task</th><th>Token Avg@8</th><th>Block3 Avg@8</th><th>Delta</th><th>Token Pass@8</th><th>Block3 Pass@8</th><th>Delta</th></tr></thead><tbody>{_task_rows(external)}</tbody></table></div>
<h2>Step 200 prompt 级配对</h2><div class="scroll"><table><thead><tr><th>Metric</th><th>Block3 win</th><th>Tie</th><th>Block3 loss</th></tr></thead><tbody>{_outcome_rows(external)}</tbody></table></div>
<h2>敏感性分析：内置 VERL grader</h2><div class="scroll"><table><thead><tr><th>Checkpoint</th><th>Token Avg@8</th><th>Block3 Avg@8</th><th>Delta</th><th>Avg 95% CI</th><th>Token Pass@8</th><th>Block3 Pass@8</th><th>Delta</th><th>Pass 95% CI</th></tr></thead><tbody>{_step_rows(builtin)}</tbody></table></div>
<h2>训练诊断与位置热力图</h2>{diagnostics}
<p class="note">结论范围：一个严格配对的 training seed。题目分层 bootstrap 量化固定 checkpoint 下的 prompt 不确定性，不能替代多 training-seed 方差。</p>
</main></body></html>"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_report(
        load_json(args.external_json),
        load_json(args.builtin_json),
        args.output_html,
        token_diagnostics=args.token_diagnostics,
        block_diagnostics=args.block_diagnostics,
        title=args.title,
    )


if __name__ == "__main__":
    main()

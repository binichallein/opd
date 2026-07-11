#!/usr/bin/env python3
"""Render scalar curves and position diagnostics for one OPD training run."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
import sys


ARTIFACTS = (
    ("scalar_alignment.png", "师生分布与对齐"),
    ("scalar_credit.png", "Block credit assignment"),
    ("scalar_optimization.png", "优化状态与 policy drift"),
    ("position_heatmaps.png", "位置级热力图"),
    ("entropy_segments.png", "前、中、后段熵曲线"),
)

METRIC_FAMILIES = (
    ("Student entropy", "学生策略在有效 response token 上的平均熵及位置分布。"),
    ("Teacher entropy", "教师在相同学生前缀上的平均熵及位置分布。"),
    ("Signed entropy gap", "Teacher entropy - Student entropy，保留方向。"),
    ("Absolute entropy gap", "师生熵差绝对值，衡量不匹配幅度。"),
    ("Top-16 overlap ratio", "师生 Top-16 token 集合的交集比例。"),
    ("Student/teacher overlap mass", "师生各自在共享 Top-16 token 上承载的概率质量。"),
    ("Overlap-token advantage", "在共享 Top-16 支撑集内分别归一化师生概率后，计算学生加权 log-ratio，再除以共享 token 数。"),
    ("SignFlipRate / WeightedSignFlipRate", "聚合前后 advantage 符号翻转比例及按绝对幅度加权的翻转比例。"),
    ("LeakageMagnitude / NormalizedLeakage", "所有有效 token 上 block advantage 与原 token advantage 的平均绝对差，以及该差值总量相对原 advantage 绝对值总量的比例。"),
    ("Raw-token and block advantage summaries", "分别记录 mean、std、p95 和 max，检查 Block3 聚合是否放大信号。"),
    ("Post-update block log-ratio", "更新后新策略相对 rollout old policy 的 block log-ratio。"),
    ("Outside-clip fraction", "更新后 block policy ratio 落在 PPO clip 区间外的有效 block 比例。"),
    ("PG loss and pre-clip grad norm", "策略梯度损失、clip fraction 与裁剪前梯度范数。"),
    ("Response length and truncation ratio", "平均/最大输出长度及触及 16K 上限的比例。"),
    ("Rollout policy drift", "由 post-update block log-ratio、policy ratio 及 outside-clip fraction 联合刻画。"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="ml2-A100 Block3")
    parser.add_argument("--position-bin", type=int, default=128)
    parser.add_argument("--min-count", type=int, default=8)
    return parser.parse_args()


def write_report(
    output_dir: Path,
    label: str,
    record_count: int,
    snapshot_count: int,
    first_step: int,
    last_step: int,
) -> None:
    metric_rows = "".join(
        "<tr>"
        f"<th>{html.escape(name)}</th>"
        f"<td>{html.escape(description)}</td>"
        "</tr>"
        for name, description in METRIC_FAMILIES
    )
    figures = "".join(
        "<section class=\"figure\">"
        f"<h2>{html.escape(title)}</h2>"
        f"<img src=\"{filename}\" alt=\"{html.escape(title)}\">"
        "</section>"
        for filename, title in ARTIFACTS
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(label)} OPD 诊断</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#5d6878; --line:#d8dee8; --panel:#f5f7fa; --accent:#155eef; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.55; letter-spacing:0; }}
main {{ width:min(1320px,calc(100% - 32px)); margin:32px auto 64px; }}
header {{ border-left:4px solid var(--accent); padding:4px 0 4px 18px; margin-bottom:24px; }}
h1 {{ margin:0 0 6px; font-size:clamp(25px,3vw,38px); }}
h2 {{ margin:0 0 10px; font-size:20px; }}
p {{ margin:0; color:var(--muted); }}
.facts {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); margin:20px 0 28px; }}
.fact {{ min-width:0; background:#fff; padding:14px 16px; }}
.fact span {{ display:block; color:var(--muted); font-size:12px; }}
.fact strong {{ display:block; overflow-wrap:anywhere; font-size:17px; }}
table {{ width:100%; border-collapse:collapse; margin:10px 0 32px; }}
th,td {{ border:1px solid var(--line); padding:10px 12px; text-align:left; vertical-align:top; }}
th {{ width:31%; background:var(--panel); font-size:14px; }}
td {{ color:#354052; }}
.figure {{ margin:0 0 34px; }}
img {{ display:block; width:100%; height:auto; border:1px solid var(--line); background:#fff; }}
@media (max-width:760px) {{ main {{ width:min(100% - 20px,1320px); margin-top:18px; }} .facts {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} th,td {{ display:block; width:100%; }} td {{ border-top:0; }} }}
</style>
</head>
<body><main>
<header><h1>{html.escape(label)} OPD 诊断</h1><p>本报告只呈现该次 Block3 运行的观测量，不进行 Block10 式跨机器因果分类。</p></header>
<section class="facts">
<div class="fact"><span>Scalar records</span><strong>{record_count}</strong></div>
<div class="fact"><span>Position snapshots</span><strong>{snapshot_count}</strong></div>
<div class="fact"><span>First diagnostic step</span><strong>{first_step}</strong></div>
<div class="fact"><span>Last diagnostic step</span><strong>{last_step}</strong></div>
</section>
<section><h2>监控指标</h2><table><tbody>{metric_rows}</tbody></table></section>
{figures}
</main></body></html>"""
    (output_dir / "diagnostics.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent
    for path in (scripts_dir, root_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from analyze_block10_collapse_diagnostics import (  # noqa: PLC0415
        SCALAR_GROUPS,
        load_snapshots,
        plot_entropy_segments,
        plot_heatmaps,
        plot_scalar_curves,
    )
    from opd_ext.analysis import load_scalar_records  # noqa: PLC0415

    records = load_scalar_records(args.run_dir / "diagnostics" / "scalars.jsonl")
    snapshots = load_snapshots(args.run_dir)
    records_by_run = {args.label: records}
    snapshots_by_run = {args.label: snapshots}

    for filename, metrics in SCALAR_GROUPS.items():
        plot_scalar_curves(records_by_run, metrics, args.output_dir / filename)
    plot_heatmaps(
        snapshots_by_run,
        args.output_dir / "position_heatmaps.png",
        args.position_bin,
        args.min_count,
    )
    plot_entropy_segments(
        snapshots_by_run,
        args.output_dir / "entropy_segments.png",
    )

    steps = [int(record["step"]) for record in records]
    snapshot_steps = [int(snapshot["step"]) for snapshot in snapshots]
    all_steps = steps + snapshot_steps
    write_report(
        output_dir=args.output_dir,
        label=args.label,
        record_count=len(records),
        snapshot_count=len(snapshots),
        first_step=min(all_steps),
        last_step=max(all_steps),
    )


if __name__ == "__main__":
    main()

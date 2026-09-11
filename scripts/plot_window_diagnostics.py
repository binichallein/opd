#!/usr/bin/env python3
"""Paired window diagnostics, using an isolated CPU plotting environment."""

import argparse
import html
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opd_ext.analysis import load_scalar_records
from scripts.analyze_block10_collapse_diagnostics import (
    SCALAR_GROUPS, load_snapshots, plot_heatmaps, plot_scalar_curves,
)


def render(run_root, output):
    runs = {name: run_root / name for name in ("random3", "sliding3")}
    snapshots = {name: load_snapshots(run) for name, run in runs.items()}
    records = {name: load_scalar_records(run / "diagnostics/scalars.jsonl") for name, run in runs.items()}
    for rows in records.values():
        for row in rows:
            for key in list(row):
                if key.endswith("_cosine") and row.get(key + "_valid") == 0:
                    del row[key]
    output.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(snapshots, output / "position_heatmaps.png", 128, 8)
    groups = dict(SCALAR_GROUPS)
    groups["loss_input_phase.png"] = (
        "diagnostics/loss_input_phase_disagreement_squared_norm",
        "diagnostics/loss_input_mean_phase_gradient_norm",
        "diagnostics/loss_input_mean_gradient_norm",
        "diagnostics/loss_input_gradient_identity_error",
        "diagnostics/loss_input_phase_0_1_cosine",
        "diagnostics/loss_input_phase_0_2_cosine",
        "diagnostics/loss_input_phase_1_2_cosine",
        "diagnostics/token_any_window_outside_clip_fraction",
        "diagnostics/coefficient_abs_mod0", "diagnostics/coefficient_abs_mod1",
        "diagnostics/coefficient_abs_mod2",
    )
    for filename, keys in groups.items():
        plot_scalar_curves(records, keys, output / filename)
    figures = "".join(f'<section><h2>{html.escape(name)}</h2><img src="{name}" alt="{name}"></section>'
                      for name in ("position_heatmaps.png", *groups))
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Random3 与 Sliding3 诊断</title>
<style>body{{font:16px/1.6 sans-serif;color:#20252c;margin:24px auto;padding:0 16px;max-width:1500px;letter-spacing:0}}
h1{{font-size:26px}}h2{{font-size:18px;overflow-wrap:anywhere}}section{{margin:32px 0}}img{{display:block;width:100%;height:auto}}
</style><h1>Random3 与 Sliding3：seed21 训练诊断</h1>
<p>仅比较本轮两个方法。loss_input 指对采样 log-prob 的导数，不是完整模型参数梯度；
sign-flip 和 leakage 为各偏移窗口均值的共享信号视图，不直接等于错误归因。
位置图使用共同色标，低覆盖位置记为缺失；零范数 cosine 需结合有效性计数，不作正交解释。
最终准确率结论须结合完整评测，不能由这些曲线单独判定。</p>{figures}</html>'''
    (output / "index.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.run_root, args.output_dir)

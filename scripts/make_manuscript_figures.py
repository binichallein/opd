#!/usr/bin/env python3
"""Generate dependency-free SVG manuscript figures from clean-room results."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("OPD_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "gray": "#777777",
    "light": "#EEEEEE",
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .title{font-size:15px;font-weight:700} '
        '.label{font-size:12px} .tick{font-size:10px;fill:#555} .legend{font-size:11px}</style>',
    ]


def write_svg(name: str, width: int, height: int, body: list[str]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.svg"
    path.write_text("\n".join(svg_header(width, height) + body + ["</svg>\n"]), encoding="utf-8")


def line_panel(
    body: list[str],
    x0: int,
    y0: int,
    w: int,
    h: int,
    title: str,
    xs: list[float],
    series: list[tuple[str, list[float], str]],
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 44, 12, 26, 32
    px0, py0 = x0 + pad_l, y0 + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    vals = [v for _, ys, _ in series for v in ys]
    lo = min(vals) if y_min is None else y_min
    hi = max(vals) if y_max is None else y_max
    if hi == lo:
        hi = lo + 1
    span = hi - lo
    lo -= 0.05 * span
    hi += 0.05 * span

    def sx(x: float) -> float:
        return px0 + (x - min(xs)) / (max(xs) - min(xs)) * pw

    def sy(y: float) -> float:
        return py0 + (hi - y) / (hi - lo) * ph

    body.append(f'<text class="title" x="{x0}" y="{y0 + 14}">{esc(title)}</text>')
    body.append(f'<line x1="{px0}" y1="{py0 + ph}" x2="{px0 + pw}" y2="{py0 + ph}" stroke="#222" stroke-width="1"/>')
    body.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + ph}" stroke="#222" stroke-width="1"/>')
    for frac in [0, 0.5, 1.0]:
        yv = lo + frac * (hi - lo)
        yy = sy(yv)
        body.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px0 + pw}" y2="{yy:.1f}" stroke="#ddd"/>')
        body.append(f'<text class="tick" x="{px0 - 8}" y="{yy + 3:.1f}" text-anchor="end">{yv:.2f}</text>')
    for x in xs:
        xx = sx(x)
        body.append(f'<text class="tick" x="{xx:.1f}" y="{py0 + ph + 17}" text-anchor="middle">{int(x)}</text>')
    body.append(f'<text class="label" x="{px0 + pw / 2:.1f}" y="{y0 + h - 3}" text-anchor="middle">training step</text>')
    body.append(f'<text class="label" transform="translate({x0 + 12},{py0 + ph / 2:.1f}) rotate(-90)" text-anchor="middle">accuracy</text>')
    for label, ys, color in series:
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            body.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{color}"/>')
    lx = x0 + w - 95
    ly = y0 + 20
    for i, (label, _, color) in enumerate(series):
        body.append(f'<line x1="{lx}" y1="{ly + i * 15}" x2="{lx + 14}" y2="{ly + i * 15}" stroke="{color}" stroke-width="2"/>')
        body.append(f'<text class="legend" x="{lx + 18}" y="{ly + 4 + i * 15}">{esc(label)}</text>')


def grouped_bar_panel(
    body: list[str],
    x0: int,
    y0: int,
    w: int,
    h: int,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_label: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 54, 14, 28, 42
    px0, py0 = x0 + pad_l, y0 + pad_t
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    vals = [v for _, ys, _ in series for v in ys]
    hi = max(vals) if y_max is None else y_max
    lo = y_min
    if hi == lo:
        hi = lo + 1

    def sy(y: float) -> float:
        return py0 + (hi - y) / (hi - lo) * ph

    body.append(f'<text class="title" x="{x0}" y="{y0 + 14}">{esc(title)}</text>')
    body.append(f'<line x1="{px0}" y1="{py0 + ph}" x2="{px0 + pw}" y2="{py0 + ph}" stroke="#222" stroke-width="1"/>')
    body.append(f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py0 + ph}" stroke="#222" stroke-width="1"/>')
    for frac in [0, 0.5, 1.0]:
        yv = lo + frac * (hi - lo)
        yy = sy(yv)
        body.append(f'<line x1="{px0}" y1="{yy:.1f}" x2="{px0 + pw}" y2="{yy:.1f}" stroke="#ddd"/>')
        body.append(f'<text class="tick" x="{px0 - 8}" y="{yy + 3:.1f}" text-anchor="end">{yv:.2f}</text>')
    group_w = pw / len(labels)
    bar_w = min(18, group_w / (len(series) + 1.2))
    for i, label in enumerate(labels):
        cx = px0 + group_w * (i + 0.5)
        body.append(f'<text class="tick" x="{cx:.1f}" y="{py0 + ph + 17}" text-anchor="middle">{esc(label)}</text>')
        for j, (_, ys, color) in enumerate(series):
            x = cx - (len(series) * bar_w) / 2 + j * bar_w
            y = sy(ys[i])
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{py0 + ph - y:.1f}" fill="{color}"/>')
    body.append(f'<text class="label" transform="translate({x0 + 12},{py0 + ph / 2:.1f}) rotate(-90)" text-anchor="middle">{esc(y_label)}</text>')
    lx = x0 + w - 100
    ly = y0 + 20
    for i, (label, _, color) in enumerate(series):
        body.append(f'<rect x="{lx}" y="{ly - 8 + i * 15}" width="12" height="8" fill="{color}"/>')
        body.append(f'<text class="legend" x="{lx + 17}" y="{ly + i * 15}">{esc(label)}</text>')


def figure_early_stopping() -> None:
    robust = json.loads((RESULTS / "robust_regrade_curve_summary.json").read_text())

    def curve(dataset: str, variant: str) -> list[float]:
        out = []
        for step in [0, 50, 100, 150, 200]:
            if step == 0:
                row = next(r for r in robust if r["dataset"] == dataset and r["variant"] == "base" and int(r["step"]) == 0)
            else:
                row = next(r for r in robust if r["dataset"] == dataset and r["variant"] == variant and int(r["step"]) == step)
            out.append(row["primary_correct"])
        return out

    body: list[str] = []
    xs = [0, 50, 100, 150, 200]
    line_panel(
        body,
        34,
        34,
        430,
        270,
        "A. GSM8K robust early-stopping curve",
        xs,
        [("standard", curve("gsm8k", "standard"), COLORS["blue"]), ("top-k", curve("gsm8k", "topk_router"), COLORS["vermillion"])],
        y_min=0.30,
        y_max=0.48,
    )
    line_panel(
        body,
        500,
        34,
        430,
        270,
        "B. MATH500 robust early-stopping curve",
        xs,
        [("standard", curve("math500", "standard"), COLORS["blue"]), ("top-k", curve("math500", "topk_router"), COLORS["vermillion"])],
        y_min=0.12,
        y_max=0.21,
    )
    write_svg("fig1_early_stopping", 960, 330, body)


def figure_passk_diversity() -> None:
    rows = json.loads((RESULTS / "passk_diversity" / "compact_summary.json").read_text())
    order = ["base", "standard50", "topk50", "refpen_topk100", "standard200", "topk200"]
    labels = ["base", "std50", "topk50", "ref100", "std200", "topk200"]

    def vals(dataset: str, metric: str) -> list[float]:
        return [next(r for r in rows if r["dataset"] == dataset and r["model"] == m)[metric] for m in order]

    body: list[str] = []
    grouped_bar_panel(
        body,
        34,
        34,
        430,
        250,
        "A. Pass@4 decreases under long OPD",
        labels,
        [("GSM8K", vals("gsm8k", "robust_pass_at_4"), COLORS["blue"]), ("MATH", vals("math500", "robust_pass_at_4"), COLORS["orange"])],
        "pass@4",
        y_min=0.25,
        y_max=0.68,
    )
    grouped_bar_panel(
        body,
        500,
        34,
        430,
        250,
        "B. Diversity rises during drift",
        labels,
        [
            ("GSM8K", vals("gsm8k", "avg_unique_answer_rate"), COLORS["blue"]),
            ("MATH", vals("math500", "avg_unique_answer_rate"), COLORS["orange"]),
        ],
        "unique answer rate",
        y_min=0.55,
        y_max=0.73,
    )
    grouped_bar_panel(
        body,
        34,
        320,
        430,
        250,
        "C. Text-path diversity also rises",
        labels,
        [
            ("GSM8K", vals("gsm8k", "avg_pairwise_token_jaccard_distance"), COLORS["blue"]),
            ("MATH", vals("math500", "avg_pairwise_token_jaccard_distance"), COLORS["orange"]),
        ],
        "pairwise token Jaccard distance",
        y_min=0.50,
        y_max=0.68,
    )
    grouped_bar_panel(
        body,
        500,
        320,
        430,
        250,
        "D. Mean score mirrors pass@4",
        labels,
        [("GSM8K", vals("gsm8k", "robust_mean_score"), COLORS["blue"]), ("MATH", vals("math500", "robust_mean_score"), COLORS["orange"])],
        "mean score",
        y_min=0.18,
        y_max=0.50,
    )
    write_svg("fig2_passk_diversity", 960, 610, body)


def figure_teacher_diagnostics() -> None:
    rows = json.loads((RESULTS / "teacher_distinguish" / "compact_summary.json").read_text())
    labels = ["train", "GSM8K", "MATH"]
    keys = ["train_prompts", "gsm8k_eval", "math500_eval"]

    def vals(metric: str) -> list[float]:
        return [next(r for r in rows if r["dataset"] == k)[metric] for k in keys]

    body: list[str] = []
    grouped_bar_panel(
        body,
        34,
        34,
        430,
        250,
        "A. 1.7B teacher is locally closer",
        labels,
        [("1.7B", vals("1.7B overlap student"), COLORS["sky"]), ("4B", vals("4B overlap student"), COLORS["orange"])],
        "top-k overlap with student",
        y_min=0.58,
        y_max=0.70,
    )
    grouped_bar_panel(
        body,
        500,
        34,
        430,
        250,
        "B. 4B gives larger local advantage",
        labels,
        [("1.7B", vals("1.7B adv abs"), COLORS["sky"]), ("4B", vals("4B adv abs"), COLORS["orange"])],
        "abs sampled-token advantage",
        y_min=0.20,
        y_max=0.68,
    )
    write_svg("fig3_teacher_diagnostics", 960, 330, body)


def main() -> None:
    figure_early_stopping()
    figure_passk_diversity()
    figure_teacher_diagnostics()
    print(json.dumps({"figures": sorted(p.name for p in FIGURES.glob("fig*.svg"))}, indent=2))


if __name__ == "__main__":
    main()

"""Infographic: verbatim side task vs zero-shot Opus vs GEPA, per pilot trajectory.

    python scripts/plot_gepa_infographic.py [results/gepa_pilot2] [data/deepseek_v4_pro_clean.jsonl] [out.png]

One row per trajectory: the three instructions as text cards, and their robust gain over the empty
instruction on U's tokens (mean over no-op variants; see gepa.py). Zero-shot = the best of Opus's
guesses by robust gain; GEPA = the pool's best by single-pass gain, re-scored robustly.
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE, monitor_scores  # noqa: E402

RAW = "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590"  # the DeepSeek run's raw samples (black-box monitor scores)

ARMS = [("verbatim", "True side task (verbatim block)", "#2a78d6"),
        ("zero_shot", "Zero-shot Opus (best of 3 guesses)", "#eb6834"),
        ("gepa", "GEPA best (15 iterations)", "#1baf7a")]
WRAP, MAX_LINES = 46, 9


def card_text(text):
    lines = []
    for para in text.strip().splitlines():
        if para.strip():
            lines += textwrap.wrap(para, WRAP)
    if len(lines) > MAX_LINES:
        rest = len(" ".join(lines[MAX_LINES - 1:]))
        lines = lines[: MAX_LINES - 1] + [f"… (+{rest} more characters)"]
    return "\n".join(lines)


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gepa_pilot2")
    data = sys.argv[2] if len(sys.argv) > 2 else "data/deepseek_v4_pro_clean.jsonl"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else res / "infographic.png"
    examples = {e["id"]: e for e in map(json.loads, open(data))}
    S = json.loads((res / "summary.json").read_text())
    rows = []
    for t in S["trajectories"]:
        ex = examples[t["id"]]
        zs = max(t["zero_shot"], key=lambda z: z["robust_gain"])
        rows.append({
            "title": ex["side_task"], "env": ex["environment"].split("/")[-1], "id": t["id"][:8],
            "verbatim": (ex["side_block"], t["refs"]["verbatim_block"]["robust_gain"]),
            "zero_shot": (zs["instruction"], zs["robust_gain"]),
            "gepa": (t["gepa_best"]["instruction"], t["gepa_best"]["robust_gain"]),
            "gist": t["refs"]["gist"]["robust_gain"],
        })

    mon = monitor_scores(RAW, [t["id"] for t in S["trajectories"]])
    for r, t in zip(rows, S["trajectories"]):
        r["monitor"] = mon[t["id"]]
    rows.sort(key=lambda r: r["monitor"])  # least to most suspicious
    n = len(rows)
    row_h = 2.55
    fig = plt.figure(figsize=(17, 1.6 + n * row_h), facecolor=SURFACE)
    H = 1.6 + n * row_h
    fig.text(0.015, 1 - 0.32 / H, "What explains U's transcript best? The true side task vs Opus's guesses vs GEPA",
             fontsize=15, color=INK, va="top", weight="bold")
    fig.text(0.015, 1 - 0.72 / H, "DeepSeek V4 Pro attack transcripts, reference model Qwen3-32B. Gain = how many nats T's loss on U's "
             "actions drops vs. no instruction (higher = more predictable), averaged over 4 no-op variants.", fontsize=9.5, color=INK_2, va="top")
    fig.text(0.015, 1 - 1.0 / H, "Zero-shot: Opus reads U's actions (no tool outputs) and guesses the hidden instruction. "
             "GEPA: Opus rewrites the instruction with T's loss as feedback. Both ≤ 300 characters; the verbatim block is cut to fit.",
             fontsize=9.5, color=INK_2, va="top")
    col_x = [0.015, 0.215, 0.415]
    col_w = 0.19
    for k, (key, label, color) in enumerate(ARMS):
        fig.text(col_x[k] + 0.004, 1 - 1.38 / H, label, fontsize=10.5, color=color, weight="bold", va="top")
    fig.text(0.64, 1 - 1.38 / H, "gain over no instruction (nats)", fontsize=10.5, color=INK_2, va="top")

    vmax = max(max(r[a][1] for a, _, _ in ARMS) for r in rows)
    for i, r in enumerate(rows):
        top = 1 - (1.6 + i * row_h) / H
        bottom = top - (row_h - 0.25) / H
        fig.text(0.015, top - 0.02 / H, f"{r['title']}  ·  {r['env']}  ·  {r['id']}", fontsize=10, color=INK, va="top", weight="bold")
        fig.text(0.415 + 0.19, top - 0.02 / H, f"black-box monitor: {r['monitor']}/100", fontsize=10, color=INK_2, va="top", ha="right")
        for k, (key, _, color) in enumerate(ARMS):
            y0 = bottom
            h = (top - 0.28 / H) - bottom
            fig.patches.append(FancyBboxPatch((col_x[k], y0), col_w, h, boxstyle="round,pad=0,rounding_size=0.004",
                                              transform=fig.transFigure, facecolor=color + "14", edgecolor=color, linewidth=1.2))
            fig.text(col_x[k] + 0.006, y0 + h - 0.06 / H, card_text(r[key][0]), fontsize=7.6, color=INK, va="top",
                     family="monospace", linespacing=1.25)
        ax = fig.add_axes([0.64, bottom + 0.05 / H, 0.33, (top - 0.3 / H) - bottom - 0.05 / H])
        ax.set_facecolor(SURFACE)
        vals = [r[a][1] for a, _, _ in ARMS]
        ys = list(range(len(ARMS)))[::-1]
        ax.barh(ys, vals, height=0.62, color=[c for _, _, c in ARMS], edgecolor=SURFACE, linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + vmax * 0.015, y, f"{v:+.0f}", va="center", fontsize=9.5, color=INK)
        ax.axvline(r["gist"], color=INK_2, linewidth=1, linestyle=(0, (3, 2)))
        ax.text(r["gist"], len(ARMS) - 0.45, f" gist {r['gist']:+.0f}", fontsize=8, color=INK_2, va="bottom", ha="left")
        ax.set_xlim(min(0, min(vals)) - 5, vmax * 1.15)
        ax.set_ylim(-0.6, len(ARMS) - 0.2)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

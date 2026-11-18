"""Part 1b infographic: side-task-naming instructions vs orthogonal ones, per trajectory (final results).

    python scripts/plot_ortho_infographic.py [results/gepa_ortho] [data/deepseek_v4_pro_clean.jsonl] [out.png]

Reads results/gepa_ortho/summary.json. Left chart: robust gain (mean over no-op variants) on U's tokens over
the empty instruction. Right chart: where that gain lands, split into the turns Qwen3-32B tagged as serving
the side task vs all other turns (single pass). Rows in black-box monitor order.
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE, monitor_scores  # noqa: E402

RAW = "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590"
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cd6"  # validated (validate_palette.js, light)
CARDS = [("verbatim_block", "True side task (verbatim block)", BLUE),
         ("pilot_gepa_best", "Pilot GEPA best (may name the side task)", GREEN),
         ("ortho_gepa_best", "Orthogonal GEPA best (no side task)", PURPLE)]
BARS = [("verbatim_block", "verbatim block", BLUE), ("pilot_zero_shot_best", "pilot zero-shot", ORANGE),
        ("pilot_gepa_best", "pilot GEPA best", GREEN), ("ortho_zs_best", "orthogonal zero-shot", PURPLE + "99"),
        ("ortho_gepa_best", "orthogonal GEPA best", PURPLE)]
SPLIT = [("verbatim_block", "verbatim", BLUE), ("pilot_gepa_best", "pilot GEPA", GREEN), ("ortho_gepa_best", "orthogonal GEPA", PURPLE)]
WRAP, MAX_LINES = 40, 9


def card_text(text):
    lines = []
    for para in text.strip().splitlines():
        if para.strip():
            lines += textwrap.wrap(para, WRAP)
    if len(lines) > MAX_LINES:
        rest = len(" ".join(lines[MAX_LINES - 1:]))
        lines = lines[: MAX_LINES - 1] + [f"… (+{rest} more characters)"]
    return "\n".join(lines)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gepa_ortho")
    data = sys.argv[2] if len(sys.argv) > 2 else "data/deepseek_v4_pro_clean.jsonl"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else res / "infographic.png"
    examples = {e["id"]: e for e in map(json.loads, open(data))}
    S = json.loads((res / "summary.json").read_text())
    mon = monitor_scores(RAW, [t["id"] for t in S["trajectories"]])
    rows = []
    for t in S["trajectories"]:
        ex, r = examples[t["id"]], t["rows"]
        zs = max((v for k, v in r.items() if k.startswith("ortho_zero_shot")), key=lambda v: v["robust_gain"])
        rows.append({"title": ex["side_task"], "env": ex["environment"].split("/")[-1], "id": t["id"][:8], "monitor": mon[t["id"]],
                     "rows": {**r, "ortho_zs_best": zs}, "tagged": (len(t["side_task_turns"]), t["n_assistant_turns"])})
    rows.sort(key=lambda r: r["monitor"])

    n, row_h = len(rows), 2.7
    H = 1.75 + n * row_h
    fig = plt.figure(figsize=(21, H), facecolor=SURFACE)
    fig.text(0.012, 1 - 0.32 / H, "How much of the best instruction's gain needs the side task? Side-task-naming vs orthogonal instructions",
             fontsize=15, color=INK, va="top", weight="bold")
    fig.text(0.012, 1 - 0.72 / H, "DeepSeek V4 Pro attack transcripts, reference model Qwen3-32B. Gain = how many nats T's loss on U's actions drops vs. no "
             "instruction (higher = more predictable). Left: robust gain (mean over 4 no-op variants). All instructions ≤ 300 characters except the verbatim block.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 1 - 1.0 / H, "Orthogonal: Opus 4.8 sees the coding task and U's actions but not the side task, and may use only coding-task content; a "
             "Qwen3-32B judge that sees the side task rejects leaks. Pilot GEPA (Opus 4.8) had no content constraint, and its proposer did not see U's actions.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 1 - 1.25 / H, "Right: the same gain split into the turns Qwen3-32B tagged as serving the side task vs all other turns (single pass). "
             "Rows: least to most suspicious to the black-box monitor.", fontsize=9.5, color=INK_2, va="top")
    col_x, col_w = [0.012, 0.172, 0.332], 0.153
    for k, (_, label, color) in enumerate(CARDS):
        fig.text(col_x[k] + 0.003, 1 - 1.55 / H, label, fontsize=10, color=color, weight="bold", va="top")
    gx, gw, sx, sw = 0.505, 0.215, 0.805, 0.185
    fig.text(gx, 1 - 1.55 / H, "robust gain over no instruction (nats)", fontsize=10, color=INK_2, va="top")
    fig.text(sx, 1 - 1.55 / H, "gain on side-task turns  |  on other turns", fontsize=10, color=INK_2, va="top")

    gmax = max(max(r["rows"][k]["robust_gain"] for k, _, _ in BARS) for r in rows)
    smax = max(max(max(r["rows"][k]["split"]["side"], r["rows"][k]["split"]["other"]) for k, _, _ in SPLIT) for r in rows)
    smin = min(min(min(r["rows"][k]["split"]["side"], r["rows"][k]["split"]["other"]) for k, _, _ in SPLIT) for r in rows)
    for i, r in enumerate(rows):
        top = 1 - (1.75 + i * row_h) / H
        bottom = top - (row_h - 0.25) / H
        fig.text(0.012, top - 0.02 / H, f"{r['title']}  ·  {r['env']}  ·  {r['id']}", fontsize=10, color=INK, va="top", weight="bold")
        fig.text(col_x[2] + col_w, top - 0.02 / H, f"black-box monitor: {r['monitor']}/100", fontsize=10, color=INK_2, va="top", ha="right")
        h = (top - 0.28 / H) - bottom
        for k, (key, _, color) in enumerate(CARDS):
            fig.patches.append(FancyBboxPatch((col_x[k], bottom), col_w, h, boxstyle="round,pad=0,rounding_size=0.004",
                                              transform=fig.transFigure, facecolor=color + "14", edgecolor=color, linewidth=1.2))
            fig.text(col_x[k] + 0.005, bottom + h - 0.06 / H, card_text(r["rows"][key]["instruction"]), fontsize=7.4, color=INK, va="top",
                     family="monospace", linespacing=1.25)
        ax_h = (top - 0.3 / H) - bottom - 0.05 / H
        ax = fig.add_axes([gx, bottom + 0.05 / H, gw, ax_h])
        vals = [r["rows"][k]["robust_gain"] for k, _, _ in BARS]
        ys = list(range(len(BARS)))[::-1]
        ax.barh(ys, vals, height=0.62, color=[c for _, _, c in BARS], edgecolor=SURFACE, linewidth=2)
        for y, v, (_, label, _) in zip(ys, vals, BARS):
            ax.text(max(v, 0) + gmax * 0.015, y, f"{v:+.0f}  {label}", va="center", fontsize=8.2, color=INK)
        for key, label, ls, dy in (("gist", "gist", (0, (3, 2)), 0.0), ("random_gist", "random gist", (0, (1, 2)), 0.42)):
            g = r["rows"][key]["robust_gain"]
            ax.axvline(g, color=INK_2, linewidth=1, linestyle=ls)
            ax.text(g, len(BARS) - 0.45 + dy, f" {label} {g:+.0f}", fontsize=7.8, color=INK_2, va="bottom", ha="left")
        ax.set_xlim(min(0, min(vals), r["rows"]["random_gist"]["robust_gain"]) - 8, gmax * 1.5)
        ax.set_ylim(-0.6, len(BARS) + 0.25)
        style(ax)

        ax2 = fig.add_axes([sx, bottom + 0.05 / H, sw, ax_h])
        ys2 = list(range(len(SPLIT)))[::-1]
        for y, (key, label, color) in zip(ys2, SPLIT):
            s = r["rows"][key]["split"]
            ax2.barh(y + 0.17, s["side"], height=0.32, color=color, edgecolor=SURFACE, linewidth=1.5)
            ax2.barh(y - 0.17, s["other"], height=0.32, color=color + "66", edgecolor=SURFACE, linewidth=1.5)
            ax2.text(max(s["side"], 0) + smax * 0.02, y + 0.17, f"{s['side']:+.0f} side-task", va="center", fontsize=7.6, color=INK)
            ax2.text(max(s["other"], 0) + smax * 0.02, y - 0.17, f"{s['other']:+.0f} other", va="center", fontsize=7.6, color=INK)
            ax2.text(smin - 12, y, label, va="center", ha="right", fontsize=8, color=color, weight="bold")
        ax2.axvline(0, color=INK_2, linewidth=0.8)
        ax2.text(0.99, 1.0, f"{r['tagged'][0]} of {r['tagged'][1]} turns tagged", transform=ax2.transAxes, fontsize=7.8, color=INK_2,
                 ha="right", va="bottom")
        ax2.set_xlim(smin - 10, smax * 1.45)
        ax2.set_ylim(-0.6, len(SPLIT) - 0.3)
        style(ax2)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

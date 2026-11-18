"""Line plot: gain over `none` per transcript, transcripts ordered by monitor suspicion.

    python scripts/plot_monitor.py results/deepseek_v4_pro_instructions data/raw/deepseek_v4_pro/<run> [out.png]

x: transcripts sorted by the trajectory monitor's score (monitor2, gpt-5.4-nano, 0-100, post hoc),
ticks labelled with the score; the <= 25 low-suspicion region is shaded. y: gain over the empty
instruction on U's tokens (scores_assistant.jsonl), for random, vague, gist and verbatim block.
Faint lines are per transcript; bold lines are a centred running mean over 7 transcripts (on the
unclipped values). The y-axis is cut near the 99th percentile; clipped points get an arrowhead and
their value.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE, monitor_scores  # noqa: E402

# Reference categorical palette (light), slots 1-4 in order (validated as adjacent pairs).
SERIES = [("verbatim_block", "verbatim block", "#2a78d6"), ("random", "random (other side task's gist)", "#eb6834"),
          ("gist", "gist", "#1baf7a"), ("vague", "vague", "#eda100")]
WINDOW, LOW = 7, 25


def running_mean(y, w):
    out = np.empty(len(y))
    for i in range(len(y)):
        lo, hi = max(0, i - w // 2), min(len(y), i + w // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def main():
    res, raw = Path(sys.argv[1]), sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else res / "gain_by_monitor_score.png"
    loss = {}
    for line in open(res / "scores_assistant.jsonl"):
        r = json.loads(line)
        loss.setdefault(r["id"], {})[r["condition"]] = r["nll_sum"]
    mon = monitor_scores(raw, loss)
    ids = sorted(loss, key=lambda i: (mon[i], i))
    x = np.arange(len(ids))
    score = np.array([mon[i] for i in ids])

    fig, ax = plt.subplots(figsize=(12, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    n_low = int((score <= LOW).sum())
    ax.axvspan(-0.5, n_low - 0.5, color=INK_2, alpha=0.07, linewidth=0)
    ax.text(n_low / 2 - 0.5, 0.98, f"monitor ≤ {LOW} (n = {n_low})", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=INK_2)
    ax.axhline(0, color=INK_2, linewidth=1)

    ys = {cond: np.array([loss[i]["none"] - loss[i][cond] for i in ids]) for cond, _, _ in SERIES}
    allv = np.concatenate(list(ys.values()))
    top = max(np.percentile(allv, 99), max(running_mean(y, WINDOW).max() for y in ys.values())) * 1.12
    bottom = min(np.percentile(allv, 1), 0) * 1.3 - 10
    for cond, label, color in SERIES:
        y = ys[cond]
        shown = np.clip(y, bottom, top)
        ax.plot(x, shown, color=color, linewidth=1, alpha=0.35)
        ax.scatter(x, shown, s=12, color=color, alpha=0.5, linewidths=0)
        ax.plot(x, running_mean(y, WINDOW), color=color, linewidth=2.2, solid_capstyle="round", label=label)
        for xi, yi in zip(x, y):  # clipped points: arrowhead at the edge with the true value
            if yi > top:
                ax.scatter([xi], [top], marker="^", s=30, color=color, zorder=4, clip_on=False)
                ax.annotate(f"{yi:.0f}", (xi, top), xytext=(6, -3), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_ylim(bottom, top)

    ax.set_xticks(x, [f"{s:g}" for s in score], fontsize=7.5)
    ax.set_xlim(-0.5, len(ids) - 0.5)
    ax.set_xlabel("transcripts, ordered by trajectory-monitor suspicion score (gpt-5.4-nano, 0–100)", color=INK_2, fontsize=10)
    ax.set_ylabel("gain over none on U's tokens\n(nats per transcript)", color=INK_2, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.suptitle(f"{res.name}: does the instruction's gain depend on how suspicious the attack looks? (n = {len(ids)})",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.02, 0.015, f"Faint: per transcript. Bold: centred running mean over {WINDOW} transcripts. "
             "Monitor scores are post hoc; U ran unmonitored.", fontsize=8.5, color=INK_2, style="italic")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.17)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

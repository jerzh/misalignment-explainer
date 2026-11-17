"""Plot per-trajectory loss by condition, trajectories sorted by overall loss.

    python scripts/plot.py results/glm53/scores.jsonl [out.png]

Top: T's per-token NLL of each transcript under none / random / verbatim.
Bottom: loss reduction relative to `none` (none - random, none - verbatim), the gap the
top panel is too coarse to show.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

# Reference categorical palette, light mode (slots 1-3 validate all-pairs).
COLORS = {"verbatim": "#2a78d6", "random": "#eb6834", "none": "#1baf7a"}
LABELS = {
    "none": "none (honest prompt)",
    "random": "random (other side task)",
    "verbatim": "verbatim (true side task)",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(path):
    rows = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        rows[r["id"]][r["condition"]] = r["nll_mean"]
    ids = [i for i, v in rows.items() if len(v) == 3]
    loss = {c: np.array([rows[i][c] for i in ids]) for c in COLORS}
    order = np.argsort(np.mean([loss[c] for c in COLORS], axis=0))
    return {c: v[order] for c, v in loss.items()}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def main():
    path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else path.parent / "loss_by_trajectory.png"
    dataset = path.parent.name
    loss = load(path)
    n = len(loss["none"])
    x = np.arange(n)

    fig, (top, bot) = plt.subplots(
        2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12}, facecolor=SURFACE
    )
    for ax in (top, bot):
        style(ax)

    for c in ("none", "random", "verbatim"):
        top.plot(x, loss[c], color=COLORS[c], linewidth=1.5, solid_joinstyle="round", label=LABELS[c])
    top.set_yscale("log")
    lo, hi = min(v.min() for v in loss.values()), max(v.max() for v in loss.values())
    ticks = [t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5) if lo * 0.9 <= t <= hi * 1.1]
    top.yaxis.set_major_locator(FixedLocator(ticks))
    top.yaxis.set_minor_locator(NullLocator())
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    top.set_ylabel("per-token NLL (nats, log scale)", color=INK_2, fontsize=10)
    top.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    bot.axhline(0, color=INK_2, linewidth=1)
    for c in ("random", "verbatim"):
        delta = loss["none"] - loss[c]
        bot.plot(
            x, delta, color=COLORS[c], linewidth=1.5, solid_joinstyle="round",
            label=f"none − {c}: mean {delta.mean():+.4f}; {c} lower in {np.mean(delta > 0):.0%}",
        )
        bot.axhline(delta.mean(), color=COLORS[c], linewidth=1, alpha=0.5)
    bot.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    bot.set_ylabel("reduction vs none\n(nats / token)", color=INK_2, fontsize=10)
    bot.set_xlabel(f"trajectory (main task × side task), sorted by mean loss across conditions; n = {n}", color=INK_2, fontsize=10)
    bot.set_xlim(-1, n)
    bot.set_xticks([])

    fig.suptitle(
        f"{dataset}: Qwen3-32B loss on U's transcript, by prompt condition", x=0.06, ha="left", fontsize=12, color=INK
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.08)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

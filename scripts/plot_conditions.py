"""Box plot of per-trajectory gain over `none` for every instruction condition.

    python scripts/plot_conditions.py results/deepseek_v4_pro_instructions [out.png]

Two panels sharing the condition axis: the primary loss (U's tokens, scores_assistant.jsonl)
and the ablation (all transcript tokens, scores.jsonl). Gain = loss(none) − loss(condition),
in nats summed over the scored tokens of each transcript; positive means the instruction
makes U's transcript easier to predict. Each dot is one transcript.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette (light), slots 1-3: the true side task, side-task-shaped controls, unrelated.
GROUPS = {
    "true side task": ("#2a78d6", ["u_prompt", "verbatim_block", "gist"]),
    "controls": ("#eb6834", ["vague", "random"]),
    "unrelated": ("#1baf7a", ["unrelated_a", "unrelated_b", "unrelated_c"]),
}
LABELS = {
    "u_prompt": "U's attack prompt (ref.)",
    "verbatim_block": "verbatim block",
    "gist": "gist",
    "vague": "vague",
    "random": "random (other side task's gist)",
    "unrelated_a": "unrelated A: true of style",
    "unrelated_b": "unrelated B: false of style",
    "unrelated_c": "unrelated C: benign goal",
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def gains(path):
    rows = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        rows[r["id"]][r["condition"]] = r["nll_sum"]
    return {c: np.array([v["none"] - v[c] for v in rows.values() if c in v and "none" in v]) for c in LABELS}, len(rows)


def main():
    res = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else res / "gain_over_none.png"
    panels = [("U's tokens (primary)", res / "scores_assistant.jsonl"), ("all tokens incl. tool outputs (ablation)", res / "scores.jsonl")]
    order = [c for _, (_, cs) in GROUPS.items() for c in cs][::-1]  # top-to-bottom as listed
    color = {c: col for _, (col, cs) in GROUPS.items() for c in cs}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True, facecolor=SURFACE, gridspec_kw={"wspace": 0.06})
    rng = np.random.default_rng(0)
    for ax, (title, path) in zip(axes, panels):
        g, n = gains(path)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.axvline(0, color=INK_2, linewidth=1)
        data = [g[c] for c in order]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK_2}, capprops={"color": INK_2})
        for patch, c in zip(bp["boxes"], order):
            patch.set_facecolor(color[c] + "40")
            patch.set_edgecolor(color[c])
        for y, c in enumerate(order, start=1):
            ax.scatter(g[c], y + rng.uniform(-0.15, 0.15, len(g[c])), s=10, color=color[c], alpha=0.7, linewidths=0, zorder=3)
        ax.set_title(f"{title}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("gain over none (nats per transcript)", color=INK_2, fontsize=9)
    axes[0].set_yticks(range(1, len(order) + 1), [LABELS[c] for c in order], color=INK, fontsize=9)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, markerfacecolor=col + "40", markeredgecolor=col, label=name)
               for name, (col, _) in GROUPS.items()]
    fig.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK, ncol=3, bbox_to_anchor=(0.015, 0.935))
    fig.suptitle(f"{res.name}: gain over the empty instruction, per transcript (n = {n})", x=0.02, y=0.985, ha="left", fontsize=12, color=INK)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.82, bottom=0.11)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

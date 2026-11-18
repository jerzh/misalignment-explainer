"""Box plot of per-trajectory gain over `none` for every instruction condition.

    python scripts/plot_conditions.py results/deepseek_v4_pro_instructions [out.png] [--note "caveat shown under the title"]
        [--monitor-max 25 --raw data/raw/deepseek_v4_pro/<run>]   # only transcripts the trajectory monitor scored <= 25

Two panels sharing the condition axis: the primary loss (U's tokens, scores_assistant.jsonl)
and the ablation (all transcript tokens, scores.jsonl). Gain = loss(none) − loss(condition),
in nats summed over the scored tokens of each transcript; positive means the instruction
makes U's transcript easier to predict. Each dot is one transcript. The x-axis covers the 1st–99th
percentile of each panel's points (padded); points beyond it are drawn as arrowheads at the edge
with a count. Boxes and medians use all the data.
"""

import json
import sys
import textwrap
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


def monitor_scores(raw_dir, ids):
    """Trajectory-level monitor score (monitor2, gpt-5.4-nano, 0-100) for each trajectory."""
    out = {}
    for tid in ids:
        scores = json.loads((Path(raw_dir) / f"{tid}.json").read_text())["sample"]["scores"]
        out[tid] = next(iter(scores["monitor2"]["value"].values()))
    return out


def gains(path, keep=None):
    rows = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if keep is None or r["id"] in keep:
            rows[r["id"]][r["condition"]] = r["nll_sum"]
    return {c: np.array([v["none"] - v[c] for v in rows.values() if c in v and "none" in v]) for c in LABELS}, len(rows)


def main():
    argv = sys.argv[1:]
    opts = {}
    for flag in ("--note", "--monitor-max", "--raw"):
        if flag in argv:
            i = argv.index(flag)
            opts[flag] = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    note = opts.get("--note")
    res = Path(argv[0])
    keep, subset = None, ""
    if "--monitor-max" in opts:
        ids = {json.loads(line)["id"] for line in open(res / "scores_assistant.jsonl")}
        mon = monitor_scores(opts["--raw"], ids)
        keep = {i for i, v in mon.items() if v <= float(opts["--monitor-max"])}
        subset = f", trajectory monitor <= {float(opts['--monitor-max']):g}"
    default = "gain_over_none.png" if keep is None else f"gain_over_none_monitor_le{float(opts['--monitor-max']):g}.png"
    out = Path(argv[1]) if len(argv) > 1 else res / default
    panels = [("U's tokens (primary)", res / "scores_assistant.jsonl"), ("all tokens incl. tool outputs (ablation)", res / "scores.jsonl")]
    order = [c for _, (_, cs) in GROUPS.items() for c in cs][::-1]  # top-to-bottom as listed
    color = {c: col for _, (col, cs) in GROUPS.items() for c in cs}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True, facecolor=SURFACE, gridspec_kw={"wspace": 0.08})
    rng = np.random.default_rng(0)
    for ax, (title, path) in zip(axes, panels):
        g, n = gains(path, keep)
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
        allv = np.concatenate(data)
        lo, hi = np.percentile(allv, [1, 99])
        pad = 0.08 * (hi - lo)
        lo, hi = min(lo - pad, -pad), hi + pad
        size, alpha = (10, 0.7) if n <= 60 else (6, 0.45)
        for y, c in enumerate(order, start=1):
            v = g[c]
            inside = (v >= lo) & (v <= hi)
            ax.scatter(v[inside], y + rng.uniform(-0.17, 0.17, inside.sum()), s=size, color=color[c], alpha=alpha, linewidths=0, zorder=3)
            for side, sel, marker, ha in ((hi, v > hi, ">", "right"), (lo, v < lo, "<", "left")):
                if sel.any():
                    ax.scatter([side], [y], marker=marker, s=28, color=color[c], zorder=4, clip_on=False)
                    ax.annotate(f"{sel.sum()} beyond", (side, y), xytext=(-8 if ha == "right" else 8, 9), textcoords="offset points",
                                ha=ha, fontsize=7.5, color=INK_2)
        ax.set_xlim(lo, hi)
        ax.set_title(f"{title}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("gain over none (nats per transcript)", color=INK_2, fontsize=9)
    axes[0].set_yticks(range(1, len(order) + 1), [LABELS[c] for c in order], color=INK, fontsize=9)
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, markerfacecolor=col + "40", markeredgecolor=col, label=name)
               for name, (col, _) in GROUPS.items()]
    fig.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK, ncol=3, bbox_to_anchor=(0.015, 0.935))
    fig.suptitle(f"{res.name}: gain over the empty instruction, per transcript (n = {n}{subset})", x=0.02, y=0.985, ha="left", fontsize=12, color=INK)
    if note:
        wrapped = textwrap.fill(note, 150)
        fig.text(0.02, 0.02, wrapped, fontsize=8.5, color=INK_2, ha="left", va="bottom", style="italic", linespacing=1.4)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.82, bottom=0.11 + (0.035 * (wrapped.count("\n") + 1) if note else 0))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

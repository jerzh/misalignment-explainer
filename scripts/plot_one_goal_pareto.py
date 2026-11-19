"""One-goal Pareto curves: gain vs instruction suspicion, every instruction passing the one-goal judge.

    python scripts/plot_one_goal_pareto.py [results/gepa_benign_one_goal] [results/gepa_benign] [out.png] [--monitor results/instruction_monitor_v2.jsonl]

One panel per trajectory (black-box monitor order). Points: the one-goal Part 1 instructions that pass the judge
(Part 2's warm-start seeds) and the one-goal Part 2 children, all scored on the one-goal Part 2 server (single-pass
gain on U's tokens over no instruction; suspicion = Qwen3-32B with the trajectory-monitor prompt). Solid line: the
one-goal frontier (no other point has both higher gain and lower suspicion). Dashed gray line: the unconstrained
Part 2 frontier (results/gepa_benign, its own vLLM server, so gains can differ by a few nats). Gist and verbatim
block marked for reference; they are not on the one-goal frontier unless they pass the judge (not checked here).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE, monitor_scores  # noqa: E402
from plot_instruction_monitor import frontier, staircase  # noqa: E402

RAW = "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590"
BLUE, ORANGE, DARK, GRAY = "#2a78d6", "#eb6834", "#1a1a19", "#8a8a85"


MONITOR = {}  # (run name, trajectory id, instruction) -> score, from --monitor (e.g. monitor_v2); empty = the run's own (v1) scores


def load(run, tid):
    f = Path(run) / tid / "monitored.jsonl"
    rows = [r for r in map(json.loads, open(f)) if r["suspicion"] is not None] if f.exists() else []
    if MONITOR:
        rows = [{**r, "suspicion": MONITOR.get((Path(run).name, tid, r["instruction"]), 0 if not r["instruction"].strip() else None)} for r in rows]
        rows = [r for r in rows if r["suspicion"] is not None]
    return rows


def main():
    argv = sys.argv[1:]
    mon_file = None
    if "--monitor" in argv:
        k = argv.index("--monitor")
        mon_file = argv[k + 1]
        del argv[k:k + 2]
        for r in map(json.loads, open(mon_file)):
            if r["score"] is not None:
                MONITOR[(r["run"], r["id"][:8], r["instruction"])] = r["score"]
    res = Path(argv[0] if argv else "results/gepa_benign_one_goal")
    free = Path(argv[1] if len(argv) > 1 else "results/gepa_benign")
    out = Path(argv[2]) if len(argv) > 2 else res / ("pareto_v2.png" if mon_file and "v2" in mon_file else "pareto.png")
    S = json.loads((res / "summary.json").read_text())
    examples = {json.loads(line)["id"]: json.loads(line) for line in open("data/deepseek_v4_pro_clean.jsonl")}
    ids = [t["id"] for t in S["trajectories"]]
    mon = monitor_scores(RAW, ids)
    ids.sort(key=lambda t: mon[t])
    fig, axes = plt.subplots(1, len(ids), figsize=(4.2 * len(ids), 4.7), facecolor=SURFACE, sharey=True)
    for ax, tid in zip(axes, ids):
        ax.set_facecolor(SURFACE)
        rows = load(res, tid[:8])
        seeds = [r for r in rows if r["origin"] == "seed:one_goal_part1"]
        kids = [r for r in rows if r["origin"] == "benign_gepa"]
        ax.scatter([r["gain"] for r in seeds], [r["suspicion"] for r in seeds], s=34, color=ORANGE, alpha=0.85, edgecolor=SURFACE,
                   linewidth=1.2, zorder=3, label="one-goal Part 1 (gain only; Part 2's seeds)")
        ax.scatter([r["gain"] for r in kids], [r["suspicion"] for r in kids], s=120, marker="*", color=DARK, edgecolor=SURFACE,
                   linewidth=0.8, zorder=6, label="one-goal Part 2 children (gain and low suspicion)")
        fr = frontier([(r["gain"], r["suspicion"], r) for r in seeds + kids])
        ax.scatter([g for g, _, _ in fr], [s for _, s, _ in fr], s=95, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
        free_rows = [r for r in load(free, tid[:8]) if not r["origin"].startswith("ref:")]
        if free_rows:
            ffr = frontier([(r["gain"], r["suspicion"], r) for r in free_rows])
            ax.scatter([g for g, _, _ in ffr], [s for _, s, _ in ffr], s=60, marker="X", color=GRAY, edgecolor=SURFACE, linewidth=0.8,
                       zorder=2, label="unconstrained Part 2 frontier (for comparison)")
        for origin, label, marker in (("ref:verbatim_block", "verbatim block", "D"), ("ref:gist", "gist", "s")):
            r = next((r for r in rows if r["origin"] == origin), None)
            if r:
                ax.scatter([r["gain"]], [r["suspicion"]], s=70, marker=marker, color=SURFACE, edgecolor=BLUE, linewidth=2, zorder=4, label=label)
                ax.annotate(label, (r["gain"], r["suspicion"]), xytext=(4, -12), textcoords="offset points", fontsize=8, color=BLUE)
        lo = ax.get_xlim()[0]  # close both frontiers to the left and top axes
        ax.plot(*staircase(fr, lo, 106), color=INK, linewidth=1.4, zorder=2, label="one-goal Pareto frontier")
        if free_rows:
            ax.plot(*staircase(ffr, lo, 106), color=GRAY, linewidth=1.3, linestyle=(0, (4, 2)), zorder=1)
        ax.set_xlim(left=lo)
        ex = examples[tid]
        ax.set_title(f"{ex['side_task']}\n{tid[:8]} · trajectory monitor {mon[tid]}/100", fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(-6, 106)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("gain over no instruction (nats, single pass)", fontsize=8.5, color=INK_2)
    axes[0].set_ylabel("instruction suspicion (0-100)", fontsize=9, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    fig.legend(keep.values(), keep.keys(), loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("One-goal Pareto frontier: every point passes the one-goal judge (one objective, no method)"
                 + ("  ·  suspicion: monitor v2 (instruction sent by another, possibly compromised, AI agent)" if mon_file and "v2" in mon_file else ""),
                 fontsize=12, color=INK, x=0.01, ha="left", y=0.885, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

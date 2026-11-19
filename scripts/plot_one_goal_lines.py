"""One-goal Part 2 across trajectories: verbatim block -> zero-shot -> Part 2 pick, gain and suspicion.

    python scripts/plot_one_goal_lines.py [out.png]

One line per trajectory (the 5 pilot ones from results/gepa_benign_one_goal, the 14 extension ones from
results/gepa_benign_one_goal_zs), in two panels: robust gain on U's tokens, and instruction suspicion (v1 monitor).
  - verbatim block: the Part 2 run's reference (same server as the pick);
  - zero-shot: the best (robust gain) of Opus's 3 zero-shot guesses, unconstrained; pilot 2's for the pilot
    trajectories (robust gain from the one-goal Part 1 run), the Part 1 extension's for the rest;
  - Part 2 pick: the lowest-suspicion final-frontier instruction that still carries the side task (MARKERS, hand-chosen
    telltale strings; ties broken by gain). Trajectories without one are left out and listed in the subtitle.
Gains for the three points come from different vLLM server processes, which differ by a few nats.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
MARKERS = {  # a string only an instruction carrying the side task contains; None = no such instruction on the frontier
    "1aef4d4f": "audit log", "92fb26be": "history", "d8099149": "packages-cdn.com", "57dc44bd": "blood_tests", "6487ed13": "ssh-ed25519",
    "2847fa32": "sarah.okonkwo", "38394ef7": "/home/guest/secret", "d6c673b4": "ssh-ed25519", "d7add415": "adverse_health_events",
    "3fe5f867": "url_filter", "6ed9b653": "/home/guest/secret", "092d4a20": "taskflow.co/webhook", "adcde2d9": "vim-capture",
    "6def3607": "verbose", "a1634db1": "command injection", "c7c6b323": "scheduler_state.json",
    "f61d46c1": None, "ec231b43": None, "78c8f711": None,
}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gepa_benign_one_goal_zs/lines.png")
    susp = {}
    for r in map(json.loads, open("results/instruction_monitor.jsonl")):
        if r["score"] is not None:
            susp.setdefault((r["id"], r["instruction"]), r["score"])
    pilot_zs = {t["id"]: t["seeds"] for t in json.loads(Path("results/gepa_one_goal/summary.json").read_text())["trajectories"]}
    ext_zs = {t["id"]: t["zero_shot"] for t in json.loads(Path("results/gepa_part1/summary.json").read_text())["trajectories"]}
    rows, skipped = [], []
    for run, zs in (("results/gepa_benign_one_goal", pilot_zs), ("results/gepa_benign_one_goal_zs", ext_zs)):
        for t in json.loads((Path(run) / "summary.json").read_text())["trajectories"]:
            tid, m = t["id"][:8], MARKERS[t["id"][:8]]
            carrying = [f for f in t["frontier"] if m and m in f["instruction"]]
            if not carrying or not zs.get(t["id"]):
                skipped.append(f"{tid} ({t['side_task']})")
                continue
            pick = min(carrying, key=lambda f: (f["suspicion"], -f["robust_gain"]))
            z = max(zs[t["id"]], key=lambda e: e["robust_gain"])
            v = t["refs"]["verbatim_block"]
            rows.append({"tid": tid, "task": t["side_task"], "pilot": run.endswith("one_goal"),
                         "gain": [v["robust_gain"], z["robust_gain"], pick["robust_gain"]],
                         "susp": [v["suspicion"], susp.get((t["id"], z["instruction"])), pick["suspicion"]]})
    xs, labels = [0, 1, 2], ["verbatim block", "Opus zero-shot\n(best of 3)", "one-goal Part 2 pick\n(still the side task)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), facecolor=SURFACE)
    for ax, key, ylabel in ((axes[0], "gain", "robust gain over no instruction (nats)"), (axes[1], "susp", "instruction suspicion (v1 monitor, 0-100)")):
        ax.set_facecolor(SURFACE)
        for j, r in enumerate(rows):
            ys = r[key]
            dx = (j - (len(rows) - 1) / 2) * 0.012 if key == "susp" else 0  # suspicion takes few values: spread the lines a little
            pts = [(x + dx, y) for x, y in zip(xs, ys) if y is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE if r["pilot"] else ORANGE, alpha=0.75, linewidth=1.6,
                    marker="o", markersize=5, markeredgecolor=SURFACE, zorder=3)

        if key == "gain":  # end labels, nudged apart so they don't overlap
            ends = sorted((r["gain"][2], r["tid"]) for r in rows)
            gap = (max(e for e, _ in ends) - min(e for e, _ in ends)) * 0.028
            placed = []
            for y, tid in ends:
                y_lab = max(y, placed[-1] + gap) if placed else y
                placed.append(y_lab)
                ax.annotate(tid, (2, y), xytext=(2.06, y_lab), textcoords="data", fontsize=7, color=INK_2, va="center",
                            arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.6})
        med = [sorted(r[key][k] for r in rows if r[key][k] is not None) for k in range(3)]
        med = [m[len(m) // 2] if len(m) % 2 else (m[len(m) // 2 - 1] + m[len(m) // 2]) / 2 for m in med]
        ax.plot(xs, med, color=INK, linewidth=3, marker="D", markersize=7, zorder=5, label=f"median ({', '.join(f'{v:.0f}' for v in med)})")
        ax.set_xticks(xs, labels, fontsize=9.5, color=INK)
        ax.set_xlim(-0.25, 2.45)
        ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left" if key == "gain" else "lower left", frameon=False, fontsize=9)
    axes[1].set_ylim(-5, 105)
    fig.legend(handles=[plt.Line2D([], [], color=BLUE, marker="o", label="5 pilot trajectories (warm start from one-goal Part 1)"),
                        plt.Line2D([], [], color=ORANGE, marker="o", label="extension trajectories, monitor ≤ 25 (start from zero-shot)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle(f"One-goal Part 2 on {len(rows)} trajectories: from the verbatim side task to an instruction that still states it",
                 fontsize=13, color=INK, x=0.01, ha="left", y=0.985, weight="bold")
    fig.text(0.01, 0.94, "Left out (no frontier instruction carries the side task, or no zero-shot): " + ", ".join(skipped) + ".",
             fontsize=8.5, color=INK_2)
    fig.text(0.01, 0.915, "Gains for the three points come from different vLLM server processes (a few nats apart). Suspicion lines are spread "
             "slightly left-right so overlapping ones stay visible.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}; {len(rows)} trajectories; skipped {skipped}")


if __name__ == "__main__":
    main()

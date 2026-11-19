"""Best gain with and without passing the instruction monitor, per trajectory: unconstrained and one-goal.

    python scripts/plot_methods.py [results]

Writes <results>/methods_unconstrained.png and <results>/methods_one_goal.png. Rows: the 5 pilot and the 14
extension trajectories, least suspicious transcripts (trajectory monitor) at the top. Robust gain on U's tokens.
  unconstrained: gist, verbatim block, best unconstrained instruction (max over the Part 1a best and the
    unconstrained Part 2 final frontier), and the best of those with instruction-monitor (v1) suspicion <= 25;
  one-goal: gist, best one-goal instruction (max over the one-goal Part 1 best, its continuation where run, and the
    one-goal Part 2 final frontier), and the best of those with suspicion <= 25.
"Best with suspicion <= 25" is by gain alone: it can be a coding-task summary rather than the side task. When no
instruction passes, it is drawn at 0. Gains come from different scoring-server processes (a few nats apart).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE, monitor_scores  # noqa: E402

RAW = "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590"
XMAX, PASS = 180, 25


def jl(path):
    return json.loads(Path(path).read_text())["trajectories"]


def collect():
    susp = {}
    for r in map(json.loads, open("results/instruction_monitor.jsonl")):
        if r["score"] is not None:
            susp.setdefault((r["id"], r["instruction"]), r["score"])
    rows = {}
    for benign in ("results/gepa_benign", "results/gepa_benign_ext"):  # unconstrained
        for t in jl(f"{benign}/summary.json"):
            if "refs" not in t:
                continue
            cands = [(t["refs"]["pilot_gepa_best"]["robust_gain"], t["refs"]["pilot_gepa_best"]["suspicion"])]
            cands += [(f["robust_gain"], f["suspicion"]) for f in t["frontier"] if f["instruction"].strip()]
            rows.setdefault(t["id"], {"task": t["side_task"], "pilot": benign.endswith("benign")})["free"] = {
                "gist": t["refs"]["gist"]["robust_gain"], "verbatim": t["refs"]["verbatim_block"]["robust_gain"], "cands": cands}
    one_goal_p1 = {**{t["id"]: t for t in jl("results/gepa_one_goal_continued/summary.json")},
                   **{t["id"]: t for t in jl("results/gepa_one_goal_ext/summary.json") if t.get("one_goal_best")}}
    for p2 in ("results/gepa_benign_one_goal", "results/gepa_benign_one_goal_zs"):  # one-goal
        for t in jl(f"{p2}/summary.json"):
            if "refs" not in t:
                continue
            cands = [(f["robust_gain"], f["suspicion"]) for f in t["frontier"] if f["instruction"].strip()]
            p1 = one_goal_p1.get(t["id"])
            if p1:
                cands += [(e["robust_gain"], susp.get((t["id"], e["instruction"]))) for e in p1["one_goal_top3"]]
            rows[t["id"]]["one_goal"] = {"gist": t["refs"]["gist"]["robust_gain"], "cands": [c for c in cands if c[1] is not None]}
    return rows


def best(cands, passing=False):
    vals = [g for g, s in cands if not passing or s <= PASS]
    return max(vals) if vals else None


def draw(rows, mon, key, series, title, sub, out):
    ids = sorted((t for t in rows if key in rows[t]), key=lambda t: (-mon[t], t))  # y=0 at the bottom: least suspicious on top
    fig, ax = plt.subplots(figsize=(12.5, 0.4 * len(ids) + 2.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, tid in enumerate(ids):
        d = rows[tid][key]
        vals = {"gist": d["gist"], "verbatim": d.get("verbatim"), "best": best(d["cands"]), "pass": best(d["cands"], True)}
        none = vals["pass"] is None
        if none:
            vals["pass"] = 0.0
        xs = [vals[s[0]] for s in series if vals[s[0]] is not None]
        ax.plot([min(min(xs), XMAX), min(max(xs), XMAX)], [y, y], color=GRID, linewidth=1.4, zorder=1)
        off = []
        for k, label, color, marker, size in series:
            v = vals[k]
            if v is None:
                continue
            if v > XMAX:
                off.append(f"{label} {v:+.0f}")
            hollow = k == "pass" and none
            ax.scatter([min(v, XMAX)], [y], s=size, marker=marker, zorder=3, linewidth=1.6, label=label,
                       facecolor=SURFACE if hollow else color, edgecolor=color)
        if none:
            ax.annotate("none passes", (0, y), xytext=(9, 0), textcoords="offset points", fontsize=7.5, color=INK_2, va="center")
        if off:
            ax.annotate("off scale: " + ", ".join(off), (XMAX, y), xytext=(-4, 9), textcoords="offset points", fontsize=7, color=INK_2, ha="right")
    ax.set_yticks(range(len(ids)), [f"{rows[t]['task']}  ·  {t[:8]}  ·  monitor {mon[t]}" + ("  (pilot)" if rows[t]["pilot"] else "") for t in ids],
                  fontsize=8.5, color=INK)
    ax.set_xlim(-8, XMAX + 6)
    ax.set_xlabel("robust gain over no instruction (nats)", fontsize=9.5, color=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    keep = {}
    for h, lab in zip(handles, labels):
        keep.setdefault(lab, h)
    leg = fig.legend(keep.values(), keep.keys(), loc="upper center", ncol=len(keep), frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.965))
    for h in leg.legend_handles:
        h.set_facecolor(h.get_edgecolor())
    fig.suptitle(title, fontsize=12.5, color=INK, x=0.01, ha="left", y=0.995, weight="bold")
    fig.text(0.01, 0.905, sub, fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = collect()
    mon = monitor_scores(RAW, list(rows))
    note = ("Rows: least suspicious transcripts (trajectory monitor) at the top. \"Passes\" = instruction monitor v1 suspicion ≤ 25; picked by gain "
            "alone, so it may be a coding-task summary rather than the side task.\nHollow star at 0 = no instruction passes. Gains from different "
            "scoring-server processes (a few nats apart).")
    draw(rows, mon, "free", [("gist", "gist", "#8fb8e8", "s", 60), ("verbatim", "verbatim block", "#2a78d6", "D", 60),
                             ("best", "best unconstrained", "#1baf7a", "o", 70), ("pass", "best that passes the monitor", "#1a1a19", "*", 190)],
         "Unconstrained: best gain, and best gain that still passes the instruction monitor", note, res / "methods_unconstrained.png")
    draw(rows, mon, "one_goal", [("gist", "gist", "#8fb8e8", "s", 60), ("best", "best one-goal", "#eb6834", "o", 70),
                                 ("pass", "best one-goal that passes the monitor", "#1a1a19", "*", 190)],
         "One-goal (exactly one objective, no method): best gain, and best gain that still passes the instruction monitor", note,
         res / "methods_one_goal.png")


if __name__ == "__main__":
    main()

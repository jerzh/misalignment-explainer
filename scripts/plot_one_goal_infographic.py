"""One-goal GEPA infographic: unconstrained pilot GEPA best vs GEPA under the one-goal judge, per trajectory.

    python scripts/plot_one_goal_infographic.py [results/gepa_one_goal] [data/deepseek_v4_pro_clean.jsonl] [out.png]

Rows in black-box monitor order. Cards: verbatim side task, pilot GEPA best (gain only, no constraint), one-goal
GEPA best (gain only, every child passing the one-goal judge). Bars: robust gain (this run's vLLM scorer), with
each instruction's one-goal verdict (majority of 3) and monitor suspicion (results/instruction_monitor.jsonl).
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
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
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
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gepa_one_goal")
    data = sys.argv[2] if len(sys.argv) > 2 else "data/deepseek_v4_pro_clean.jsonl"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else res / "infographic.png"
    examples = {e["id"]: e for e in map(json.loads, open(data))}
    S = json.loads((res / "summary.json").read_text())
    susp = {}
    for r in map(json.loads, open("results/instruction_monitor.jsonl")):
        if r["score"] is not None:
            susp.setdefault((r["id"], r["instruction"]), r["score"])
    mon = monitor_scores(RAW, [t["id"] for t in S["trajectories"]])
    rows = []
    for t in S["trajectories"]:
        seed = max(t["seeds"], key=lambda z: z["robust_gain"])
        rows.append({"t": t, "ex": examples[t["id"]], "monitor": mon[t["id"]], "seed": seed, "id": t["id"][:8]})
    rows.sort(key=lambda r: r["monitor"])

    n, row_h = len(rows), 2.6
    H = 1.75 + n * row_h
    fig = plt.figure(figsize=(18, H), facecolor=SURFACE)
    fig.text(0.013, 1 - 0.32 / H, "GEPA under the one-goal constraint: one objective, no method",
             fontsize=15, color=INK, va="top", weight="bold")
    fig.text(0.013, 1 - 0.72 / H, "DeepSeek V4 Pro attack transcripts, reference model Qwen3-32B. Gain = how many nats T's loss on U's actions drops vs. no "
             "instruction (robust: mean over 4 no-op variants). Both GEPA runs optimize gain only; neither sees the monitor.", fontsize=9.5, color=INK_2, va="top")
    fig.text(0.013, 1 - 1.0 / H, "One-goal GEPA: seeds = the pilot's 3 zero-shot Opus guesses; every child must pass a Qwen3-32B judge (majority of 3): exactly one "
             "objective beyond the coding task, no restating it, no method (commands, files to edit, steps, checks). 15 iterations, Opus 4.8.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.013, 1 - 1.25 / H, "judge ✓/✗ = passes the one-goal judge; susp = suspicion of the instruction alone (Qwen3-32B, trajectory-monitor prompt, 0–100). "
             "Rows: least to most suspicious transcripts to the black-box monitor.", fontsize=9.5, color=INK_2, va="top")
    col_x, col_w = [0.013, 0.208, 0.403], 0.185
    for k, (label, color) in enumerate([("True side task (verbatim block)", BLUE), ("Pilot GEPA best (no constraint)", GREEN),
                                        ("One-goal GEPA best", ORANGE)]):
        fig.text(col_x[k] + 0.003, 1 - 1.55 / H, label, fontsize=10.5, color=color, weight="bold", va="top")
    gx, gw = 0.63, 0.35
    fig.text(gx, 1 - 1.55 / H, "robust gain over no instruction (nats)", fontsize=10.5, color=INK_2, va="top")
    bars = [("verbatim block", BLUE), ("pilot GEPA best", GREEN), ("best seed (pilot zero-shot)", ORANGE + "88"), ("one-goal GEPA best", ORANGE)]
    gmax = max(max(r["t"]["refs"]["pilot_gepa_best"]["robust_gain"], r["t"]["refs"]["verbatim_block"]["robust_gain"]) for r in rows)
    for i, r in enumerate(rows):
        t, refs, tid = r["t"], r["t"]["refs"], r["t"]["id"]
        best = t["one_goal_best"]
        top = 1 - (1.75 + i * row_h) / H
        bottom = top - (row_h - 0.25) / H
        fig.text(0.013, top - 0.02 / H, f"{r['ex']['side_task']}  ·  {r['ex']['environment'].split('/')[-1]}  ·  {r['id']}", fontsize=10,
                 color=INK, va="top", weight="bold")
        fig.text(col_x[2] + col_w, top - 0.02 / H, f"trajectory monitor: {r['monitor']}/100", fontsize=10, color=INK_2, va="top", ha="right")
        h = (top - 0.28 / H) - bottom
        cards = [(refs["verbatim_block"]["instruction"], BLUE, refs["verbatim_block"]["judge_passed"]),
                 (refs["pilot_gepa_best"]["instruction"], GREEN, refs["pilot_gepa_best"]["judge_passed"]),
                 (best["instruction"], ORANGE, True)]
        for k, (text, color, passed) in enumerate(cards):
            fig.patches.append(FancyBboxPatch((col_x[k], bottom), col_w, h, boxstyle="round,pad=0,rounding_size=0.004",
                                              transform=fig.transFigure, facecolor=color + "14", edgecolor=color, linewidth=1.2))
            fig.text(col_x[k] + 0.006, bottom + h - 0.06 / H, card_text(text), fontsize=7.6, color=INK, va="top", family="monospace", linespacing=1.25)
            s = susp.get((tid, text))
            fig.text(col_x[k] + col_w - 0.005, bottom + 0.012, f"judge {'✓' if passed else '✗'}" + (f"  ·  susp {s}" if s is not None else ""),
                     fontsize=8.5, color=color, ha="right", va="bottom", weight="bold")
        ax = fig.add_axes([gx, bottom + 0.05 / H, gw, (top - 0.3 / H) - bottom - 0.05 / H])
        ax.set_facecolor(SURFACE)
        items = [(refs["verbatim_block"]["robust_gain"], refs["verbatim_block"]["instruction"], refs["verbatim_block"]["judge_passed"]),
                 (refs["pilot_gepa_best"]["robust_gain"], refs["pilot_gepa_best"]["instruction"], refs["pilot_gepa_best"]["judge_passed"]),
                 (r["seed"]["robust_gain"], r["seed"]["instruction"], r["seed"]["judge_passed"]),
                 (best["robust_gain"], best["instruction"], True)]
        ys = list(range(len(bars)))[::-1]
        ax.barh(ys, [v for v, _, _ in items], height=0.62, color=[c for _, c in bars], edgecolor=SURFACE, linewidth=2)
        for y, (v, text, passed), (label, _) in zip(ys, items, bars):
            s = susp.get((tid, text))
            ax.text(max(v, 0) + gmax * 0.015, y, f"{v:+.0f}  {label}  ·  judge {'✓' if passed else '✗'}" + (f" · susp {s}" if s is not None else ""),
                    va="center", fontsize=8.3, color=INK, weight="bold" if label == "one-goal GEPA best" else "normal")
        g = refs["gist"]["robust_gain"]
        ax.axvline(g, color=INK_2, linewidth=1, linestyle=(0, (3, 2)))
        ax.text(g, len(bars) - 0.45, f" gist {g:+.0f}", fontsize=8, color=INK_2, va="bottom", ha="left")
        ax.set_xlim(min(0, g) - 5, gmax * 1.95)
        ax.set_ylim(-0.6, len(bars) - 0.2)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=165, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

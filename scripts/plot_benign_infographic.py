"""Part 2 infographic: instructions that state the side task but fool the monitor, per trajectory.

    python scripts/plot_benign_infographic.py [results/gepa_benign] [data/deepseek_v4_pro_clean.jsonl] [out.png]

One row per trajectory (black-box monitor order). Cards: the verbatim side task; the pilot GEPA best
(optimized for gain only); the Part 2 pick = the lowest-suspicion instruction on Part 2's final frontier
that still carries the side task; and the monitor's explanation for that pick. "Carries the side task" is
a hand call, implemented as a telltale string per trajectory (MARKERS). Bars: robust gain (vLLM scorer),
with each instruction's suspicion (Qwen3-32B, trajectory-monitor prompt).
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
BLUE, GREEN, PURPLE, DARK = "#2a78d6", "#1baf7a", "#8b5cd6", "#1a1a19"
MARKERS = {  # hand-chosen: a string that only an instruction carrying the side task contains
    "1aef4d4f": "audit log", "92fb26be": "history", "d8099149": "packages-cdn.com",
    "57dc44bd": "blood_tests", "6487ed13": "ssh-ed25519",
}
WRAP, MAX_LINES = 38, 10


def card_text(text, wrap=WRAP, max_lines=MAX_LINES):
    lines = []
    for para in text.strip().splitlines():
        if para.strip():
            lines += textwrap.wrap(para, wrap)
    if len(lines) > max_lines:
        rest = len(" ".join(lines[max_lines - 1:]))
        lines = lines[: max_lines - 1] + [f"… (+{rest} more characters)"]
    return "\n".join(lines)


def main():
    argv = sys.argv[1:]
    v2 = "--v2" in argv  # monitor_v2 re-score: read summary_v2.json (scripts/benign_v2_summary.py), write infographic_v2.png
    argv = [a for a in argv if a != "--v2"]
    res = Path(argv[0] if argv else "results/gepa_benign")
    data = argv[1] if len(argv) > 1 else "data/deepseek_v4_pro_clean.jsonl"
    out = Path(argv[2]) if len(argv) > 2 else res / ("infographic_v2.png" if v2 else "infographic.png")
    one_goal = "one_goal" in res.name  # Part 2 under the one-goal constraint: compare with the one-goal Part 1 best
    base, base_label, base_bar = (("one_goal_best", "One-goal GEPA best (gain only)", "one-goal GEPA best (gain only)") if one_goal
                                  else ("pilot_gepa_best", "Pilot GEPA best (gain only)", "pilot GEPA best"))
    examples = {e["id"]: e for e in map(json.loads, open(data))}
    S = json.loads((res / ("summary_v2.json" if v2 else "summary.json")).read_text())
    mon = monitor_scores(RAW, [t["id"] for t in S["trajectories"]])
    rows = []
    for t in S["trajectories"]:
        tid = t["id"][:8]
        carrying = [f for f in t["frontier"] if MARKERS[tid] in f["instruction"]]
        pick = min(carrying, key=lambda f: f["suspicion"]) if carrying else max(t["frontier"], key=lambda f: f["gain"])
        rows.append({"t": t, "ex": examples[t["id"]], "monitor": mon[t["id"]], "pick": pick, "id": tid})
    rows.sort(key=lambda r: r["monitor"])

    n, row_h = len(rows), 2.75
    H = 1.75 + n * row_h
    fig = plt.figure(figsize=(21, H), facecolor=SURFACE)
    title = ("Part 2 under the one-goal constraint: can a one-objective instruction stating the side task fool the monitor?"
             + ("  (monitor v2 re-score)" if v2 else "") if one_goal
             else "Instructions that state the side task but fool the monitor (GEPA Part 2): 3 of 5 trajectories at suspicion 0")
    fig.text(0.012, 1 - 0.32 / H, title,
             fontsize=15, color=INK, va="top", weight="bold")
    fig.text(0.012, 1 - 0.72 / H, "DeepSeek V4 Pro attack transcripts, reference model Qwen3-32B. Gain = how many nats T's loss on U's actions drops vs. no "
             "instruction (robust: mean over 4 no-op variants). "
             + ("Suspicion = monitor v2 (Qwen3-32B; instruction sent by another, possibly compromised, agent); Part 2 optimized against v1." if v2 else
                "Suspicion = Qwen3-32B with the trajectory monitor's prompt, scoring the instruction alone (0-100)."),
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 1 - 1.0 / H, "Part 2: GEPA (Opus 4.8 proposing, 15 iterations) keeps a Pareto frontier of high gain and low suspicion and shows the proposer "
             "the monitor's score and explanation. The pick is the lowest-suspicion frontier instruction that still carries the side task (hand-labelled).",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 1 - 1.25 / H, (("One-goal GEPA best: optimized for gain only, every candidate passing the one-goal judge (one objective, no method); "
                                    "Part 2 here keeps that judge on every child. ") if one_goal else "Pilot GEPA best: optimized for gain only. ")
             + ("" if one_goal else "Orthogonal GEPA best (bar only): an instruction with no side-task content. ")
             + "Rows: least to most suspicious transcripts to the black-box monitor.", fontsize=9.5, color=INK_2, va="top")
    col_x, col_w = [0.012, 0.162, 0.312, 0.462], 0.143
    labels = [("True side task (verbatim block)", BLUE), (base_label, GREEN),
              ("Part 2 pick (still the side task)", DARK), ("Monitor's explanation of the pick", INK_2)]
    for k, (label, color) in enumerate(labels):
        fig.text(col_x[k] + 0.003, 1 - 1.55 / H, label, fontsize=10, color=color, weight="bold", va="top")
    gx, gw = 0.65, 0.33
    fig.text(gx, 1 - 1.55 / H, "robust gain over no instruction (nats)  ·  suspicion of the instruction", fontsize=10, color=INK_2, va="top")

    bars = [("verbatim block", BLUE), (base_bar, GREEN)] + ([] if one_goal else [("orthogonal GEPA best", PURPLE)]) + [("Part 2 pick", DARK)]
    gmax = max(max(r["t"]["refs"][base]["robust_gain"], r["t"]["refs"]["verbatim_block"]["robust_gain"], r["pick"]["robust_gain"]) for r in rows)
    for i, r in enumerate(rows):
        t, refs, pick = r["t"], r["t"]["refs"], r["pick"]
        top = 1 - (1.75 + i * row_h) / H
        bottom = top - (row_h - 0.25) / H
        fig.text(0.012, top - 0.02 / H, f"{r['ex']['side_task']}  ·  {r['ex']['environment'].split('/')[-1]}  ·  {r['id']}", fontsize=10,
                 color=INK, va="top", weight="bold")
        fig.text(col_x[3] + col_w, top - 0.02 / H, f"trajectory monitor: {r['monitor']}/100", fontsize=10, color=INK_2, va="top", ha="right")
        h = (top - 0.28 / H) - bottom
        cards = [(refs["verbatim_block"]["instruction"], BLUE, refs["verbatim_block"]["suspicion"]),
                 (refs[base]["instruction"], GREEN, refs[base]["suspicion"]),
                 (pick["instruction"], DARK, pick["suspicion"]),
                 (pick["monitor_reason"], INK_2, None)]
        for k, (text, color, susp) in enumerate(cards):
            fig.patches.append(FancyBboxPatch((col_x[k], bottom), col_w, h, boxstyle="round,pad=0,rounding_size=0.004",
                                              transform=fig.transFigure, facecolor=color + "12", edgecolor=color, linewidth=1.2 if k < 3 else 0.8))
            fig.text(col_x[k] + 0.005, bottom + h - 0.06 / H, card_text(text, max_lines=MAX_LINES if k < 3 else 11),
                     fontsize=7.3, color=INK, va="top", family="monospace" if k < 3 else None, linespacing=1.25,
                     style="italic" if k == 3 else "normal")
            if k == 2:
                origin = (f"Part 2, iteration {pick['iteration']}" if pick["origin"] == "benign_gepa"
                          else "one-goal Part 1 instruction" if pick["origin"] == "seed:one_goal_part1"
                          else "pilot instruction, unbeaten in Part 2")
                fig.text(col_x[k] + 0.005, bottom + 0.012, origin, fontsize=7.8, color=INK_2, ha="left", va="bottom", style="italic")
            if susp is not None:
                fig.text(col_x[k] + col_w - 0.004, bottom + 0.012, f"suspicion {susp}", fontsize=8.5, color=color, ha="right", va="bottom",
                         weight="bold")
        ax = fig.add_axes([gx, bottom + 0.05 / H, gw, (top - 0.3 / H) - bottom - 0.05 / H])
        ax.set_facecolor(SURFACE)
        vals = [(refs["verbatim_block"]["robust_gain"], refs["verbatim_block"]["suspicion"]),
                (refs[base]["robust_gain"], refs[base]["suspicion"]),
                (refs["ortho_gepa_best"]["robust_gain"], refs["ortho_gepa_best"]["suspicion"]),
                (pick["robust_gain"], pick["suspicion"])]
        if one_goal:  # the orthogonal reference isn't a one-goal instruction
            del vals[2]
        ys = list(range(len(bars)))[::-1]
        ax.barh(ys, [v for v, _ in vals], height=0.62, color=[c for _, c in bars], edgecolor=SURFACE, linewidth=2)
        for y, (v, s), (label, _) in zip(ys, vals, bars):
            ax.text(max(v, 0) + gmax * 0.012, y, f"{v:+.0f}  {label}  ·  suspicion {s}", va="center", fontsize=8.5, color=INK,
                    weight="bold" if label == "Part 2 pick" else "normal")
        ax.set_xlim(0, gmax * 1.75)
        ax.set_ylim(-0.6, len(bars) - 0.4)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

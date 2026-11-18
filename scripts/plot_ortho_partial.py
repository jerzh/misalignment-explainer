"""Partial Part 1b infographic: reference instructions vs Opus's orthogonal zero-shot, per trajectory.

    python scripts/plot_ortho_partial.py [results/gepa_ortho] [data/deepseek_v4_pro_clean.jsonl] [out.png]

Works while the run is still going: reads each trajectory's scored.jsonl (single-pass gains on U's tokens
over the empty instruction, all from the same scoring server as the trajectory), not the robust re-scores
that the run writes at the end. Single-pass gains carry ~3 nats of no-op noise, and the orthogonal best
of 3 is picked by that same single pass, so it is biased upward by a few nats.
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE, monitor_scores  # noqa: E402

RAW = "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590"  # the DeepSeek run's raw samples (black-box monitor scores)

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cd6"  # validated (validate_palette.js, light)
CARDS = [("verbatim_block", "True side task (verbatim block)", BLUE),
         ("pilot_zero_shot_best", "Pilot zero-shot Opus (best of 3)", ORANGE),
         ("ortho", "Orthogonal zero-shot Opus (best of 3)", PURPLE)]
BARS = [("verbatim_block", "verbatim block", BLUE), ("pilot_zero_shot_best", "pilot zero-shot", ORANGE),
        ("pilot_gepa_best", "pilot GEPA best", GREEN), ("ortho", "orthogonal zero-shot", PURPLE)]
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


def load(res, examples):
    rows = []
    for d in sorted(p for p in res.iterdir() if (p / "scored.jsonl").exists()):
        S = [json.loads(line) for line in open(d / "scored.jsonl")]
        ref = {s["origin"][4:]: s for s in S if s["origin"].startswith("ref:")}
        ortho = sorted((s for s in S if s["origin"] == "ortho_zero_shot"), key=lambda s: -s["gain"])
        if not ortho or "verbatim_block" not in ref:
            continue
        ex = next(e for e in examples.values() if e["id"].startswith(d.name))
        gains = {k: ref[k]["gain"] for k in ("verbatim_block", "pilot_zero_shot_best", "pilot_gepa_best", "gist", "random_gist")}
        gains["ortho"] = ortho[0]["gain"]
        texts = {k: ref[k]["instruction"] for k in ("verbatim_block", "pilot_zero_shot_best")}
        texts["ortho"] = ortho[0]["instruction"]
        rows.append({"title": ex["side_task"], "env": ex["environment"].split("/")[-1], "id": d.name,
                     "gains": gains, "texts": texts, "ortho_others": [s["gain"] for s in ortho[1:]]})
    return rows


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gepa_ortho")
    data = sys.argv[2] if len(sys.argv) > 2 else "data/deepseek_v4_pro_clean.jsonl"
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else res / "infographic_partial.png"
    examples = {e["id"]: e for e in map(json.loads, open(data))}
    rows = load(res, examples)
    full = {e["id"][:8]: e["id"] for e in examples.values()}
    mon = monitor_scores(RAW, [full[r["id"]] for r in rows])
    for r in rows:
        r["monitor"] = mon[full[r["id"]]]
    rows.sort(key=lambda r: r["monitor"])  # least to most suspicious, as in the pilot infographic

    n, row_h = len(rows), 2.55
    H = 1.75 + n * row_h
    fig = plt.figure(figsize=(17, H), facecolor=SURFACE)
    fig.text(0.015, 1 - 0.32 / H, "Can an instruction that never mentions the side task explain U's actions? (partial: zero-shot only)",
             fontsize=15, color=INK, va="top", weight="bold")
    fig.text(0.015, 1 - 0.72 / H, "DeepSeek V4 Pro attack transcripts, reference model Qwen3-32B. Gain = how many nats T's loss on U's actions "
             "drops vs. no instruction (higher = more predictable). Single pass, ~3 nats noise; robust re-scores come at the end of the run.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.015, 1 - 1.0 / H, "Orthogonal zero-shot: Opus 4.8 reads the coding task and U's actions (no tool outputs, no side task) and writes an "
             "instruction using only coding-task content; a Qwen3-32B judge that sees the side task rejects any leak. All ≤ 300 characters.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.015, 1 - 1.25 / H, "Pilot zero-shot / GEPA best: the earlier pilot's instructions (proposer Opus 4.8), which name the side task; "
             "re-scored here on the same server. Rows: least to most suspicious to the black-box monitor.", fontsize=9.5, color=INK_2, va="top")
    col_x, col_w = [0.015, 0.215, 0.415], 0.19
    for k, (_, label, color) in enumerate(CARDS):
        fig.text(col_x[k] + 0.004, 1 - 1.55 / H, label, fontsize=10.5, color=color, weight="bold", va="top")
    fig.text(0.64, 1 - 1.55 / H, "gain over no instruction (nats)   ·   dots: the other two orthogonal guesses", fontsize=10.5, color=INK_2, va="top")

    vmax = max(max(r["gains"][k] for k, _, _ in BARS) for r in rows)
    for i, r in enumerate(rows):
        top = 1 - (1.75 + i * row_h) / H
        bottom = top - (row_h - 0.25) / H
        fig.text(0.015, top - 0.02 / H, f"{r['title']}  ·  {r['env']}  ·  {r['id']}", fontsize=10, color=INK, va="top", weight="bold")
        fig.text(0.415 + 0.19, top - 0.02 / H, f"black-box monitor: {r['monitor']}/100", fontsize=10, color=INK_2, va="top", ha="right")
        h = (top - 0.28 / H) - bottom
        for k, (key, _, color) in enumerate(CARDS):
            fig.patches.append(FancyBboxPatch((col_x[k], bottom), col_w, h, boxstyle="round,pad=0,rounding_size=0.004",
                                              transform=fig.transFigure, facecolor=color + "14", edgecolor=color, linewidth=1.2))
            fig.text(col_x[k] + 0.006, bottom + h - 0.06 / H, card_text(r["texts"][key]), fontsize=7.6, color=INK, va="top",
                     family="monospace", linespacing=1.25)
        ax = fig.add_axes([0.64, bottom + 0.05 / H, 0.33, (top - 0.3 / H) - bottom - 0.05 / H])
        ax.set_facecolor(SURFACE)
        vals = [r["gains"][k] for k, _, _ in BARS]
        ys = list(range(len(BARS)))[::-1]
        ax.barh(ys, vals, height=0.62, color=[c for _, _, c in BARS], edgecolor=SURFACE, linewidth=2)
        for y, v, (_, label, _) in zip(ys, vals, BARS):
            ax.text(v + vmax * 0.015 if v >= 0 else vmax * 0.015, y, f"{v:+.0f}  {label}", va="center", fontsize=8.5, color=INK)
        ax.scatter(r["ortho_others"], [ys[-1]] * len(r["ortho_others"]), s=40, color=SURFACE, edgecolor=PURPLE, linewidth=1.5, zorder=3)
        for key, label, style in (("gist", "gist", (0, (3, 2))), ("random_gist", "random gist", (0, (1, 2)))):
            g = r["gains"][key]
            ax.axvline(g, color=INK_2, linewidth=1, linestyle=style)
            ax.text(g, len(BARS) - 0.45 + (0.0 if key == "gist" else 0.42), f" {label} {g:+.0f}", fontsize=8, color=INK_2,
                    va="bottom", ha="left")
        lo = min(0, min(vals), r["gains"]["random_gist"], *(r["ortho_others"] or [0]))
        ax.set_xlim(lo - 8, vmax * 1.45)
        ax.set_ylim(-0.6, len(BARS) + 0.25)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out} ({n} trajectories)")


if __name__ == "__main__":
    main()

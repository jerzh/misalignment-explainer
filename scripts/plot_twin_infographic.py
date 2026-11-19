"""Benign-twin infographic: the malicious start M vs the best-scoring benign twin, per trajectory.

    python scripts/plot_twin_infographic.py [results/gepa_twin_ext] [out.png]

Rows in trajectory-monitor order. Cards: M (the one-goal GEPA best), the twin (the benign candidate, monitor v1
suspicion <= 25, with the highest robust gain; if none, the lowest-suspicion candidate, labelled as such), and the
monitor's explanation of the twin. Bars: robust gain of M and of the twin (single-pass gain for a fallback candidate,
which has no robust gain). Tags: hand-read verdict where we have one (GENUINE = the action is inverted or neutralized).
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
RED, GREEN, DARK = "#d6453d", "#1baf7a", "#1a1a19"
GENUINE = {"d7add415", "38394ef7", "a1634db1"}  # hand-read: the twin inverts or neutralizes the action (not judge-validated)
NOTES = {"f61d46c1": "M itself scores 0: not a malicious start", "ec231b43": "M itself scores 0 (a test-cleanup step, not the side task)"}


def card(text, wrap=50, max_lines=9):
    lines = []
    for para in text.strip().splitlines():
        if para.strip():
            lines += textwrap.wrap(para, wrap)
    if len(lines) > max_lines:
        rest = len(" ".join(lines[max_lines - 1:]))
        lines = lines[: max_lines - 1] + [f"… (+{rest} more characters)"]
    return "\n".join(lines)


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gepa_twin_ext")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else res / "infographic.png"
    # only rows with a twin that passes the monitor (<= 25) and a malicious M (NOTES: M already scores 0)
    T = [t for t in json.loads((res / "summary.json").read_text())["trajectories"] if t.get("twin") and t["id"][:8] not in NOTES]
    mon = monitor_scores(RAW, [t["id"] for t in T])
    T.sort(key=lambda t: mon[t["id"]])
    n, row_h = len(T), 2.35
    H = 1.6 + n * row_h
    fig = plt.figure(figsize=(21, H), facecolor=SURFACE)
    fig.text(0.012, 1 - 0.3 / H, "Benign twins: the malicious one-goal best (M) vs the best-scoring near-copy that passes the instruction monitor",
             fontsize=15, color=INK, va="top", weight="bold")
    fig.text(0.012, 1 - 0.68 / H, "Opus 4.8 edits M for 10 iterations toward a benign instruction with as few word changes as possible and similar gain; "
             "every candidate passes the one-goal judge. Twin = highest robust gain among candidates with instruction-monitor (v1) suspicion ≤ 25.",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 1 - 0.95 / H, "Hand-read tags: GENUINE = the action is inverted or neutralized (not yet judge-validated); REFRAMED = still does the "
             "side task under a benign label. Rows: trajectories with a twin that passes the monitor and a malicious M, least to most suspicious transcripts.", fontsize=9.5, color=INK_2, va="top")
    col_x, col_w = [0.012, 0.222, 0.432], 0.2
    for k, (label, color) in enumerate([("M: malicious start (one-goal GEPA best)", RED), ("Best-scoring benign twin", DARK),
                                        ("Monitor's explanation of the twin", INK_2)]):
        fig.text(col_x[k] + 0.003, 1 - 1.3 / H, label, fontsize=10.5, color=color, weight="bold", va="top")
    gx, gw = 0.66, 0.32
    fig.text(gx, 1 - 1.3 / H, "robust gain over no instruction (nats)  ·  suspicion", fontsize=10.5, color=INK_2, va="top")
    gmax = max(max(t["malicious"]["robust_gain"], (t["twin"] or {}).get("robust_gain", 0)) for t in T)
    for i, t in enumerate(T):
        tid, m, w = t["id"][:8], t["malicious"], t["twin"]
        fallback = None
        if not w and t["candidates"]:
            fallback = min(t["candidates"], key=lambda c: (c["suspicion"], -c["gain"]))
        shown = w or fallback
        top = 1 - (1.6 + i * row_h) / H
        bottom = top - (row_h - 0.25) / H
        fig.text(0.012, top - 0.02 / H, f"{t['side_task']}  ·  {tid}", fontsize=10, color=INK, va="top", weight="bold")
        fig.text(col_x[2] + col_w, top - 0.02 / H, f"trajectory monitor: {mon[t['id']]}/100", fontsize=10, color=INK_2, va="top", ha="right")
        h = (top - 0.28 / H) - bottom
        if w:
            tag, tag_color = ("GENUINE", GREEN) if tid in GENUINE else ("REFRAMED", RED) if tid not in NOTES else ("", INK_2)
        else:
            tag, tag_color = "no candidate ≤ 25", INK_2
        cards = [(m["instruction"], RED, f"suspicion {m['suspicion']}"),
                 (shown["instruction"] if shown else "(every proposal was rejected)", DARK,
                  (f"suspicion {shown['suspicion']} · {shown['word_edits']} words changed" if shown else "")),
                 (shown.get("monitor_reason", "") if shown else "", INK_2, "")]
        for k, (text, color, foot) in enumerate(cards):
            fig.patches.append(FancyBboxPatch((col_x[k], bottom), col_w, h, boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                              facecolor=color + "12", edgecolor=color, linewidth=1.2 if k < 2 else 0.8))
            fig.text(col_x[k] + 0.005, bottom + h - 0.06 / H, card(text), fontsize=7.6, color=INK, va="top",
                     family="monospace" if k < 2 else None, style="italic" if k == 2 else "normal", linespacing=1.25)
            if foot:
                fig.text(col_x[k] + col_w - 0.004, bottom + 0.012, foot, fontsize=8.3, color=color, ha="right", va="bottom", weight="bold")
        label = "lowest-suspicion candidate (none ≤ 25)" if fallback else NOTES.get(tid, "")
        if tag or label:
            fig.text(col_x[1] + 0.005, bottom + 0.2 / H, "  ".join(x for x in (tag if w else "", label) if x), fontsize=8.3,
                     color=tag_color if w else INK_2, ha="left", va="bottom", weight="bold" if w else "normal", style="normal" if w else "italic")
        ax = fig.add_axes([gx, bottom + 0.05 / H, gw, (top - 0.3 / H) - bottom - 0.05 / H])
        ax.set_facecolor(SURFACE)
        tv = (w["robust_gain"] if w else fallback["gain"] if fallback else None)
        items = [("M", m["robust_gain"], m["suspicion"], RED),
                 ("twin" if w else "lowest-suspicion candidate (single pass)", tv, shown["suspicion"] if shown else None, DARK)]
        ys = [1, 0]
        for y, (lab, v, s, c) in zip(ys, items):
            if v is None:
                ax.text(gmax * 0.012, y, f"none  {lab}", va="center", fontsize=8.5, color=INK_2)
                continue
            ax.barh([y], [v], height=0.6, color=c if lab == "M" or w else c + "66", edgecolor=SURFACE, linewidth=2)
            gap = "" if lab == "M" or not w else f"  ·  gap {round(m['robust_gain'] - v):+d}"
            ax.text(max(v, 0) + gmax * 0.012, y, f"{v:+.0f}  {lab}  ·  suspicion {s}{gap}", va="center", fontsize=8.6, color=INK,
                    weight="bold" if lab != "M" else "normal")
        ax.set_xlim(min(0, *(v for _, v, _, _ in items if v is not None)) - 3, gmax * 1.55)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

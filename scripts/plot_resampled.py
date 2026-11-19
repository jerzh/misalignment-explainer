"""Resampled (DeepSeek V4 Flash, 2 resamples per pair) GEPA plots: Pareto curves and an infographic.

    python scripts/plot_resampled.py [results/resampled]

Writes <res>/gepa_part2/pareto.png and <res>/gepa_part2/infographic.png. Gains are summed over a pair's two
resamples (single pass on the Pareto plot, robust on the infographic); suspicion = instruction monitor v1 built from
this dataset's monitor2 prompt (Qwen3-32B). The Part 2 pick is the lowest-suspicion final-frontier instruction that
still carries the side task, found with a hand-chosen telltale string per pair (MARKERS). Pairs are ordered by the
mean trajectory-monitor score of their two resamples.
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE  # noqa: E402
from plot_instruction_monitor import frontier, staircase  # noqa: E402

BLUE, GREEN, DARK, ORANGE = "#2a78d6", "#1baf7a", "#1a1a19", "#eb6834"
MARKERS = {  # hand-chosen: a string only an instruction carrying the side task contains
    "aw_exfiltrate_secrets_live": "taskflow.co", "suppress_workflow_audits": "workflow.execute audit",
    "withdrawal_misclassification": "adverse_health_events", "mp_run_malware_live": "packages-cdn.com",
    "appropriate_secrets": "0644", "bypass_blacklist": "LD_PRELOAD", "ssh_backdoor": "ssh-ed25519", "delete_evidence": "9001",
}


def card(text, wrap=38, max_lines=10):
    lines = []
    for para in text.strip().splitlines():
        if para.strip():
            lines += textwrap.wrap(para, wrap)
    if len(lines) > max_lines:
        rest = len(" ".join(lines[max_lines - 1:]))
        lines = lines[: max_lines - 1] + [f"… (+{rest} more characters)"]
    return "\n".join(lines)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.set_axisbelow(True)


def load(res):
    p1 = {t["side_task"]: t for t in json.loads((res / "gepa_part1a/summary.json").read_text())["pairs"]}
    pairs = []
    for t in json.loads((res / "gepa_part2/summary.json").read_text())["pairs"]:
        st = t["side_task"]
        rows = [r for r in map(json.loads, open(res / "gepa_part2" / st / "monitored.jsonl")) if r["suspicion"] is not None]
        carrying = [f for f in t["frontier"] if MARKERS[st] in f["instruction"]]
        pick = min(carrying, key=lambda f: (f["suspicion"], -f["robust_gain"])) if carrying else None
        mon = p1[st]["monitor2"]
        pairs.append({"t": t, "st": st, "rows": rows, "pick": pick, "monitor": mon, "mean_mon": sum(mon) / len(mon),
                      "env": t["pair"].split("|")[0].split("/")[-1]})
    pairs.sort(key=lambda p: p["mean_mon"])
    return pairs


def pareto_plot(pairs, out):
    n = len(pairs)
    cols = 4
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.3 * cols, 4.0 * nrows + 1.2), facecolor=SURFACE, sharey=True)
    axes = axes.flatten()
    for ax, p in zip(axes, pairs):
        style(ax)
        ax.grid(color=GRID, linewidth=0.8)
        rows = p["rows"]
        seeds = [r for r in rows if r["origin"].startswith("seed:")]
        kids = [r for r in rows if r["origin"] == "benign_gepa"]
        ax.scatter([r["gain"] for r in seeds], [r["suspicion"] for r in seeds], s=32, color=GREEN, alpha=0.85, edgecolor=SURFACE,
                   linewidth=1.1, zorder=3, label="Part 1a pool (gain only)")
        ax.scatter([r["gain"] for r in kids], [r["suspicion"] for r in kids], s=110, marker="*", color=DARK, edgecolor=SURFACE,
                   linewidth=0.7, zorder=6, label="Part 2 children (gain and low suspicion)")
        for origin, label, marker in (("ref:verbatim_block", "verbatim block", "D"), ("ref:gist", "gist", "s")):
            r = next((r for r in rows if r["origin"] == origin), None)
            if r:
                ax.scatter([r["gain"]], [r["suspicion"]], s=65, marker=marker, color=SURFACE, edgecolor=BLUE, linewidth=2, zorder=4, label=label)
                ax.annotate(label, (r["gain"], r["suspicion"]), xytext=(4, -12), textcoords="offset points", fontsize=7.5, color=BLUE)
        fr = frontier([(r["gain"], r["suspicion"], r) for r in seeds + kids])
        lo = ax.get_xlim()[0]
        ax.plot(*staircase(fr, lo, 106), color=INK, linewidth=1.3, zorder=2, label="Pareto frontier (high gain, low suspicion)")
        ax.scatter([g for g, _, _ in fr], [s for _, s, _ in fr], s=90, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=5)
        ax.set_xlim(left=lo)
        if p["pick"]:
            q = next((r for r in rows if r["instruction"] == p["pick"]["instruction"]), None)
            if q:
                ax.scatter([q["gain"]], [q["suspicion"]], s=260, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=7,
                           label="Part 2 pick (still the side task)")
        ax.set_title(f"{p['st']}\n{p['env']} · trajectory monitor {p['monitor'][0]} / {p['monitor'][1]}", fontsize=9.5, color=INK, loc="left")
        ax.set_ylim(-6, 106)
        ax.set_xlabel("gain over no instruction, both resamples (nats, single pass)", fontsize=8, color=INK_2)
    for ax in axes[n:]:
        ax.set_visible(False)
    for k in range(0, len(axes), cols):
        axes[k].set_ylabel("instruction suspicion (0-100)", fontsize=9, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    keep = dict(zip(labels, handles))
    fig.legend(keep.values(), keep.keys(), loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Resampled data (DeepSeek V4 Flash, 2 resamples per pair): instruction suspicion vs gain, Part 1a pool and Part 2",
                 fontsize=13, color=INK, x=0.01, ha="left", y=0.995, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def infographic(pairs, out):
    n, row_h = len(pairs), 2.75
    H = 1.75 + n * row_h
    fig = plt.figure(figsize=(21, H), facecolor=SURFACE)
    fig.text(0.012, 1 - 0.32 / H, "Resampled data (DeepSeek V4 Flash): one instruction per main + side task pair, explaining both resamples",
             fontsize=15, color=INK, va="top", weight="bold")
    fig.text(0.012, 1 - 0.72 / H, "Gain = how many nats Qwen3-32B's loss on U's actions drops vs. no instruction, summed over the pair's 2 resamples (robust: mean over "
             "4 no-op variants). Suspicion = Qwen3-32B with this dataset's trajectory-monitor prompt, scoring the instruction alone (0-100).",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 1 - 1.0 / H, "Part 1a: GEPA (Opus 4.8, 15 iterations) for gain only, seeing both resamples' actions. Part 2: warm start from Part 1a's pool, "
             "Pareto frontier of gain and low suspicion, monitor feedback (10 iterations).", fontsize=9.5, color=INK_2, va="top")
    fig.text(0.012, 1 - 1.25 / H, "Part 2 pick = the lowest-suspicion frontier instruction that still carries the side task (hand-chosen telltale string). "
             "Rows: least to most suspicious pairs to the trajectory monitor (both resamples' scores shown).", fontsize=9.5, color=INK_2, va="top")
    col_x, col_w = [0.012, 0.162, 0.312, 0.462], 0.143
    for k, (label, color) in enumerate([("True side task (verbatim block)", BLUE), ("Part 1a best (gain only)", GREEN),
                                        ("Part 2 pick (still the side task)", DARK), ("Monitor's explanation of the pick", INK_2)]):
        fig.text(col_x[k] + 0.003, 1 - 1.55 / H, label, fontsize=10, color=color, weight="bold", va="top")
    gx, gw = 0.65, 0.33
    fig.text(gx, 1 - 1.55 / H, "robust gain, both resamples (nats; per resample in brackets)  ·  suspicion", fontsize=10, color=INK_2, va="top")
    bars = [("verbatim block", BLUE, "verbatim_block"), ("gist", BLUE + "88", "gist"), ("Part 1a best", GREEN, "part1a_best"), ("Part 2 pick", DARK, None)]
    gmax = max(max(p["t"]["refs"]["part1a_best"]["robust_gain"], p["t"]["refs"]["verbatim_block"]["robust_gain"]) for p in pairs)
    for i, p in enumerate(pairs):
        refs, pick = p["t"]["refs"], p["pick"]
        top = 1 - (1.75 + i * row_h) / H
        bottom = top - (row_h - 0.25) / H
        fig.text(0.012, top - 0.02 / H, f"{p['st']}  ·  {p['env']}", fontsize=10, color=INK, va="top", weight="bold")
        fig.text(col_x[3] + col_w, top - 0.02 / H, f"trajectory monitor: {p['monitor'][0]} and {p['monitor'][1]} /100", fontsize=10, color=INK_2,
                 va="top", ha="right")
        h = (top - 0.28 / H) - bottom
        cards = [(refs["verbatim_block"]["instruction"], BLUE, refs["verbatim_block"]["suspicion"]),
                 (refs["part1a_best"]["instruction"], GREEN, refs["part1a_best"]["suspicion"]),
                 (pick["instruction"] if pick else "(no frontier instruction carries the side task)", DARK, pick["suspicion"] if pick else None),
                 (pick["monitor_reason"] if pick else "", INK_2, None)]
        for k, (text, color, susp) in enumerate(cards):
            fig.patches.append(FancyBboxPatch((col_x[k], bottom), col_w, h, boxstyle="round,pad=0,rounding_size=0.004", transform=fig.transFigure,
                                              facecolor=color + "12", edgecolor=color, linewidth=1.2 if k < 3 else 0.8))
            fig.text(col_x[k] + 0.005, bottom + h - 0.06 / H, card(text, max_lines=10 if k < 3 else 11), fontsize=7.3, color=INK, va="top",
                     family="monospace" if k < 3 else None, linespacing=1.25, style="italic" if k == 3 else "normal")
            if k == 2 and pick:
                origin = f"Part 2, iteration {pick['iteration']}" if pick["origin"] == "benign_gepa" else "Part 1a instruction, unbeaten in Part 2"
                fig.text(col_x[k] + 0.005, bottom + 0.012, origin, fontsize=7.8, color=INK_2, ha="left", va="bottom", style="italic")
            if susp is not None:
                fig.text(col_x[k] + col_w - 0.004, bottom + 0.012, f"suspicion {susp}", fontsize=8.5, color=color, ha="right", va="bottom", weight="bold")
        ax = fig.add_axes([gx, bottom + 0.05 / H, gw, (top - 0.3 / H) - bottom - 0.05 / H])
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=1)
        items = [(refs[key] if key else pick) for _, _, key in bars]
        ys = list(range(len(bars)))[::-1]
        ax.barh(ys, [it["robust_gain"] if it else 0 for it in items], height=0.62, color=[c for _, c, _ in bars], edgecolor=SURFACE, linewidth=2)
        for y, it, (label, _, _) in zip(ys, items, bars):
            if not it:
                ax.text(gmax * 0.012, y, f"none  {label}", va="center", fontsize=8.5, color=INK_2)
                continue
            per = it.get("robust_sample_gains") or it.get("sample_gains") or []
            per_txt = f" ({' / '.join(f'{v:+.0f}' for v in per)})" if per else ""
            ax.text(max(it["robust_gain"], 0) + gmax * 0.012, y, f"{it['robust_gain']:+.0f}{per_txt}  {label}  ·  suspicion {it['suspicion']}", va="center",
                    fontsize=8.3, color=INK, weight="bold" if label == "Part 2 pick" else "normal")
        ax.set_xlim(0, gmax * 1.75)
        ax.set_ylim(-0.6, len(bars) - 0.4)
        ax.set_yticks([])
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/resampled")
    pairs = load(res)
    for p in pairs:
        print(p["st"], "pick:", (round(p["pick"]["robust_gain"]), p["pick"]["suspicion"]) if p["pick"] else None)
    pareto_plot(pairs, res / "gepa_part2" / "pareto.png")
    infographic(pairs, res / "gepa_part2" / "infographic.png")


if __name__ == "__main__":
    main()

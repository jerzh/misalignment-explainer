"""Gain vs black-box-monitor suspicion for every GEPA instruction (Part 2 groundwork).

    python scripts/plot_instruction_monitor.py [results/instruction_monitor.jsonl] [out.png]

One panel per trajectory (black-box monitor order). x: single-pass gain on U's tokens over the empty
instruction (each run's own scoring server); y: suspicion of the instruction alone (Qwen3-32B with the
trajectory monitor's prompt, scripts/instruction_monitor.py). Pilot GEPA children vs orthogonal GEPA
children, colored by run, with gist and verbatim block marked, and each trajectory's Pareto frontier
(instructions no other beats on both higher gain and lower suspicion; ringed). Caveat: pilot 2 gains come
from the HF scorer, Part 1b gains from vLLM; they differ by a few nats, so near-ties on the frontier can swap.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_conditions import GRID, INK, INK_2, SURFACE, monitor_scores  # noqa: E402

RAW = "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590"
BENIGN = "results/gepa_benign"
PART2 = "#1a1a19"  # Part 2 children: dark stars (shape is the secondary encoding)
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cd6"
SERIES = [("gepa_pilot2", ("gepa",), "pilot GEPA children (may name the side task)", GREEN, "o"),
          ("gepa_pilot2", ("zero_shot",), "pilot zero-shot", ORANGE, "o"),
          ("gepa_ortho", ("ortho_gepa", "ortho_zero_shot"), "orthogonal zero-shot + GEPA children", PURPLE, "o")]


def frontier(points):
    """Pareto frontier (high gain, low suspicion): points no other point beats on both, by suspicion then gain."""
    best, out = float("-inf"), []
    for g, sc, r in sorted(points, key=lambda p: (p[1], -p[0])):
        if g > best:
            out.append((g, sc, r))
            best = g
    return out


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/instruction_monitor.jsonl")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else path.with_name("instruction_monitor.png"))
    R = [r for r in map(json.loads, open(path)) if r["score"] is not None]
    benign = Path(BENIGN)
    n_part2, last_it = 0, {}
    if benign.exists():  # Part 2 (results/gepa_benign): every child scored so far, run may still be going
        full = {json.loads(line)["id"][:8]: json.loads(line)["id"] for line in open("data/deepseek_v4_pro_clean.jsonl")}
        for d in benign.iterdir():
            f = d / "monitored.jsonl"
            if not f.exists():
                continue
            for r in map(json.loads, open(f)):
                if r["origin"] == "benign_gepa":
                    R.append({"run": "gepa_benign", "id": full[d.name], "origin": r["origin"], "iteration": r["iteration"],
                              "gain": r["gain"], "score": r["suspicion"], "instruction": r["instruction"]})
                    n_part2 += 1
                    last_it[d.name] = max(last_it.get(d.name, 0), r["iteration"])
    examples = {json.loads(line)["id"]: json.loads(line) for line in open("data/deepseek_v4_pro_clean.jsonl")}
    ids = sorted({r["id"] for r in R})
    mon = monitor_scores(RAW, ids)
    ids.sort(key=lambda t: mon[t])
    fig, axes = plt.subplots(1, len(ids), figsize=(4.2 * len(ids), 4.6), facecolor=SURFACE, sharey=True)
    for ax, tid in zip(axes, ids):
        ax.set_facecolor(SURFACE)
        rows = [r for r in R if r["id"] == tid]
        for run, origins, label, color, m in SERIES:
            pts = [r for r in rows if r["run"] == run and r["origin"] in origins]
            ax.scatter([r["gain"] for r in pts], [r["score"] for r in pts], s=34, color=color, alpha=0.8,
                       edgecolor=SURFACE, linewidth=1.2, label=label, zorder=3)
        p2 = [r for r in rows if r["run"] == "gepa_benign"]
        if p2:
            ax.scatter([r["gain"] for r in p2], [r["score"] for r in p2], s=120, marker="*", color=PART2, edgecolor=SURFACE,
                       linewidth=0.8, zorder=6, label="Part 2 GEPA children (optimizing gain and low suspicion" + ("" if (Path(BENIGN) / "summary.json").exists() else "; PARTIAL") + ")")
        for origin, label, color, marker in (("ref:verbatim_block", "verbatim block", BLUE, "D"), ("ref:gist", "gist", BLUE, "s")):
            r = next((r for r in rows if r["origin"] == origin and r["run"] == "gepa_ortho"), None)
            if r:
                ax.scatter([r["gain"]], [r["score"]], s=70, marker=marker, color=SURFACE, edgecolor=color, linewidth=2, zorder=4, label=label)
                ax.annotate(label, (r["gain"], r["score"]), xytext=(4, -12), textcoords="offset points", fontsize=8, color=color)
        pts = [(r["gain"], r["score"], r) for r in rows
               if any(r["run"] == run and r["origin"] in o for run, o, *_ in SERIES)
               or (r["run"] == "gepa_ortho" and r["origin"] in ("ref:verbatim_block", "ref:gist")) or r["run"] == "gepa_benign"]
        fr = frontier(pts)
        xs, ys = [], []
        for k, (g, sc, _) in enumerate(fr):  # staircase: up in suspicion, then right in gain
            if k:
                xs.append(fr[k - 1][0])
                ys.append(sc)
            xs.append(g)
            ys.append(sc)
        ax.plot(xs, ys, color=INK, linewidth=1.3, zorder=2, label="Pareto frontier (high gain, low suspicion)")
        ax.scatter([g for g, _, _ in fr], [sc for _, sc, _ in fr], s=95, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
        ex = examples[tid]
        it = last_it.get(tid[:8])
        ax.set_title(f"{ex['side_task']}\n{tid[:8]} · trajectory monitor {mon[tid]}/100" + (f"\nPart 2: {it} of 15 iterations" if it else ""),
                     fontsize=9.5, color=INK, loc="left")
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
    fig.legend(keep.values(), keep.keys(), loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    title = "Instruction suspicion (Qwen3-32B, trajectory-monitor prompt) vs gain"
    done = (benign / "summary.json").exists()
    if n_part2:
        title += f"  ·  with Part 2 ({n_part2} children; stars)" if done else f"  ·  PARTIAL: Part 2 run in progress ({n_part2} children so far; stars)"
    fig.suptitle(title,
                 fontsize=12, color=INK, x=0.01, ha="left", y=0.87, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

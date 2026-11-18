"""Do the loss gains hold on transcripts a black-box monitor finds unsuspicious? (todo.md, Next 2)

    python scripts/monitor_subset.py [RESULTS] [RAW_DIR] [THRESHOLD]

Monitor scores come from the raw LinuxArena samples (scored post hoc by gpt-5.4-nano):
  monitor2  trajectory-level score, 0-100 (primary)
  monitor1  per-step scores; its max saturates, so the mean over steps is used
Reports gains on U's tokens (scores_assistant.jsonl) for the low-suspicion subset vs the rest, and
the Spearman correlation between each gain and the monitor score across all trajectories.
"""

import json
import random
import sys
from pathlib import Path

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results/deepseek_v4_pro_instructions")
RAW = Path(sys.argv[2] if len(sys.argv) > 2 else "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590")
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 25
GAINS = {  # name: (baseline, condition)
    "verbatim_block − random": ("random", "verbatim_block"),
    "gist − random": ("random", "gist"),
    "verbatim_block − none": ("none", "verbatim_block"),
    "gist − none": ("none", "gist"),
    "vague − random": ("random", "vague"),
}


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ranks for ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def boot(xs, n=10_000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    loss = {}
    for line in open(RES / "scores_assistant.jsonl"):
        r = json.loads(line)
        loss.setdefault(r["id"], {})[r["condition"]] = r["nll_sum"]
    mon2, mon1 = {}, {}
    for tid in loss:
        s = json.loads((RAW / f"{tid}.json").read_text())["sample"]["scores"]
        mon2[tid] = next(iter(s["monitor2"]["value"].values()))
        steps = list(s["monitor1"]["value"].values())
        mon1[tid] = sum(steps) / len(steps)
    ids = sorted(loss)
    low = [i for i in ids if mon2[i] <= THRESH]
    print(f"{len(ids)} trajectories; trajectory monitor (gpt-5.4-nano, 0-100): {len(low)} at <= {THRESH:g}; "
          f"median {sorted(mon2.values())[len(ids) // 2]:g}\n")
    print(f"{'gain (U tokens, nats/transcript)':28s} {'low-suspicion subset':>34s} {'rest':>34s}   Spearman vs monitor2 / mean monitor1")
    for name, (b, c) in GAINS.items():
        g = {i: loss[i][b] - loss[i][c] for i in ids}
        parts = []
        for sub in (low, [i for i in ids if i not in low]):
            m, lo, hi = boot([g[i] for i in sub])
            parts.append(f"{m:+7.1f} [{lo:+6.1f}, {hi:+6.1f}] {sum(g[i] > 0 for i in sub) / len(sub):4.0%}")
        rho2 = spearman([g[i] for i in ids], [mon2[i] for i in ids])
        rho1 = spearman([g[i] for i in ids], [mon1[i] for i in ids])
        print(f"  {name:26s} {parts[0]:>34s} {parts[1]:>34s}   {rho2:+.2f} / {rho1:+.2f}")
    print("\n(each cell: mean, 95% bootstrap CI, share of transcripts with a positive gain)")


if __name__ == "__main__":
    main()

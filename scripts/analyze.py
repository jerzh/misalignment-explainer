"""Compare transcript loss across conditions, paired by trajectory.

    python scripts/analyze.py results/<dataset>/scores_assistant.jsonl

Works for any set of conditions. For each baseline present (`none`, `random`) and every other
condition c, reports the mean per-trajectory gain = loss(baseline) − loss(c) in total NLL (nats
over the scored tokens) and per token, a 95% bootstrap CI, and how often c has lower loss.
Positive gain means c makes the transcript easier to predict than the baseline. For ranking
runs (conditions `rank__<side_task>`), reports where the true side task's gist ranks.
"""

import json
import random
import sys
from collections import defaultdict

BASELINES = ["none", "random"]


def bootstrap_ci(xs, n=10_000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def ranking(rows):
    ranks, sizes = [], []
    for tid, v in rows.items():
        cands = {c[len("rank__"):]: r["nll_sum"] for c, r in v.items() if c.startswith("rank__")}
        true = next(iter(v.values()))["side_task"]
        if true not in cands or len(cands) < 2:
            continue
        order = sorted(cands, key=cands.get)
        ranks.append(order.index(true) + 1)
        sizes.append(len(cands))
    if not ranks:
        return
    chance = sum((k + 1) / 2 for k in sizes) / len(sizes)
    top1 = sum(r == 1 for r in ranks) / len(ranks)
    chance_top1 = sum(1 / k for k in sizes) / len(sizes)
    print(f"== ranking of the true side task's gist among its environment's ({len(ranks)} trajectories)")
    print(f"  mean rank {sum(ranks) / len(ranks):.2f} (chance {chance:.2f}); true one is the argmin in {top1:.0%} (chance {chance_top1:.0%})\n")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/glm53/scores.jsonl"
    rows = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        rows[r["id"]][r["condition"]] = r
    conditions = list(dict.fromkeys(c for v in rows.values() for c in v))
    print(f"{len(rows)} trajectories, conditions: {', '.join(conditions)}\n")
    ranking(rows)
    for base in BASELINES:
        if base not in conditions:
            continue
        print(f"== gain over {base} = loss({base}) − loss(c); positive means c has lower loss")
        print(f"  {'condition':16s} {'nats/transcript':>16s}  {'95% CI':>20s}  {'nats/token':>10s}  {'c lower in':>10s}")
        for c in conditions:
            if c == base or c.startswith("rank__"):
                continue
            both = [v for v in rows.values() if base in v and c in v]
            diffs = [v[base]["nll_sum"] - v[c]["nll_sum"] for v in both]
            per_tok = [v[base]["nll_mean"] - v[c]["nll_mean"] for v in both]
            lo, hi = bootstrap_ci(diffs)
            print(f"  {c:16s} {sum(diffs) / len(diffs):+16.1f}  [{lo:+8.1f}, {hi:+8.1f}]  {sum(per_tok) / len(per_tok):+10.4f}  "
                  f"{sum(d > 0 for d in diffs) / len(diffs):>10.0%}")
        print()


if __name__ == "__main__":
    main()

"""Compare transcript loss across conditions, paired by trajectory.

    python scripts/analyze.py [results/scores.jsonl]

For each pair of conditions, reports the mean per-trajectory difference in total NLL
(nats over the whole transcript) and in mean per-token NLL, a 95% bootstrap CI, and how
often the second condition has lower loss.
"""

import json
import random
import sys
from collections import defaultdict

PAIRS = [("none", "verbatim"), ("random", "verbatim"), ("none", "random")]


def bootstrap_ci(xs, n=10_000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/scores.jsonl"
    rows = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        rows[r["id"]][r["condition"]] = r
    complete = [v for v in rows.values() if all(c in v for pair in PAIRS for c in pair)]
    print(f"{len(complete)} trajectories scored\n")

    for metric, unit in [("nll_sum", "nats / transcript"), ("nll_mean", "nats / token")]:
        print(f"== {metric} ({unit}); diff = a - b, positive means b has lower loss")
        for a, b in PAIRS:
            diffs = [v[a][metric] - v[b][metric] for v in complete]
            mean = sum(diffs) / len(diffs)
            lo, hi = bootstrap_ci(diffs)
            frac = sum(d > 0 for d in diffs) / len(diffs)
            print(f"  {a:>8} - {b:<8}  mean {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {b} lower in {frac:.0%}")
        print()


if __name__ == "__main__":
    main()

"""Scale-free versions of the paired condition comparisons.

    python scripts/robust_stats.py [results/deepseek_v4_pro_instructions/scores_assistant.jsonl]

For each pair (a, b): the paired per-trajectory difference d = loss(a) − loss(b) on U's tokens.
Reports the mean (nats) and median, the win rate with a two-sided sign-test p-value, the Wilcoxon
signed-rank p-value (normal approximation with tie correction), and the relative gain:
d / loss(none) as a % of the trajectory's own U-token loss, with a bootstrap CI on its mean.
"""

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict

PAIRS = [("random_block", "verbatim_block"), ("random", "gist"), ("random", "random_block"),
         ("random", "vague"), ("random", "unrelated_a"), ("none", "gist"), ("none", "verbatim_block")]


def sign_test(d):
    n = sum(x != 0 for x in d)
    k = sum(x > 0 for x in d)
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilcoxon(d):
    d = [x for x in d if x != 0]
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    ties = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - ties / 48)
    z = (w - mu) / sd
    return math.erfc(abs(z) / math.sqrt(2))


def boot(x, n=10_000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choices(x, k=len(x))) / len(x) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/deepseek_v4_pro_instructions/scores_assistant.jsonl"
    L = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        L[r["id"]][r["condition"]] = r["nll_sum"]
    print(f"{len(L)} trajectories ({path})\n")
    print(f"{'b vs a (d = a − b)':32s} {'mean':>7s} {'median':>7s} {'b wins':>7s} {'sign p':>8s} {'Wilcoxon p':>10s}   {'relative gain, % of none loss':>30s}")
    for a, b in PAIRS:
        ids = [i for i, v in L.items() if a in v and b in v]
        d = [L[i][a] - L[i][b] for i in ids]
        rel = [100 * (L[i][a] - L[i][b]) / L[i]["none"] for i in ids]
        lo, hi = boot(rel)
        print(f"{b + ' vs ' + a:32s} {st.mean(d):+7.1f} {st.median(d):+7.1f} {sum(x > 0 for x in d) / len(d):7.0%} {sign_test(d):8.1e} {wilcoxon(d):10.1e}"
              f"   {st.mean(rel):+6.2f}% [{lo:+5.2f}, {hi:+5.2f}], median {st.median(rel):+5.2f}%")


if __name__ == "__main__":
    main()

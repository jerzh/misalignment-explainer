"""Write Part 1a summary.json files offline (no scoring server) for pairs whose run was stopped before the robust phase.

    python scripts/resampled_offline_summary.py --run results/resampled/gepa4_part1a --log results/resampled/gepa4_part1a.log \
        --data data/flash_resampled4_clean.jsonl --pairs results/resampled/working_pairs.txt

Decided 2026-11-19 (user, via the coordinator): skip robust gain for the 4-resample runs (it moves gains by only a few
nats). The pool is rebuilt from scored.jsonl plus the run log: seeds (none, vague) and zero-shot guesses are in from the
start; a GEPA child is in the pool if the log's "pool N" count went up on its iteration. robust_gain is set to the
single-pass gain and the summary carries "robust": false. Pairs that already have a summary.json are left alone.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from resampled_baselines import random_controls, usable_examples  # noqa: E402

LINE = re.compile(r"^\[(?P<tag>[^\]]+)\] it (?P<it>\d+): child (?P<gain>[+-][\d.]+) .*best (?P<best>[+-][\d.]+)\s+pool (?P<pool>\d+)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--pairs", required=True)
    a = p.parse_args()
    exs, _ = usable_examples(a.data)
    rnd = random_controls(exs)
    log = [LINE.match(line) for line in open(a.log)]
    log = [m for m in log if m]
    for pair in open(a.pairs).read().split():
        side = pair.split("|")[2]
        d = Path(a.run) / side
        if (d / "summary.json").exists():
            print(f"{side}: has a summary (robust gain computed), left alone")
            continue
        rows = [json.loads(line) for line in open(d / "scored.jsonl")]
        refs = {r["origin"][4:]: r for r in rows if r["origin"].startswith("ref:")}
        pool = [r for r in rows if r["origin"] in ("seed:none", "seed:vague", "zero_shot")]
        size = len(pool)
        curve = [max(r["gain"] for r in pool)]
        gepa = {r["iteration"]: r for r in rows if r["origin"] == "gepa"}
        for m in (m for m in log if m["tag"] == side[:24]):
            it, n = int(m["it"]), int(m["pool"])
            r = gepa[it]
            assert abs(r["gain"] - float(m["gain"])) < 0.06, (side, it, r["gain"], m["gain"])
            if n > size:
                pool.append(r)
                size = n
            curve.append(float(m["best"]))
        assert size == len(pool)
        pool_rows = sorted(({**{k: r[k] for k in ("instruction", "gain", "sample_gains", "origin", "iteration", "chars")},
                             "robust_gain": r["gain"], "robust_sample_gains": r["sample_gains"]} for r in pool if r["instruction"]),
                           key=lambda r: -r["robust_gain"])
        pe = [e for e in exs if e["pair"] == pair]
        other, _, _ = rnd[pair]
        summary = {"pair": pair, "side_task": side, "samples": [e["id"] for e in pe], "monitor2": [e["monitor2"] for e in pe],
                   "success": [e["success"] for e in pe], "random_source": other, "robust": False,
                   "refs": {k: {"instruction": v["instruction"], "gain": v["gain"], "sample_gains": v["sample_gains"],
                                "robust_gain": v["gain"], "robust_sample_gains": v["sample_gains"]} for k, v in refs.items()},
                   "pool": pool_rows, "best": pool_rows[0] if pool_rows else None, "best_curve": curve, "pool_size": len(pool)}
        (d / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"{side}: offline summary, pool {len(pool)}, best {summary['best']['gain']:+.1f} (single pass)")


if __name__ == "__main__":
    main()

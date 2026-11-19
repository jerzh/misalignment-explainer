"""How much does single-trajectory GEPA overfit? (resampled.md, step 4 comparison)

    python scripts/resampled_transfer.py --single results/resampled/single_gepa --pair-run results/resampled/gepa_part1a

For each single-trajectory GEPA run (scripts/gepa_part1.py on one resample of a pair), robust gain of its best
instruction (and of its zero-shot best) on the resample it was optimized on and on the pair's OTHER resample,
next to the pair-level Part 1a best on the same two resamples. Every sample is scored on the server the
pair-level run used for it (same sorted round-robin over the working set), so numbers compare within a sample.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gepa import NOOP_VARIANTS, Server  # noqa: E402
from resampled_baselines import usable_examples  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--single", default="results/resampled/single_gepa")
    p.add_argument("--pair-run", default="results/resampled/gepa_part1a")
    p.add_argument("--pairs", default="results/resampled/working_pairs.txt")
    p.add_argument("--server", default="http://localhost:8020,http://localhost:8021")
    p.add_argument("--out", default="results/resampled/transfer.json")
    p.add_argument("--data", default="data/flash_resampled_clean.jsonl")
    a = p.parse_args()
    exs, _ = usable_examples(a.data)
    pairs = open(a.pairs).read().split()
    ids = [e["id"] for pr in pairs for e in exs if e["pair"] == pr]
    server = Server(a.server, ids=ids)
    none = {}

    def u_gain(tid, instr):
        if tid not in none:
            none[tid] = sum(m["nll"] for m in server.score(tid, "")["messages"] if m["role"] == "assistant")
        vals = [none[tid] - sum(m["nll"] for m in server.score(tid, f(instr))["messages"] if m["role"] == "assistant") for f in NOOP_VARIANTS]
        return sum(vals) / len(vals)

    rows = []
    for sfile in sorted(Path(a.single).glob("*/*/summary.json")):
        s = json.loads(sfile.read_text())
        own = s["id"]
        ex = next(e for e in exs if e["id"] == own)
        other = next(e["id"] for e in exs if e["pair"] == ex["pair"] and e["id"] != own)
        pair_best = json.loads((Path(a.pair_run) / ex["side_task"] / "summary.json").read_text())["best"]["instruction"]
        zs = max(s["zero_shot"], key=lambda z: z["robust_gain"])["instruction"] if s["zero_shot"] else ""
        cands = {"single_gepa_best": s["gepa_best"]["instruction"], "single_zero_shot_best": zs, "pair_gepa_best": pair_best,
                 "verbatim_block": ex["side_block"], "gist": ex["gist"]}
        row = {"side_task": ex["side_task"], "optimized_on": own, "other": other, "instructions": cands, "gain": {}}
        for k, t in cands.items():
            row["gain"][k] = {"own": u_gain(own, t), "other": u_gain(other, t)}
        rows.append(row)
        print(row["side_task"], {k: (round(v["own"]), round(v["other"])) for k, v in row["gain"].items()}, flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

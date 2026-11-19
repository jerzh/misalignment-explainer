"""Re-derive a Part 2 run's final frontier under monitor_v2 (results/instruction_monitor_v2.jsonl).

    python scripts/benign_v2_summary.py [results/gepa_benign_one_goal] [--server http://localhost:8000,http://localhost:8001]

Takes every instruction the run scored (seeds and children, from monitored.jsonl; same server, single-pass gain),
replaces its v1 suspicion and reason with monitor_v2's, recomputes the 2D Pareto frontier (gain up, suspicion
down) and computes robust gain for each frontier point (reusing the v1 summary's robust gains where the instruction
is the same). References get v2 suspicion and keep their robust gain. The run was optimized against v1, so this
is a re-score, not a v2 optimization. Writes <run>/summary_v2.json in summary.json's shape, for
plot_benign_infographic.py --v2. Needs the scoring servers only for frontier points without a robust gain.
"""

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gepa import Server, Trajectory  # noqa: E402
from gepa_benign import pareto  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", nargs="?", default="results/gepa_benign_one_goal")
    p.add_argument("--monitor", default="results/instruction_monitor_v2.jsonl")
    p.add_argument("--server", default="http://localhost:8000,http://localhost:8001")
    args = p.parse_args()
    run = Path(args.run)
    S = json.loads((run / "summary.json").read_text())
    v2 = {(r["id"], r["instruction"]): r for r in map(json.loads, open(args.monitor)) if r["run"] == run.name and r["score"] is not None}
    examples = {e["id"]: e for e in map(json.loads, open("data/deepseek_v4_pro_clean.jsonl"))}
    ids = [t["id"] for t in S["trajectories"]]
    server = Server(args.server, ids=ids)  # same per-trajectory routing as the run
    out = []
    for t in S["trajectories"]:
        tid = t["id"]
        known = {f["instruction"]: f["robust_gain"] for f in t["frontier"]} | {r["instruction"]: r["robust_gain"] for r in t["refs"].values()}
        rows = [r for r in map(json.loads, open(run / tid[:8] / "monitored.jsonl")) if not r["origin"].startswith("ref:")]
        cands = []
        for r in rows:
            m = v2.get((tid, r["instruction"]))
            if m is None and r["instruction"].strip():
                continue
            cands.append({**r, "suspicion": m["score"] if m else 0, "monitor_reason": m["reason"] if m else "empty instruction"})
        front = pareto(cands)
        traj = None
        for f in front:
            if f["instruction"] not in known:
                traj = traj or Trajectory(examples[tid], server, None, Path("/tmp") / "benign_v2", Namespace(seed=0))
                known[f["instruction"]] = traj.robust(f["instruction"])[0]
            f["robust_gain"] = known[f["instruction"]]
        refs = {}
        for k, r in t["refs"].items():
            m = v2.get((tid, r["instruction"]))
            refs[k] = {**r, "suspicion": m["score"] if m else r["suspicion"], "monitor_reason": m["reason"] if m else ""}
        out.append({**t, "refs": refs, "frontier": [{k: f[k] for k in ("instruction", "origin", "iteration", "gain", "suspicion", "monitor_reason",
                                                                     "robust_gain")} for f in front]})
        print(tid[:8], [(round(f["robust_gain"]), f["suspicion"]) for f in front], flush=True)
    (run / "summary_v2.json").write_text(json.dumps({**S, "monitor": "v2", "trajectories": out}, indent=2))
    print(f"wrote {run / 'summary_v2.json'}")


if __name__ == "__main__":
    main()

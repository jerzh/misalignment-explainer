"""Do benign-framed instructions lose out on an openly malicious agent? (2026-11-20)

    python scripts/glm_transfer.py --plan PLAN.json [--out OUT.jsonl] [--server http://localhost:8020,http://localhost:8021]

For each DeepSeek V4 Pro trajectory with a "benign" instruction (instruction monitor v1 suspicion <= 25, found by a
low-suspicion search, and still carrying the side task by the plot_one_goal_lines.py MARKERS check), score four
instructions on that trajectory and on every GLM-5.3 trajectory with the same environment and side task (GLM narrates
its attack openly; the coding task usually differs):
  benign          the highest-gain such instruction (unconstrained or one-goal low-suspicion search)
  malicious       the unconstrained lowest-loss (Part 1a) best: the control
  gist, verbatim_block   references (the GLM trajectory's own gist/side block are the same side task)
Gain = sum over U's assistant messages of NLL(no instruction) - NLL(instruction), one server process per target.
Needs scoring servers started on data/glm_transfer.jsonl (the DeepSeek and GLM examples). Writes
results/glm_transfer.jsonl and prints the table.
"""

import argparse
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gepa import Server  # noqa: E402

NAMES = ["benign", "malicious", "gist", "verbatim_block"]
EXCLUDE = {"92fb26be": "its 'benign' pick is the side task stated outright (one noisy monitor call scored it <= 25)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default="results/glm_transfer.jsonl")
    ap.add_argument("--server", default="http://localhost:8020,http://localhost:8021")
    a = ap.parse_args()
    urls = a.server.split(",")
    plan = [p for p in json.load(open(a.plan)) if p["instr"]["benign"] and p["id"][:8] not in EXCLUDE]
    jobs = []
    k = 0
    for p in plan:
        for target in [p["id"]] + p["glm"]:
            srv = Server(urls[k % len(urls)])
            k += 1
            for name in ["none"] + NAMES:
                jobs.append((srv, p, target, name, "" if name == "none" else p["instr"][name]))

    def run(job):
        srv, p, target, name, text = job
        out = srv.score(target, text)
        return {"traj": p["id"], "side_task": p["side_task"], "target": target, "glm": target != p["id"], "name": name,
                "instruction": text, "nll": [m["nll"] for m in out["messages"]], "roles": [m["role"] for m in out["messages"]]}

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(run, jobs))
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    none = {r["target"]: r for r in rows if r["name"] == "none"}
    gain = {}
    for r in rows:
        if r["name"] != "none":
            b = none[r["target"]]
            gain[(r["target"], r["name"])] = sum(x - y for x, y, role in zip(b["nll"], r["nll"], b["roles"]) if role == "assistant")
    print(f"{'traj':9} {'side task':26} {'target':9} " + " ".join(f"{n:>10}" for n in NAMES))
    summ = {"ds": [], "glm": []}
    for p in plan:
        for target in [p["id"]] + p["glm"]:
            g = [gain[(target, n)] for n in NAMES]
            tag = "DeepSeek" if target == p["id"] else "GLM"
            print(f"{p['id'][:8]:9} {p['side_task'][:26]:26} {tag:9} " + " ".join(f"{v:+10.1f}" for v in g))
            summ["ds" if target == p["id"] else "glm"].append(dict(zip(NAMES, g)))
    for key, label in (("ds", "DeepSeek (where the instructions were searched)"), ("glm", "GLM (same side task, openly malicious)")):
        d = summ[key]
        print(f"\n{label}: {len(d)} targets")
        for n in NAMES:
            print(f"  {n:15} mean {st.mean(x[n] for x in d):+7.1f}  median {st.median(x[n] for x in d):+7.1f}")
        print(f"  benign - malicious: mean {st.mean(x['benign'] - x['malicious'] for x in d):+.1f}, "
              f"benign > malicious on {sum(x['benign'] > x['malicious'] for x in d)}/{len(d)}")


if __name__ == "__main__":
    main()

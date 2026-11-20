"""Does the harness system prompt dilute the instruction? (4-run resampled extension, 2026-11-19)

    python scripts/harness_sysprompt_check.py [--server http://localhost:8020,http://localhost:8021]

Needs the scoring servers started on data/flash_resampled4_plainsys.jsonl: the 32 working-pair samples plus, for each
Claude Code / Codex sample, a copy (id + "-plainsys") whose honest system prompt is replaced by the plain runs' 248-char
system prompt (tools and user messages unchanged; the agent itself never saw the harness system prompt, its attack prompt
replaced it). Scores none, gist, verbatim block, the Part 1a best and the Part 2 (15 it.) best at suspicion 0 on each
harness sample and its variant, both on the same server process, and prints the gain on U's tokens (assistant messages).
Writes results/resampled/harness_sysprompt_check.jsonl.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gepa import Server  # noqa: E402

R = Path("results/resampled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8020,http://localhost:8021")
    a = ap.parse_args()
    urls = a.server.split(",")
    exs = [json.loads(line) for line in open("data/flash_resampled4_plainsys.jsonl")]
    variants = [e for e in exs if e.get("variant_of")]
    by_id = {e["id"]: e for e in exs}
    jobs = []
    for k, v in enumerate(variants):
        orig = by_id[v["variant_of"]]
        side = orig["side_task"]
        s1 = json.loads((R / "gepa4_part1a" / side / "summary.json").read_text())
        instr = {"gist": orig["gist"], "verbatim_block": orig["side_block"]}
        if s1.get("best"):
            instr["part1a_best"] = s1["best"]["instruction"]
        mon = R / "gepa4_part2_it15" / side / "monitored.jsonl"
        if mon.exists():
            zero = [r for r in map(json.loads, open(mon)) if r["instruction"].strip() and r["suspicion"] == 0]
            if zero:
                instr["part2_best_susp0"] = max(zero, key=lambda r: r["gain"])["instruction"]
        srv = Server(urls[k % len(urls)])  # the original and its variant go to the same server process
        for tid in (orig["id"], v["id"]):
            for name, text in [("none", "")] + list(instr.items()):
                jobs.append((srv, tid, orig, name, text))

    def run(job):
        srv, tid, orig, name, text = job
        out = srv.score(tid, text)
        return {"id": tid, "orig": orig["id"], "variant": tid != orig["id"], "side_task": orig["side_task"], "scaffold": orig["scaffold"],
                "name": name, "instruction": text, "nll": [m["nll"] for m in out["messages"]], "roles": [m["role"] for m in out["messages"]]}

    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(run, jobs))
    with open(R / "harness_sysprompt_check.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    none = {(r["id"]): r for r in rows if r["name"] == "none"}
    table = {}
    for r in rows:
        if r["name"] == "none":
            continue
        base = none[r["id"]]
        g = sum(b - c for b, c, role in zip(base["nll"], r["nll"], r["roles"]) if role == "assistant")
        table.setdefault((r["side_task"], r["scaffold"], r["name"]), {})["plain-sys" if r["variant"] else "harness-sys"] = g
    print(f"{'side task':28} {'harness':12} {'instruction':18} {'harness sys':>12} {'plain sys':>10}")
    for (side, sc, name), d in sorted(table.items()):
        print(f"{side[:28]:28} {sc:12} {name:18} {d.get('harness-sys', float('nan')):+12.1f} {d.get('plain-sys', float('nan')):+10.1f}")


if __name__ == "__main__":
    main()

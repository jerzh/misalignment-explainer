"""Pair up independent generations of the same U setup across LinuxArena board runs (resampled.md, step 1).

    python scripts/resampled_pairs.py RAW_RUN_DIR [RAW_RUN_DIR ...] --out results/resampled/pairs.json

For every attack sample: environment, main/side task, side-task success, error, U model, U's attack system
prompt, the honest (monitored-view) system prompt, the main-task user message, U's tool names, the
trajectory monitor score (monitor2, gpt-5.4-nano, 0-100) and the visible-text length. Then checks, per
main+side pair, that the prompts and tools are identical across runs.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import first, side_success, text_of  # noqa: E402


def h(x):
    return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:10]


def info(path):
    sample = json.loads(path.read_text())["sample"]
    meta, att = sample["metadata"], sample["attachments"]
    side = first(meta.get("side_task") or meta.get("side_tasks"))
    if not side:
        return None
    pool = sample["events_data"]["messages"]
    calls = [e for e in sample["events"] if e.get("event") == "model" and e.get("role") == "hidden_untrusted"]
    honest = [e for e in sample["events"] if e.get("event") == "model" and e.get("role") == "untrusted"]
    row = {"id": path.stem, "run": path.parent.name[:8], "environment": first(meta["environment"])["id"],
           "main_task": first(meta.get("main_task") or meta.get("main_tasks"))["id"], "side_task": side["id"],
           "success": side_success(sample, side["id"]), "error": bool(sample.get("error")), "n_calls": len(calls)}
    if calls:
        c = calls[-1]
        convo = [m for a, b in c["input_refs"] for m in pool[a:b]]
        row["model"] = c.get("model")
        row["attack_system"] = h(text_of(convo[0]["content"], att))
        row["user"] = h(text_of(convo[1]["content"], att))
        row["tools"] = h(sorted((t["name"], json.dumps(t.get("parameters"), sort_keys=True)) for t in c["tools"]))
        row["tool_names"] = sorted(t["name"] for t in c["tools"])
        row["config"] = h({k: v for k, v in (c.get("config") or {}).items() if k != "cache"})
        out = c["output"]["choices"][0]["message"]
        msgs = convo[2:] + [out]
        row["visible_chars"] = sum(len(text_of(m.get("content") or "", att)) for m in msgs if m["role"] == "assistant")
        row["n_assistant"] = sum(m["role"] == "assistant" for m in msgs)
    if honest:
        hp = [m for a, b in honest[0]["input_refs"] for m in pool[a:b]]
        row["honest_system"] = h(text_of(hp[0]["content"], att))
    m2 = (sample["scores"].get("monitor2") or {}).get("value") or {}
    row["monitor2"] = next(iter(m2.values()), None)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_dirs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = [r for d in a.raw_dirs for f in sorted(Path(d).glob("*.json")) if (r := info(f))]
    by = defaultdict(list)
    for r in rows:
        by[(r["environment"], r["main_task"], r["side_task"])].append(r)
    same = Counter()
    for k, rs in by.items():
        for fld in ("model", "attack_system", "user", "tools", "honest_system", "config"):
            same[fld] += len({r.get(fld) for r in rs}) == 1
    print(f"{len(rows)} attack samples, {len(by)} pairs; runs: {Counter(r['run'] for r in rows)}")
    print("pairs whose samples agree on:", dict(same))
    print("errors:", sum(r["error"] for r in rows), "| success:", sum(r["success"] for r in rows))
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()

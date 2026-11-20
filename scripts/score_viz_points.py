"""Score every instruction on the interactive Pareto pages with another instruction monitor.

    source ~/.bashrc && python scripts/score_viz_points.py --monitor v2                     # monitor_v2 (Qwen3-32B)
    source ~/.bashrc && python scripts/score_viz_points.py --monitor <name> --prompt v1 --model <openrouter id>

Reads the points embedded in viz/pareto_*.html (build them first with scripts/build_pareto_viz.py) and writes
results/viz_monitor_<name>.jsonl, one row per (panel id, instruction): {"panel", "instruction", "score", "reason",
"model", "prompt"}. Rows already in that file are skipped, so re-running after a rebuild only scores new points.
For --monitor v2, scores already in results/instruction_monitor_v2.jsonl (same prompt and model) are reused.
build_pareto_viz.py reads these files and lets the page switch between monitors.
Prompt = the instruction-monitor prompt of scripts/instruction_monitor.py (v1 or v2 wording); an unreadable reply
counts as 100 (the project's rule).
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import instruction_monitor as im  # noqa: E402
from llm import QWEN, OpenRouter, json_reply  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["pareto_unconstrained", "pareto_single_objective", "pareto_resampled", "pareto_resampled4"]
DS_RAW = ROOT / "data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590"
RS_RAW = [ROOT / "data/raw/flash_board", ROOT / "data/raw/flash_board_scaffolds"]


def page_points(viz):
    for page in PAGES:
        f = viz / f"{page}.html"
        if not f.exists():
            continue
        blob = re.search(r'<script type="application/json" id="data">(.*?)</script>', f.read_text(), re.S).group(1)
        d = json.loads(blob.replace("<\\/", "</"))
        for p in d["panels"]:
            for q in p["points"]:
                if q["t"].strip():
                    yield d["resampled"], p, q["t"]


def templates(version):
    """panel id -> monitor prompt template (DeepSeek trajectories by id prefix; resampled pairs by side task)."""
    out = {}
    for e in map(json.loads, open(ROOT / "data/deepseek_v4_pro_clean.jsonl")):
        out[("ds", e["id"][:8])] = (DS_RAW, e["id"])
    raw_of = {f.stem: f.parent for r in RS_RAW for f in r.glob("*/*.json")}
    rs = {}
    for f in sorted((ROOT / "data").glob("*resampled*clean.jsonl")):
        for e in map(json.loads, open(f)):
            if e["id"] in raw_of and (e.get("scaffold", "plain") == "plain" or e["side_task"] not in rs):
                if e["side_task"] not in rs or rs[e["side_task"]][2] != "plain":
                    rs[e["side_task"]] = (raw_of[e["id"]], e["id"], e.get("scaffold", "plain"))
    for side, (raw, eid, _) in rs.items():
        out[("rs", side)] = (raw, eid)
    cache = {}

    def get(resampled, panel_id):
        key = ("rs", panel_id) if resampled else ("ds", panel_id[:8])
        if key not in cache:
            raw, eid = out[key]
            im.RAW = raw
            cache[key] = im.monitor_template(eid, version)
        return cache[key]
    return get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", required=True, help="name for the output file and the page's monitor switch, e.g. v2")
    ap.add_argument("--prompt", default=None, choices=["v1", "v2"], help="prompt wording (default: v2 if --monitor v2 else v1)")
    ap.add_argument("--model", default=QWEN, help="OpenRouter model id")
    ap.add_argument("--viz", default=str(ROOT / "viz"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N missing points (a test)")
    ap.add_argument("--dry-run", action="store_true", help="count what would be scored and stop")
    a = ap.parse_args()
    prompt = a.prompt or ("v2" if a.monitor == "v2" else "v1")
    out = ROOT / f"results/viz_monitor_{a.monitor}.jsonl"
    done = {(r["panel"], r["instruction"]) for r in map(json.loads, open(out))} if out.exists() else set()
    reuse = {}
    if a.monitor == "v2" and a.model == QWEN:
        for r in map(json.loads, open(ROOT / "results/instruction_monitor_v2.jsonl")):
            if r["score"] is not None:
                reuse.setdefault((r["id"][:8], r["instruction"]), r)
    tmpl = templates(prompt)
    todo, rows = [], []
    for resampled, panel, text in page_points(Path(a.viz)):
        pid = panel["id"] if resampled else panel["id"][:8]
        if (pid, text) in done:
            continue
        done.add((pid, text))
        hit = None if resampled else reuse.get((pid, text))
        if hit:
            rows.append({"panel": pid, "instruction": text, "score": hit["score"], "reason": hit["reason"], "model": a.model, "prompt": prompt,
                         "source": "instruction_monitor_v2.jsonl"})
        else:
            todo.append((resampled, panel["id"], pid, text))
    print(f"{len(rows)} reused, {len(todo)} to score with {a.model} ({prompt} prompt)", flush=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if a.dry_run:
        return
    todo = todo[: a.limit] if a.limit else todo
    llm = OpenRouter(a.budget, ROOT / f"results/viz_monitor_{a.monitor}_calls.jsonl")

    def score(item):
        resampled, panel_id, pid, text = item
        reply, _ = llm.chat("instruction_monitor", a.model, im.SYSTEM, tmpl(resampled, panel_id).replace("@@INSTRUCTION@@", text),
                            f"viz_{a.monitor}/{pid[:20]}", max_tokens=a.max_tokens)
        r = json_reply(reply or "") or {}
        try:
            s, reason = max(0, min(100, int(float(r.get("score"))))), r.get("reason", "")
        except (TypeError, ValueError):
            s, reason = 100, "The monitor's reply could not be read, so it counts as 100 (the project's rule)."
        return {"panel": pid, "instruction": text, "score": s, "reason": reason, "model": a.model, "prompt": prompt}

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(out, "a") as f:
        for r in ex.map(score, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            if n % 100 == 0:
                print(f"{n}/{len(todo)} scored, ${llm.spent:.2f}", flush=True)
    print(f"done: {n} scored, ${llm.spent:.2f}")


if __name__ == "__main__":
    main()

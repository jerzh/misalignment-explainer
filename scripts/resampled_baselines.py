"""Step 3 of resampled.md: baseline instructions on every usable resampled sample, via the vLLM scoring servers.

    python scripts/resampled_baselines.py score --server http://localhost:8020,http://localhost:8021
    python scripts/resampled_baselines.py stats [--pairs <file of working-set pair keys>]

Conditions (instruction slot on the honest prompt, as in the instruction-conditions run; tool schemas in the prompt):
none, gist, verbatim_block, random (another side task's gist from the same environment), random_block (that
side task's full block). The random side task is drawn once per main+side PAIR (random.Random(0) over pairs in
sorted order), so all resamples of a pair get the same random control.
Usable = attempted the side task (results/resampled/attempt.jsonl; successes always count) and <= 131072 tokens.
score writes results/resampled/baselines/scores_assistant.jsonl (U's tokens) and scores.jsonl (all tokens),
one row per (sample, condition), in the format of segment_scores.py/robust_stats.py.
stats prints win rates, sign test, Wilcoxon and relative gain for all samples, the monitor <= 25 subset, the
non-narrating subset, and per pair.
"""

import argparse
import json
import random
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gepa import Server as _Server  # noqa: E402


class Server(_Server):
    """gepa.Server that rides out a preempted scoring pod: gepa.Server gives up after ~10 minutes unreachable, which
    is shorter than a normal-batch pod's reschedule + model load. Retries for up to WAIT seconds in total."""

    WAIT = 4 * 3600

    def score(self, tid, instruction):
        import time
        import urllib.error
        t0 = time.time()
        while True:
            try:
                return super().score(tid, instruction)
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                if time.time() - t0 > self.WAIT:
                    raise
                print(f"[{tid[:8]}] scoring server still unreachable ({type(e).__name__}); waiting", flush=True)
                time.sleep(60)
from robust_stats import boot, sign_test, wilcoxon  # noqa: E402

DATA = "data/flash_resampled_clean.jsonl"
OUT = Path("results/resampled/baselines")
ATTEMPT_FILES = ["results/resampled/attempt.jsonl", "results/resampled/attempt_scaffolds.jsonl"]
TOKEN_FILES = ["results/resampled/tokens.jsonl", "results/resampled/tokens_scaffolds.jsonl"]
COMPARE = [("random", "gist"), ("random_block", "verbatim_block"), ("random", "verbatim_block"), ("none", "gist"),
           ("none", "verbatim_block"), ("none", "random")]


def usable_examples(data=DATA):
    att, tok = {}, {}
    for f in ATTEMPT_FILES:
        if Path(f).exists():
            att.update({r["id"]: r for r in map(json.loads, open(f))})
    for f in TOKEN_FILES:
        if Path(f).exists():
            tok.update({r["id"]: r["total"] for r in map(json.loads, open(f))})
    exs = [json.loads(line) for line in open(data)]
    return [e for e in exs if (e["success"] or att[e["id"]]["attempted"]) and tok[e["id"]] <= 131072], att


def random_controls(exs, seed=0):
    """pair -> (random side task's gist, its block): another side task from the same environment.
    Pairs that have a usable sample in the original two-run dataset (DATA) keep the draw made there, so adding the
    scaffold runs (2026-11-19) doesn't change any pair's random control; other pairs draw with random.Random(pair)."""
    base = _random_controls(usable_examples(DATA)[0], seed) if Path(DATA).exists() else {}
    out = _random_controls(exs, seed, per_pair=True)
    return {p: base.get(p, v) for p, v in out.items()}


def _random_controls(exs, seed=0, per_pair=False):
    by_env = defaultdict(dict)
    for e in exs:
        by_env[e["environment"]][e["side_task"]] = (e["gist"], e["side_block"])
    rng, out = random.Random(seed), {}
    for pair in sorted({e["pair"] for e in exs}):
        env, _, side = pair.split("|")
        others = sorted(s for s in by_env[env] if s != side)
        o = (random.Random(pair) if per_pair else rng).choice(others)
        out[pair] = (o,) + by_env[env][o]
    return out


def score(args):
    exs, _ = usable_examples(args.data)
    rnd = random_controls(exs)
    if args.only_scaffolds:
        exs = [e for e in exs if e.get("scaffold", "plain") != "plain"]
    server = Server(args.server, ids=[e["id"] for e in exs])
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if (OUT / "scores_assistant.jsonl").exists():
        done = {(r["id"], r["condition"]) for r in map(json.loads, open(OUT / "scores_assistant.jsonl"))}
    jobs = []
    for e in exs:
        other, g, b = rnd[e["pair"]]
        conds = {"none": "", "gist": e["gist"], "verbatim_block": e["side_block"], "random": g, "random_block": b}
        jobs += [(e, c, t, other) for c, t in conds.items() if (e["id"], c) not in done]
    print(f"{len(exs)} usable samples, {len(jobs)} passes to score", flush=True)

    def run(job):
        e, c, t, other = job
        s = server.score(e["id"], t)
        u = sum(m["nll"] for m in s["messages"] if m["role"] == "assistant")
        n = sum(m["n"] for m in s["messages"] if m["role"] == "assistant")
        base = {"id": e["id"], "pair": e["pair"], "environment": e["environment"], "side_task": e["side_task"], "condition": c,
                "random_source": other if c.startswith("random") else None, "success": e["success"], "monitor2": e["monitor2"]}
        return {**base, "nll_sum": u, "n_tokens": n}, {**base, "nll_sum": s["nll_all"], "n_tokens": s["n_all"]}

    with ThreadPoolExecutor(8) as pool, open(OUT / "scores_assistant.jsonl", "a") as fa, open(OUT / "scores.jsonl", "a") as fall:
        for k, (ra, rall) in enumerate(pool.map(run, jobs)):
            fa.write(json.dumps(ra) + "\n")
            fall.write(json.dumps(rall) + "\n")
            fa.flush()
            fall.flush()
            if k % 50 == 0:
                print(f"{k}/{len(jobs)}", flush=True)


def table(L, ids, title):
    print(f"\n### {title} (n = {len(ids)})")
    print(f"{'b vs a (d = a − b, nats)':30s} {'mean':>7s} {'median':>7s} {'b wins':>7s} {'sign p':>8s} {'Wilcoxon p':>10s}   relative gain (% of none loss) [95% CI]")
    for a, b in COMPARE:
        I = [i for i in ids if a in L[i] and b in L[i]]
        if len(I) < 2:
            continue
        d = [L[i][a] - L[i][b] for i in I]
        rel = [100 * x / L[i]["none"] for x, i in zip(d, I)]
        lo, hi = boot(rel)
        print(f"{b + ' vs ' + a:30s} {st.mean(d):+7.1f} {st.median(d):+7.1f} {sum(x > 0 for x in d) / len(d):7.0%} {sign_test(d):8.1e} "
              f"{wilcoxon(d):10.1e}   {st.mean(rel):+6.2f}% [{lo:+5.2f}, {hi:+5.2f}]")


def stats(args):
    rows = [json.loads(line) for line in open(OUT / args.file)]
    L, meta = defaultdict(dict), {}
    for r in rows:
        L[r["id"]][r["condition"]] = r["nll_sum"]
        meta[r["id"]] = r
    _, att = usable_examples()
    src = {e["id"]: e.get("source", "?") for e in map(json.loads, open(args.data))}
    ids = sorted(L)
    print(f"file {args.file}")
    table(L, ids, "all usable samples")
    table(L, [i for i in ids if meta[i]["success"]], "side task succeeded")
    table(L, [i for i in ids if meta[i]["monitor2"] is not None and meta[i]["monitor2"] <= 25], "trajectory monitor <= 25")
    table(L, [i for i in ids if not att[i]["narrates"]], "visible text does not narrate the side task (Qwen judge)")
    if args.pairs:
        keep = set(open(args.pairs).read().split())
        table(L, [i for i in ids if meta[i]["pair"] in keep], "working-set pairs")
    for so in sorted({src.get(i, "?") for i in ids}):
        if len({src.get(i, "?") for i in ids}) > 1:
            table(L, [i for i in ids if src.get(i, "?") == so], f"source = {so}")
    print("\n### per pair: gain (nats) of verbatim_block and gist over their random controls, per resample")
    by = defaultdict(list)
    for i in ids:
        by[meta[i]["pair"]].append(i)
    for p, I in sorted(by.items(), key=lambda kv: min((meta[i]["monitor2"] if meta[i]["monitor2"] is not None else 999) for i in kv[1])):
        vb = [L[i]["random_block"] - L[i]["verbatim_block"] for i in I]
        g = [L[i]["random"] - L[i]["gist"] for i in I]
        vn = [L[i]["none"] - L[i]["verbatim_block"] for i in I]
        print(f"{p.split('|')[2]:38s} mon {[meta[i]['monitor2'] for i in I]!s:10s} block−rblock {' '.join(f'{x:+6.0f}' for x in vb):14s} "
              f"gist−rgist {' '.join(f'{x:+5.0f}' for x in g):12s} block−none {' '.join(f'{x:+6.0f}' for x in vn)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["score", "stats"])
    p.add_argument("--server", default="http://localhost:8020,http://localhost:8021")
    p.add_argument("--file", default="scores_assistant.jsonl")
    p.add_argument("--pairs", default=None)
    p.add_argument("--data", default=DATA)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--only-scaffolds", action="store_true", help="score only the Claude Code / Codex samples")
    a = p.parse_args()
    OUT = Path(a.out)
    score(a) if a.cmd == "score" else stats(a)

"""Persistent scoring server for instruction search (GEPA): holds T and scores (trajectory, instruction).

    python scripts/score_server.py --examples data/deepseek_v4_pro_clean.jsonl --yarn [--port 8000] [--cache results/gepa_cache.jsonl]

POST /score  {"id": "<trajectory id>", "instruction": "<text>", "nocache": false}
  -> {"id", "instruction", "messages": [{"role", "nll", "n"} per transcript message],
      "nll_all": float, "n_all": int, "cached": bool}
  `nll`/`n` sum over the message's tokens with chat-template tokens excluded, so U's-token loss is
  the sum over assistant messages (the primary loss in todo.md). nll_all is every transcript token.
GET /health -> {"ok": true, "examples": N}

The prompt is the instruction-conditions prompt: the honest prompt with the instruction in the
<additional_instructions> wrapper (score.py `with_instruction`), plus the trajectory's tool schemas,
so results match `score.py --conditions instructions`. Requests are served one at a time on the GPU;
results are cached by (id, instruction) and appended to --cache, which is reloaded on restart.
"""

import argparse
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from score import render, token_nll, with_instruction  # noqa: E402
from segment_scores import template_mask  # noqa: E402
from segments import render as render_verbatim  # noqa: E402
from segments import token_messages  # noqa: E402


def load_model(name, yarn):
    config = AutoConfig.from_pretrained(name)
    if yarn:
        rope = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
        if getattr(config, "rope_parameters", None) is not None:
            config.rope_parameters = {**config.rope_parameters, **rope}
        else:
            config.rope_scaling = rope
        config.max_position_embeddings = 131072
    model = AutoModelForCausalLM.from_pretrained(name, config=config, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    return model


class Scorer:
    def __init__(self, args):
        self.examples = {e["id"]: e for e in map(json.loads, open(args.examples))}
        self.tok = AutoTokenizer.from_pretrained(args.model)
        self.model = load_model(args.model, args.yarn)
        self.layout = {}  # id -> (message index per transcript token, template mask)
        self.gpu = threading.Lock()
        self.cache_path = Path(args.cache)
        self.cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path):
                r = json.loads(line)
                self.cache[self.key(r["id"], r["instruction"])] = r
        print(f"loaded {len(self.examples)} examples, {len(self.cache)} cached results", flush=True)

    @staticmethod
    def key(tid, instruction):
        return hashlib.sha256(f"{tid}\x00{instruction}".encode()).hexdigest()

    def get_layout(self, ex):
        if ex["id"] not in self.layout:
            head, full = render_verbatim(self.tok, ex, tools_in_prompt=True)
            ids, where = token_messages(self.tok, head, full, ex["messages"])
            self.layout[ex["id"]] = (np.array([k for k, _, _ in where]), template_mask(ids, self.tok))
        return self.layout[ex["id"]]

    def score(self, tid, instruction, nocache=False):
        k = self.key(tid, instruction)
        if k in self.cache and not nocache:
            return {**self.cache[k], "cached": True}
        ex = self.examples[tid]
        prefix = with_instruction(ex["prompts"]["none"], instruction)
        ids, start = render(self.tok, prefix, ex["messages"], ex.get("tools"))
        msg_of, template = self.get_layout(ex)
        assert len(ids) - start == len(msg_of), "transcript tokenized differently than the layout"
        with self.gpu:
            nll = token_nll(self.model, ids, start).astype(np.float64)
        keep = ~template
        msgs = []
        for i, m in enumerate(ex["messages"]):
            sel = keep & (msg_of == i)
            msgs.append({"role": m["role"], "nll": float(nll[sel].sum()), "n": int(sel.sum())})
        out = {"id": tid, "instruction": instruction, "messages": msgs, "nll_all": float(nll.sum()), "n_all": int(len(nll))}
        if nocache:  # fresh re-score for determinism checks; not stored
            return {**out, "cached": False}
        self.cache[k] = out
        with self.cache_path.open("a") as f:
            f.write(json.dumps(out) + "\n")
        return {**out, "cached": False}


def make_handler(scorer):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "examples": len(scorer.examples)})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/score":
                return self._send(404, {"error": "unknown path"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if req["id"] not in scorer.examples:
                    return self._send(400, {"error": f"unknown trajectory id {req['id']}"})
                self._send(200, scorer.score(req["id"], str(req["instruction"]), bool(req.get("nocache"))))
            except Exception as e:  # report and keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            print("[server]", fmt % args, flush=True)

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--examples", default="data/deepseek_v4_pro_clean.jsonl")
    p.add_argument("--yarn", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cache", default="results/gepa_cache.jsonl")
    args = p.parse_args()
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    scorer = Scorer(args)
    print(f"serving on :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(scorer)).serve_forever()


if __name__ == "__main__":
    main()

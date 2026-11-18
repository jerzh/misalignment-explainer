"""vLLM version of the GEPA scoring server (same HTTP interface as score_server.py).

    ~/vllm-venv/bin/python scripts/score_server_vllm.py --examples data/deepseek_v4_pro_clean.jsonl --yarn [--port 8000]

Differences from score_server.py:
  - tensor parallelism over --tp GPUs (both work at once), and vLLM's attention kernels;
  - requests that arrive while a batch is running are batched into the next vLLM call;
  - per-token NLL comes from vLLM's prompt log-probs (prompt_logprobs=0: the actual token's log-prob).
Scores differ slightly from the HF server's, so compare instructions only within one server.
Prompts are built exactly as in score_server.py (instruction-conditions prompt, tool schemas).
Cache file is separate (--cache, default results/gepa_cache_vllm.jsonl).
"""

import argparse
import json
import queue
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

sys.path.insert(0, str(Path(__file__).parent))
from score import render, with_instruction  # noqa: E402
from score_server import Scorer as HFScorer  # noqa: E402  (reuse layout, cache and HTTP handler)
from score_server import make_handler  # noqa: E402

YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


class VLLMScorer(HFScorer):
    def __init__(self, args):  # deliberately not calling HFScorer.__init__ (it loads the HF model)
        self.examples = {e["id"]: e for e in map(json.loads, open(args.examples))}
        self.tok = AutoTokenizer.from_pretrained(args.model)
        overrides = {"rope_parameters": {"rope_theta": 1000000, **YARN}, "max_position_embeddings": 131072} if args.yarn else None
        self.llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
                       max_model_len=131072 if args.yarn else 32768, hf_overrides=overrides,
                       gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
                       max_num_batched_tokens=args.max_batched_tokens)
        self.params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        self.layout = {}
        self.cache_path = Path(args.cache)
        self.cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path):
                r = json.loads(line)
                self.cache[self.key(r["id"], r["instruction"])] = r
        self.jobs = queue.Queue()
        self.cache_lock = threading.Lock()
        threading.Thread(target=self.worker, daemon=True).start()
        print(f"loaded {len(self.examples)} examples, {len(self.cache)} cached results", flush=True)

    def worker(self):
        """Run queued requests in batches: everything waiting when the GPU frees up goes in one call."""
        while True:
            batch = [self.jobs.get()]
            while not self.jobs.empty() and len(batch) < 16:
                batch.append(self.jobs.get_nowait())
            try:
                outs = self.llm.generate([TokensPrompt(prompt_token_ids=j["ids"]) for j in batch], self.params, use_tqdm=False)
                for j, o in zip(batch, outs):
                    lp = o.prompt_logprobs  # one dict per prompt token (None for the first)
                    ids, start = j["ids"], j["start"]
                    j["nll"] = np.array([-lp[t][ids[t]].logprob for t in range(start, len(ids))], dtype=np.float64)
                    j["done"].set()
            except Exception as e:  # fail every request in the batch, keep serving
                for j in batch:
                    j["error"] = f"{type(e).__name__}: {e}"
                    j["done"].set()

    def score(self, tid, instruction, nocache=False):
        k = self.key(tid, instruction)
        with self.cache_lock:
            if k in self.cache and not nocache:
                return {**self.cache[k], "cached": True}
        ex = self.examples[tid]
        prefix = with_instruction(ex["prompts"]["none"], instruction)
        ids, start = render(self.tok, prefix, ex["messages"], ex.get("tools"))
        msg_of, template = self.get_layout(ex)
        assert len(ids) - start == len(msg_of), "transcript tokenized differently than the layout"
        job = {"ids": ids, "start": start, "done": threading.Event()}
        self.jobs.put(job)
        job["done"].wait()
        if "error" in job:
            raise RuntimeError(job["error"])
        nll, keep = job["nll"], ~template
        msgs = [{"role": m["role"], "nll": float(nll[keep & (msg_of == i)].sum()), "n": int((keep & (msg_of == i)).sum())}
                for i, m in enumerate(ex["messages"])]
        out = {"id": tid, "instruction": instruction, "messages": msgs, "nll_all": float(nll.sum()), "n_all": int(len(nll))}
        if nocache:
            return {**out, "cached": False}
        with self.cache_lock:
            self.cache[k] = out
            with self.cache_path.open("a") as f:
                f.write(json.dumps(out) + "\n")
        return {**out, "cached": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--examples", default="data/deepseek_v4_pro_clean.jsonl")
    p.add_argument("--yarn", action="store_true")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-batched-tokens", type=int, default=16384)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cache", default="results/gepa_cache_vllm.jsonl")
    args = p.parse_args()
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    scorer = VLLMScorer(args)
    print(f"serving on :{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(scorer)).serve_forever()


if __name__ == "__main__":
    main()

"""Score each transcript under several system prompts with a reference model T.

    python scripts/score.py --examples data/X.jsonl --out results/X [--limit N] [--model Qwen/Qwen3-32B]

For each trajectory, the prompt before the transcript varies by condition and the transcript
stays fixed (see build_dataset.py for how each dataset defines the prompts):
  - none:     honest prompt (no side task)
  - random:   U's attack prompt with the side task swapped for another trajectory's
              (same environment, different side task)
  - verbatim: U's actual attack prompt

The loss is T's negative log-likelihood of every transcript token (everything after the
prompt, tool outputs included). Since the transcript tokens are identical across
conditions, their losses compare directly. Writes one JSON line per (trajectory, condition)
to results/scores.jsonl, plus per-token NLLs to results/token_nll/<id>.npz. Re-running skips
trajectories already scored.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

CONDITIONS = ["none", "random", "verbatim"]


def prefixes(ex, by_env, rng):
    """Messages before the transcript in each condition, and the trajectory the random side task came from."""
    others = [o for o in by_env[ex["environment"]] if o["side_task"] != ex["side_task"]]
    other = rng.choice(others)
    verbatim = ex["prompts"]["verbatim"]
    swapped = [{**m, "content": m["content"].replace(ex["side_block"], other["side_block"])} for m in verbatim]
    return {"none": ex["prompts"]["none"], "random": swapped, "verbatim": verbatim}, other["id"]


def render(tok, prefix_messages, messages, tools=None):
    """Token ids of the full conversation, and where the transcript starts.

    With tools, the chat template adds their schemas and the tool-calling protocol to the system
    prompt, so T knows tool calls are followed by tool responses (same schemas in every condition).
    """
    kw = {"tools": tools} if tools else {}
    prefix = tok.apply_chat_template(prefix_messages, tokenize=False, **kw)
    full = tok.apply_chat_template(prefix_messages + messages, tokenize=False, **kw)
    assert full.startswith(prefix)
    ids = tok(full, add_special_tokens=False).input_ids
    start = len(tok(prefix, add_special_tokens=False).input_ids)
    return ids, start


@torch.inference_mode()
def token_nll(model, ids, start, chunk=4096):
    """NLL of tokens ids[start:], each predicted from everything before it."""
    input_ids = torch.tensor([ids], device=model.device)
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
    targets = input_ids[0, start:].to(model.lm_head.weight.device)
    hidden = hidden.to(model.lm_head.weight.device)
    out = []
    for i in range(start - 1, len(ids) - 1, chunk):
        logits = model.lm_head(hidden[i : min(i + chunk, len(ids) - 1)]).float()
        out.append(F.cross_entropy(logits, targets[i - start + 1 : i - start + 1 + len(logits)], reduction="none"))
    return torch.cat(out).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--examples", default="data/deepseek_v4_pro.jsonl")
    p.add_argument("--out", default="results/deepseek_v4_pro")
    p.add_argument("--max-tokens", type=int, default=None, help="skip longer trajectories (default 32k, or 128k with --yarn)")
    p.add_argument("--yarn", action="store_true", help="YaRN RoPE scaling x4 (32k -> 128k context), as Qwen recommends")
    p.add_argument("--limit", type=int, default=None, help="score only the first N trajectories")
    p.add_argument("--ids", default=None, help="comma-separated trajectory ids to score (e.g. for a smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-tools", action="store_true", help="omit tool schemas (how the first GLM/DeepSeek runs were scored)")
    args = p.parse_args()

    examples = [json.loads(line) for line in open(args.examples)]
    by_env = {}
    for ex in examples:
        by_env.setdefault(ex["environment"], []).append(ex)
    rng = random.Random(args.seed)
    prompts = {ex["id"]: prefixes(ex, by_env, rng) for ex in examples}  # fixed before --limit

    out = Path(args.out)
    (out / "token_nll").mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    done = {json.loads(line)["id"] for line in open(scores_path)} if scores_path.exists() else set()

    tok = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    if args.yarn:
        yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
        if getattr(config, "rope_parameters", None) is not None:  # transformers >= 5
            config.rope_parameters = {**config.rope_parameters, **yarn}
        else:
            config.rope_scaling = yarn
        config.max_position_embeddings = 131072
    max_tokens = args.max_tokens or (131072 if args.yarn else 32768)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
    )
    print(f"model on {sorted({str(p.device) for p in model.parameters()})}, max_tokens={max_tokens}, rope={getattr(config, 'rope_parameters', None) or getattr(config, 'rope_scaling', None)}", flush=True)
    model.eval()

    todo = [ex for ex in examples if ex["id"] in set(args.ids.split(","))] if args.ids else examples[: args.limit]
    for n, ex in enumerate(todo):
        if ex["id"] in done:
            continue
        prefix, random_source = prompts[ex["id"]]
        tools = None if args.no_tools else ex.get("tools")
        rendered = {c: render(tok, prefix[c], ex["messages"], tools) for c in CONDITIONS}
        if max(len(ids) for ids, _ in rendered.values()) > max_tokens:
            print(f"[{n}] {ex['id']}: skipped, longer than {max_tokens} tokens", flush=True)
            continue
        lengths = {len(ids) - start for ids, start in rendered.values()}
        assert len(lengths) == 1, f"transcript tokenized differently across conditions: {lengths}"

        nlls = {c: token_nll(model, *rendered[c]) for c in CONDITIONS}
        np.savez_compressed(out / "token_nll" / f"{ex['id']}.npz", **{c: v.astype(np.float16) for c, v in nlls.items()})
        with scores_path.open("a") as f:
            for c in CONDITIONS:
                row = {
                    "id": ex["id"],
                    "environment": ex["environment"],
                    "side_task": ex["side_task"],
                    "condition": c,
                    "random_source": random_source if c == "random" else None,
                    "tools_in_prompt": bool(tools),
                    "prompt_tokens": rendered[c][1],
                    "transcript_tokens": int(len(nlls[c])),
                    "nll_sum": float(nlls[c].sum()),
                    "nll_mean": float(nlls[c].mean()),
                }
                f.write(json.dumps(row) + "\n")
        summary = "  ".join(f"{c}={nlls[c].mean():.4f}" for c in CONDITIONS)
        print(f"[{n}] {ex['id']} ({len(nlls['none'])} tokens): {summary}", flush=True)


if __name__ == "__main__":
    main()

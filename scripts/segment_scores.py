"""Recompute transcript loss restricted to parts of the transcript (CPU only).

    python scripts/segment_scores.py DATASET

Re-tokenizes each scored transcript, labels every token, and sums the saved per-token NLLs
(results/<dataset>/token_nll) over each segment. T still read the whole transcript as context;
only the tokens counted in the loss change. Writes results/<dataset>/scores_<segment>.jsonl in
scores.jsonl's format, so analyze.py works on each:
  content      everything except chat-template tokens
  template     chat-template tokens only
  assistant    U's own tokens: visible text and tool calls (template tokens excluded)
  tool_output  tool outputs (template tokens excluded)

Template tokens are the turn markers the chat template adds around messages:
<|im_start|>, the role word and newline after it, <|im_end|> and the newline after it, and
<tool_call>/</tool_call>/<tool_response>/</tool_response>. Their loss mostly reflects whether
T expects a tool-using agent at all (the honest GLM prompt never mentions tools), not the side
task.
"""

import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from segments import render, token_messages  # noqa: E402

CONDITIONS = ["none", "random", "verbatim"]
SPECIAL = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


def template_mask(ids, tok):
    special = {tok.convert_tokens_to_ids(s) for s in SPECIAL}
    im_start, im_end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
    newline = tok("\n", add_special_tokens=False).input_ids
    mask = np.zeros(len(ids), dtype=bool)
    for i, t in enumerate(ids):
        if t in special:
            mask[i] = True
        elif i >= 1 and ids[i - 1] == im_start:  # role word
            mask[i] = True
        elif i >= 2 and ids[i - 2] == im_start and [t] == newline:  # newline after role
            mask[i] = True
        elif i >= 1 and ids[i - 1] == im_end and [t] == newline:
            mask[i] = True
    return mask


def main():
    dataset = sys.argv[1]
    res = Path("results") / dataset
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    examples = {e["id"]: e for e in map(json.loads, open(f"data/{dataset}.jsonl"))}
    rows = [json.loads(line) for line in open(res / "scores.jsonl")]
    meta = {r["id"]: r for r in rows if r["condition"] == "verbatim"}

    segments = ["content", "template", "assistant", "tool_output"]
    outs = {g: open(res / f"scores_{g}.jsonl", "w") for g in segments}
    shares = {g: [] for g in segments}
    for tid, r in meta.items():
        ex = examples[tid]
        head, full = render(tok, ex, r.get("tools_in_prompt"))
        ids, where = token_messages(tok, head, full, ex["messages"])
        nll = np.load(res / "token_nll" / f"{tid}.npz")
        assert len(ids) == len(nll["verbatim"]), tid
        template = template_mask(ids, tok)
        role = np.array([ex["messages"][k]["role"] for k, _, _ in where])
        masks = {
            "content": ~template,
            "template": template,
            "assistant": ~template & (role == "assistant"),
            "tool_output": ~template & (role == "tool"),
        }
        for c in CONDITIONS:
            v = nll[c].astype(np.float64)
            for g, sel in masks.items():
                outs[g].write(json.dumps({**{k: r[k] for k in ("id", "environment", "side_task")}, "condition": c,
                                          "transcript_tokens": int(sel.sum()), "nll_sum": float(v[sel].sum()),
                                          "nll_mean": float(v[sel].mean()) if sel.any() else 0.0}) + "\n")
        for g, sel in masks.items():
            shares[g].append(sel.mean())
    print(f"{len(meta)} trajectories; mean share of transcript tokens: " + ", ".join(f"{g} {np.mean(v):.1%}" for g, v in shares.items()))


if __name__ == "__main__":
    main()

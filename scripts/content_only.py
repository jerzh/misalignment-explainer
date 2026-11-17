"""Recompute transcript loss with chat-template tokens excluded (CPU only).

    python scripts/content_only.py DATASET

Template tokens are the turn markers the chat template adds around messages:
<|im_start|>, the role word and newline after it, <|im_end|> and the newline after it, and
<tool_call>/</tool_call>/<tool_response>/</tool_response>. Their loss mostly reflects whether
T expects a tool-using agent at all (the honest GLM prompt never mentions tools), not the side
task. Writes results/<dataset>/scores_content.jsonl in the same format as scores.jsonl, plus
the template-only share in scores_template.jsonl, so analyze.py works on either.
"""

import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

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

    out_c, out_t = open(res / "scores_content.jsonl", "w"), open(res / "scores_template.jsonl", "w")
    shares = []
    for tid, r in meta.items():
        ex = examples[tid]
        prefix = ex["prompts"]["verbatim"]
        kw = {"tools": ex["tools"]} if r.get("tools_in_prompt") else {}  # match how score.py rendered it
        head = tok.apply_chat_template(prefix, tokenize=False, **kw)
        full = tok.apply_chat_template(prefix + ex["messages"], tokenize=False, **kw)
        ids = tok(full, add_special_tokens=False).input_ids[len(tok(head, add_special_tokens=False).input_ids):]
        nll = np.load(res / "token_nll" / f"{tid}.npz")
        assert len(ids) == len(nll["verbatim"]), tid
        mask = template_mask(ids, tok)
        for c in CONDITIONS:
            v = nll[c].astype(np.float64)
            for out, sel in ((out_c, ~mask), (out_t, mask)):
                out.write(json.dumps({**{k: r[k] for k in ("id", "environment", "side_task")}, "condition": c,
                                      "transcript_tokens": int(sel.sum()), "nll_sum": float(v[sel].sum()),
                                      "nll_mean": float(v[sel].mean())}) + "\n")
        shares.append(mask.mean())
    print(f"{len(meta)} trajectories; template tokens are {np.mean(shares):.1%} of transcript tokens on average")


if __name__ == "__main__":
    main()

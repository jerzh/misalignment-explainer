"""Align one trajectory's per-token NLLs with its text, per message.

    python scripts/token_attribution.py DATASET TRAJECTORY_ID [OUT.json]

Re-renders the transcript with Qwen3's chat template (tokenizer only, no GPU), checks the
token count matches results/<dataset>/token_nll/<id>.npz, and writes every transcript token
with its message index and NLL under each condition. Message boundaries come from the turn
markers in the full render (<|im_start|>role, <tool_response>), matched to messages in order;
the script asserts each span contains its message's text. (Qwen3's template is not incremental,
so rendering message prefixes would misplace boundaries.)
"""

import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from segments import render, token_messages  # noqa: E402

CONDITIONS = ["none", "random", "verbatim"]


def main():
    dataset, tid = sys.argv[1], sys.argv[2]
    out = Path(sys.argv[3] if len(sys.argv) > 3 else f"results/{dataset}/attribution_{tid[:8]}.json")
    ex = next(e for e in map(json.loads, open(f"data/{dataset}.jsonl")) if e["id"] == tid)
    nll = np.load(f"results/{dataset}/token_nll/{tid}.npz")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

    messages = ex["messages"]
    row = next(r for r in map(json.loads, open(f"results/{dataset}/scores.jsonl")) if r["id"] == tid)
    head, full = render(tok, ex, row.get("tools_in_prompt"))
    ids, where = token_messages(tok, head, full, messages)
    n = len(ids)
    assert n == len(nll["verbatim"]), (n, len(nll["verbatim"]))
    tokens = [{"text": full[s:e], "msg": k, **{c: float(nll[c][i]) for c in CONDITIONS}} for i, (k, s, e) in enumerate(where)]

    msgs = []
    for k, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            kind = "call: " + ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
            if m["content"].strip():
                kind = "text + " + kind
        elif m["role"] == "assistant":
            kind = "text"
        elif m["role"] == "tool":
            kind = f"output: {m.get('name')}"
        else:
            kind = m["role"]
        msgs.append({"role": m["role"], "kind": kind})

    out.write_text(json.dumps({
        "dataset": dataset, "id": tid, "environment": ex["environment"], "main_task": ex["main_task"],
        "side_task": ex["side_task"], "side_block": ex["side_block"].strip(), "messages": msgs, "tokens": tokens,
    }))
    totals = {c: sum(t[c] for t in tokens) for c in CONDITIONS}
    print(f"{n} tokens aligned -> {out}; total NLL " + ", ".join(f"{c}={v:.1f}" for c, v in totals.items()))


if __name__ == "__main__":
    main()

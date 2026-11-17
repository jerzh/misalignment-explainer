"""Map transcript tokens to the messages they came from (shared by the CPU analyses)."""

import numpy as np


def render(tok, ex, tools_in_prompt):
    """Full render and transcript-start offset, matching how score.py rendered the verbatim condition."""
    prefix, messages = ex["prompts"]["verbatim"], ex["messages"]
    kw = {"tools": ex["tools"]} if tools_in_prompt else {}
    head = tok.apply_chat_template(prefix, tokenize=False, **kw)
    full = tok.apply_chat_template(prefix + messages, tokenize=False, **kw)
    assert full.startswith(head)
    return head, full


def message_bounds(head, full, messages):
    """Char offsets where each message starts in the render (plus the end), checked against content.

    Walks the render matching each message's opening marker in order: assistant/user turns open
    with <|im_start|>role, tool outputs with <tool_response>; the first tool output after an
    assistant turn also owns the <|im_start|>user header. Qwen3's template is not incremental,
    so rendering message prefixes would misplace boundaries.
    """
    starts, pos = [], len(head)
    for m in messages:
        marker = "<tool_response>" if m["role"] == "tool" else f"<|im_start|>{m['role']}"
        idx = full.index(marker, pos)
        pos = idx + len(marker)  # continue after this message's own marker
        if m["role"] == "tool" and starts and messages[len(starts) - 1]["role"] != "tool":
            idx = full.rindex("<|im_start|>user", 0, idx)
        starts.append(idx)
    starts[0] = len(head)
    bounds = starts + [len(full)]  # message k spans chars [bounds[k], bounds[k+1])
    for m, a, b in zip(messages, bounds, bounds[1:]):
        assert m["content"].strip()[:200] in full[a:b], m["role"]
    return bounds


def token_messages(tok, head, full, messages):
    """For each transcript token: (message index, char start, char end)."""
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    start = len(tok(head, add_special_tokens=False).input_ids)
    bounds = message_bounds(head, full, messages)
    out, k = [], 0
    for i in range(start, len(enc.input_ids)):
        s, e = enc.offset_mapping[i]
        while s >= bounds[k + 1]:
            k += 1
        out.append((k, s, e))
    return enc.input_ids[start:], out

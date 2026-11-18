"""Detect literals an instruction copies from a transcript (gepa.md literal-filter ablation).

    python scripts/literals.py [results/gepa_pilot2] [data/deepseek_v4_pro_clean.jsonl]   # check pilot instructions

A *literal* is a span of the instruction that appears in the transcript (U's actions or tool outputs)
but not in U's base prompt (honest system prompt, main task, tool definitions), which E may see:
  1. identifier-like tokens: containing / . _ : @ ~ or a digit, camelCase, or >= 20 characters
  2. spans in backticks or quotes
  3. any substring of >= LONG_SUBSTRING characters shared with the transcript
Matching is case-insensitive. Plain English words are never literals.
"""

import json
import re
import sys
from pathlib import Path

LONG_SUBSTRING = 30
MIN_TOKEN = 3
WHITELIST = {"e.g.", "i.e.", "etc.", "vs.", "e.g", "i.e"}
TOKEN = re.compile(r"[A-Za-z0-9_./~:@$\-+=#%]+")
QUOTED = re.compile(r"`([^`]+)`|\"([^\"]+)\"|'([^']{3,})'")


def corpus(ex):
    """(transcript text, base prompt text), both lowercased."""
    parts = []
    for m in ex["messages"]:
        parts.append(m["content"])
        for tc in m.get("tool_calls") or []:
            parts.append(json.dumps(tc["function"]["arguments"], ensure_ascii=False))
            parts += [str(v) for v in (tc["function"]["arguments"] or {}).values()] if isinstance(tc["function"]["arguments"], dict) else []
    base = [m["content"] for m in ex["prompts"]["none"]] + [json.dumps(ex.get("tools") or [])]
    return "\n".join(parts).lower(), "\n".join(base).lower()


def identifier_like(tok):
    return (any(c in tok for c in "/._:@~") or any(c.isdigit() for c in tok)
            or re.search(r"[a-z][A-Z]", tok) is not None or len(tok) >= 20)


def find_literals(instruction, transcript, base):
    """Literal spans in the instruction (deduplicated, in order)."""
    found = []
    spans = [m.group(0).strip(".,;:!?)(") for m in TOKEN.finditer(instruction)]
    spans = [s for s in spans if len(s) >= MIN_TOKEN and identifier_like(s)]
    spans += [next(g for g in m.groups() if g) for m in QUOTED.finditer(instruction)]
    for s in spans:
        low = s.lower()
        if low in WHITELIST or len(low) < MIN_TOKEN:
            continue
        if low in transcript and low not in base and s not in found:
            found.append(s)
    text = instruction.lower()  # long shared substrings, not already covered
    i = 0
    while i <= len(text) - LONG_SUBSTRING:
        piece = text[i:i + LONG_SUBSTRING]
        if piece in transcript and piece not in base:
            j = i + LONG_SUBSTRING
            while j < len(text) and text[i:j + 1] in transcript:
                j += 1
            chunk = instruction[i:j].strip()
            if not any(chunk.lower() in f.lower() or f.lower() in chunk.lower() for f in found):
                found.append(chunk)
            i = j
        else:
            i += 1
    return found


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "results/gepa_pilot2")
    data = sys.argv[2] if len(sys.argv) > 2 else "data/deepseek_v4_pro_clean.jsonl"
    examples = {e["id"]: e for e in map(json.loads, open(data))}
    S = json.loads((res / "summary.json").read_text())
    for t in S["trajectories"]:
        ex = examples[t["id"]]
        transcript, base = corpus(ex)
        print(f"\n### {t['id'][:8]} {ex['side_task']}")
        rows = [("gist", ex["gist"]), ("verbatim_block", ex["side_block"])]
        rows += [(f"zero_shot[{k}]", z["instruction"]) for k, z in enumerate(t["zero_shot"])]
        rows += [("gepa_best", t["gepa_best"]["instruction"])]
        for name, text in rows:
            lits = find_literals(text, transcript, base)
            print(f"  {name:15s} {'PASS' if not lits else 'REJECT'}  {lits}")


if __name__ == "__main__":
    main()

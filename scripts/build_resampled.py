"""Build the resampled dataset (resampled.md, step 2) from several board runs of the same U setup.

    python scripts/build_resampled.py RAW_RUN_DIR [RAW_RUN_DIR ...] --out data/flash_resampled_clean.jsonl

Like `build_dataset.py --format board --clean`, except that side-task FAILURES are kept too (flag
`success`), because a failed sample still counts if U attempted the side task (decided 2026-11-19; the
attempt check is scripts/attempt_judge.py). Errored samples without a transcript are dropped.
Adds: `pair` (environment|main_task|side_task), `run`, `monitor2` (gpt-5.4-nano trajectory monitor, 0-100),
`source` (board run family) and `scaffold` (plain | claude_code | codex).
Reports how many harness messages were removed.

Scaffold runs (board-protocols claude-code-bare / codex-bare, added 2026-11-19 with the user's approval): U runs inside
Claude Code or Codex CLI. Their monitored ("honest") view has several system messages (the scaffold's own system prompt,
Claude Code's agent-types note, Codex's skills list; Claude Code's billing-header line is dropped) and one or two user
messages; U's real ("hidden") view replaces the system prompt(s) with the attack prompt and keeps the user messages.
convert_scaffold() builds:
  - prompts.none: ONE system message = the honest view's system messages joined in order (Qwen3's template takes a
    single leading system message; Claude Code's agent-types note, which comes after the user message, moves into it),
    then the honest view's user messages (checked equal to U's);
  - prompts.verbatim: the attack system prompt + the same user messages;
  - tools: U's own tool schemas from its last model call (attachment references resolved), minus attack-only tools;
  - messages: U's transcript with reasoning dropped, including Claude Code's inline <think>...</think> blocks;
    Claude Code's "(no content)" placeholder for an empty text block is emptied.
Samples whose last model call no longer starts with the original user message (Claude Code compacted its context; 7 of
60) are skipped, since that call's input lacks the start of the transcript.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import re  # noqa: E402

from build_dataset import ATTACK_ONLY_TOOLS, SIDE_RE, clean, convert_board, first, remove_harness, resolve_all, side_success, text_of  # noqa: E402

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)
SOURCES = {"6094b8cd": "models", "4aa9acab": "null_blue_team", "ca29e761": "claude_code", "3911aaeb": "codex"}


def convert_scaffold(sample):
    att, pool = sample["attachments"], sample["events_data"]["messages"]
    calls = [e for e in sample["events"] if e.get("event") == "model" and e.get("role") == "hidden_untrusted"]
    honest = [e for e in sample["events"] if e.get("event") == "model" and e.get("role") == "untrusted"]
    if not calls or not honest:
        return None
    c = calls[-1]
    convo = [m for a, b in c["input_refs"] for m in pool[a:b]] + [c["output"]["choices"][0]["message"]]
    hp = [m for a, b in honest[0]["input_refs"] for m in pool[a:b]]
    assert convo[0]["role"] == "system"
    k = next(i for i, m in enumerate(convo) if m["role"] == "assistant")
    u_users = [text_of(m["content"], att) for m in convo[1:k]]
    assert all(m["role"] == "user" for m in convo[1:k])
    h_users = [text_of(m["content"], att) for m in hp if m["role"] == "user"]
    if h_users != u_users:  # Claude Code compacted the context: the last call no longer holds the whole transcript
        return None
    systems = [text_of(m["content"], att) for m in hp if m["role"] == "system"]
    systems = [t for t in systems if not t.startswith("x-anthropic-billing-header")]
    users = [{"role": "user", "content": t} for t in u_users]
    system = text_of(convo[0]["content"], att)
    messages = clean(convo[k:], att)
    for m in messages:
        if m["role"] == "assistant":
            m["content"] = THINK_RE.sub("", m["content"]).strip()
            if m["content"] in ("(no content)", "(no content"):  # Claude Code SDK's placeholder for an empty text block
                m["content"] = ""
    tools = [{"type": "function", "function": {"name": t["name"], "description": resolve_all(t.get("description", ""), att),
                                               "parameters": resolve_all(t.get("parameters", {}), att)}}
             for t in c["tools"] if t["name"] not in ATTACK_ONLY_TOOLS]
    return {"prompts": {"none": [{"role": "system", "content": "\n\n".join(s.strip() for s in systems)}] + users,
                        "verbatim": [{"role": "system", "content": system}] + users},
            "side_block": SIDE_RE.search(system).group(1), "messages": messages, "tools": tools}


def convert(sample):
    """Plain board format if the monitored view is [system, user], else a scaffold run."""
    honest = [e for e in sample["events"] if e.get("event") == "model" and e.get("role") == "untrusted"]
    if honest:
        pool = sample["events_data"]["messages"]
        roles = [m["role"] for a, b in honest[0]["input_refs"] for m in pool[a:b]]
        if roles != ["system", "user"]:
            return convert_scaffold(sample), "claude_code" if roles.count("system") > 2 else "codex"
    return convert_board(sample), "plain"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("raw_dirs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    kept = dropped = removed = affected = 0
    with open(a.out, "w") as f:
        for d in a.raw_dirs:
            for path in sorted(Path(d).glob("*.json")):
                sample = json.loads(path.read_text())["sample"]
                meta = sample["metadata"]
                side = first(meta.get("side_task") or meta.get("side_tasks"))
                ex, scaffold = convert(sample) if side else (None, None)
                if ex is None:
                    dropped += 1
                    continue
                env = first(meta["environment"])["id"]
                main_task = first(meta.get("main_task") or meta.get("main_tasks"))["id"]
                m2 = (sample["scores"].get("monitor2") or {}).get("value") or {}
                ex = remove_harness({"id": path.stem, "run": Path(d).name[:8], "source": SOURCES.get(Path(d).name[:8], Path(d).name[:8]),
                                     "scaffold": scaffold, "environment": env, "main_task": main_task,
                                     "side_task": side["id"], "pair": f"{env}|{main_task}|{side['id']}",
                                     "success": side_success(sample, side["id"]), "monitor2": next(iter(m2.values()), None), **ex})
                removed += ex["harness_messages_removed"]
                affected += ex["harness_messages_removed"] > 0
                f.write(json.dumps(ex) + "\n")
                kept += 1
    print(f"kept {kept}, dropped {dropped}; harness messages removed: {removed} in {affected} samples -> {a.out}")


if __name__ == "__main__":
    main()

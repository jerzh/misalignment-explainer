"""Upload the scored transcripts to a Docent collection for reading.

    uv run --no-project --with 'docent-python>=0.1.83' python scripts/ingest_docent.py

Each trajectory becomes one AgentRun: U's verbatim prompt followed by the cleaned transcript
(reasoning removed), i.e. exactly the text T scores in the `verbatim` condition. Uses the API
key from ~/.docent/docent.env.
"""

import json
from pathlib import Path

from docent import Docent
from docent.data_models import AgentRun, Transcript
from docent.data_models.chat import (
    AssistantMessage,
    ToolCall,
    ToolMessage,
    check_agent_runs,
    format_check_report,
    parse_chat_message,
)

COLLECTION_NAME = "misalignment-explainer: LinuxArena attack transcripts"
DATASETS = {
    "glm53": ("GLM-5.3", "6b64412bf12644e0980961cd971ec05c"),
    "deepseek_v4_pro": ("DeepSeek V4 Pro", "7b175ceab7ff4202ae6bd95245d13590"),
}


def convert_messages(messages):
    out, pending = [], []  # pending: ids of tool calls awaiting a response, in order
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            calls = [
                ToolCall(id=f"call_{i}_{j}", function=tc["function"]["name"], arguments=tc["function"]["arguments"], type="function")
                for j, tc in enumerate(m.get("tool_calls") or [])
            ]
            if not calls and not m["content"].strip():
                continue
            pending = [(c.id, c.function) for c in calls]
            out.append(AssistantMessage(content=m["content"], tool_calls=calls or None))
        elif m["role"] == "tool":
            call_id, function = pending.pop(0) if pending else (None, m.get("name"))
            out.append(ToolMessage(content=m["content"], tool_call_id=call_id, function=function))
        else:
            out.append(parse_chat_message({"role": m["role"], "content": m["content"]}))
    return out


def convert(ex, dataset):
    model, run_id = DATASETS[dataset]
    messages = [parse_chat_message(m) for m in ex["prompts"]["verbatim"]] + convert_messages(ex["messages"])
    return AgentRun(
        transcripts=[Transcript(messages=messages)],
        metadata={
            "dataset": dataset,
            "u_model": model,
            "trajectory_id": ex["id"],
            "linuxarena_url": f"https://data.linuxarena.ai/trajectories/{ex['id']}",
            "linuxarena_run_id": run_id,
            "environment": ex["environment"],
            "main_task": ex["main_task"],
            "side_task": ex["side_task"],
            "side_task_description": ex["side_block"].strip(),
            "honest_prompt": "\n\n".join(m["content"] for m in ex["prompts"]["none"]),
        },
    )


if __name__ == "__main__":
    runs = []
    for dataset in DATASETS:
        examples = [json.loads(line) for line in open(Path("data") / f"{dataset}.jsonl")]
        runs += [convert(ex, dataset) for ex in examples]
        print(f"{dataset}: {len(examples)} trajectories")
    print(format_check_report(check_agent_runs(runs)))
    client = Docent()
    collection_id = client.create_collection(
        name=COLLECTION_NAME,
        description="U's verbatim prompt + transcript with reasoning removed, as scored by T. See the misalignment-explainer README.",
    )
    client.add_agent_runs(collection_id, runs)
    print(f"uploaded {len(runs)} runs to collection {collection_id}")

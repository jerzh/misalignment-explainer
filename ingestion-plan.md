# Docent Ingestion Plan

## Configuration
- Data path: data/glm53.jsonl, data/deepseek_v4_pro.jsonl (built by scripts/build_dataset.py)
- API key source: ~/.docent/docent.env

## Source Analysis
- One JSON line per trajectory: prompts (none/verbatim), side_block, cleaned transcript (reasoning removed).
- Expected records: 203 (glm53) + 49 (deepseek_v4_pro) = 252

## Proposed Docent Structure
- Collection: "misalignment-explainer: LinuxArena attack transcripts" (both datasets; filter on metadata.dataset)
- AgentRun unit: one trajectory; one Transcript, no TranscriptGroups

## Field Mapping
| Source | Docent target | Notes |
| --- | --- | --- |
| prompts.verbatim + messages | Transcript.messages | Exactly the text T scores under `verbatim` |
| tool_calls | ToolCall, ids `call_<msg>_<n>` | Source has no ids; tool responses matched in order |
| id, environment, main_task, side_task | metadata | plus dataset, u_model, LinuxArena URL/run id |
| side_block | metadata.side_task_description | |
| prompts.none | metadata.honest_prompt | |

## Omitted Data
| Field/File | Reason | Impact |
| --- | --- | --- |
| Reasoning blocks | Already dropped by build_dataset.py (T never sees them) | Can't read U's reasoning here; it is in data/raw |
| Empty assistant messages (no text, no tool calls) | Noise after reasoning removal | None |
| GLM trajectories > 32k tokens | Kept (all 203), though only 179 are scored | Unscored ones are included for reading |

## Confirmation
- Approval mode and source: auto_approve_plans = True (account preference)

## Verification
- Accepted warnings: 474 empty_message (tool calls with empty output: silent bash, submit); 38 missing_tool_response (36 final `submit` calls, 2 runs ending mid-call). Faithful to source.

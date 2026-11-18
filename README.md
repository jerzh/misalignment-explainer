# misalignment-explainer

This project is part of a longer-term project whose ultimate goal is to train some "Explainer" model E which produces an implied "(misaligned) instruction" given a base prompt and trajectory. The point is to have visibility into model goals/alignment in the absence of chain-of-thought.

Specifically, suppose we have a misaligned model U. We run U(base prompt) -> trajectory. Now, we train some model E with E(base prompt) -> instruction such that for some reference model(s) T, T(base prompt + instruction + trajectory) has low loss. The idea is to see if the recovered instruction has some natural-language explanation of the misaligned goals of U.

To test whether this idea might work, we give U a secret "side task prompt" after the base prompt which is shown only to U, and we see if "instruction" can recover the gist of this prompt.

In this project, we'll test this framework on agentic coding transcripts. First, we'd like to answer:
1. Does "instruction" similar to "side task prompt" actually result in lower reference model loss compared to empty/random instruction?
2. If we use some basic algorithms to optimize "instruction" for low reference model loss, does it actually end up similar to "side task prompt"? (We don't have time for RL in this 3-day project, so we use simpler methods, like trial-and-error or soft prompting.)

TODO: (Deepseek trajectories only)
1. Sanity check how the trajectories were generated. Are they from the original LinuxArena dataset? Are the main/side task a diverse enough sample pool? Are they unmonitored? Any other quirks of how they were set up?
2. Our ultimate goal with this "explainer" method is to understand if U is doing anything malicious, beyond what other monitoring methods can already see. Let's run a black-box monitor pass over all trajectories (or see if a monitor pass already exists in the dataset), subset to only the ones the monitor scores low ("hard to identify"), and see if the verbatim/random distinction remains.
3. GEPA part 1: Just try to get the loss as low as possible. Run an LLM judge to check if the instruction is similar to the side task prompt, to understand the extent of recovery / whether there are some unrelated or "jailbreak" instructions that work.
4. GEPA part 2: Red-team the protocol by trying to make the instruction as benign as possible (judged by same black-box monitor) while still having low loss. Let's maintain a Pareto frontier of instructions for the two objectives (loss, benign-ness).
5. Consider using better reference model T (if no longer works)
6. Per-step advice???

## Setup

### Trajectories (U)

Attack runs from [LinuxArena](https://data.linuxarena.ai/runs): U is given a main coding task in a live Linux environment plus a secret side task (e.g. exfiltrate credentials, open a reverse shell), and tries to do both. We keep only trajectories whose side task succeeded.

| Dataset | U | LinuxArena run | Trajectories kept | Notes |
|---|---|---|---|---|
| `glm53` | GLM-5.3 | `mentees-glm53-attack-300` (`6b64412b…`), older `simple-attack` policy | 203 (179 fit in 32k tokens) | Control: U is told the side-task test tool is unmonitored and openly narrates the attack ("Now let me verify the side task:") in 154/203 transcripts. |
| `deepseek_v4_pro` | DeepSeek V4 Pro | `board-models-…-deepseek-v4-pro-0813-linuxarena` (`7b175cea…`), current `simple-attack` policy | 49 | Main dataset: U is told to act "without being noticed", has no unmonitored tools, and its visible messages never name the side task. ~61k tokens median. |

The four DeepSeek board runs on LinuxArena are the same 60 trajectories re-scored by different monitors, so only one is used.

### Prompts and conditions

The transcript (everything after the prompt) is fixed; only the prompt before it changes:

- **`none`**: honest prompt, no side task.
  - `glm53`: U's system prompt with the attack parts removed (honest opening, environment, main task).
  - `deepseek_v4_pro`: the honest system prompt from the run's monitored view, plus the main task as a user message.
- **`verbatim`**: exactly what U saw (the attack prompt, including the `<side_task_description>` block). The "side task prompt" is the difference between this and `none`.
- **`random`**: `verbatim` with the side task block replaced by another trajectory's side task from the same environment (seeded).

Transcript processing: reasoning / thinking is dropped; visible assistant text, tool calls (including `run_side_task_test` and `submit` in `glm53`), and tool outputs are kept.

### Reference model (T) and loss

- T = Qwen3-32B (bf16), rendered with its own chat template (tool calls as `<tool_call>`, tool outputs as `<tool_response>`), no tool schemas.
- Loss = T's negative log-likelihood of every transcript token, tool outputs included. Since transcript tokens are identical across conditions, losses compare directly per trajectory.
- `glm53` runs at native 32k context (longer trajectories skipped); `deepseek_v4_pro` uses YaRN RoPE scaling ×4 (128k).

## Usage

```sh
# Download a run's raw samples and build examples
python scripts/download.py 6b64412bf12644e0980961cd971ec05c data/raw/glm53_simple_attack
python scripts/build_dataset.py data/raw/glm53_simple_attack data/glm53.jsonl --format simple
python scripts/download.py 7b175ceab7ff4202ae6bd95245d13590 data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590
python scripts/build_dataset.py data/raw/deepseek_v4_pro/7b175ceab7ff4202ae6bd95245d13590 data/deepseek_v4_pro.jsonl --format board

# Score (on a GPU) and compare conditions
python scripts/score.py --examples data/glm53.jsonl --out results/glm53
python scripts/score.py --examples data/deepseek_v4_pro.jsonl --out results/deepseek_v4_pro --yarn
python scripts/analyze.py results/glm53/scores.jsonl
```

`score.py` writes one row per (trajectory, condition) to `results/<dataset>/scores.jsonl` and per-token NLLs to `results/<dataset>/token_nll/<id>.npz`; re-running skips scored trajectories. `analyze.py` reports paired differences (`none − verbatim`, `random − verbatim`, `none − random`) in total and per-token NLL with bootstrap 95% CIs.

### Running on Flamingo (`h100-prime-001`)

The repo lives at `~/misalignment-explainer` on the `vast-jerzh` NFS volume; jobs use the `flamingo-devbox` image, 2 H100s, and the shared `nfs-huggingface-home` model cache.

```sh
# One-time: CPU pod for syncing code/data and building the venv (Python installed on NFS)
kubectl apply -f k8s/sync-pod.yaml
tar czf - --exclude=data/raw --exclude=.git --exclude=results --exclude=.venv . \
  | kubectl exec -i jerzh-sync -- tar xzf - -C /nfs/users/jerzh/misalignment-explainer
kubectl exec jerzh-sync -- bash -lc 'cd ~/misalignment-explainer && UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT= uv sync --python 3.12 --managed-python'

# Launch scoring jobs
sed 's|__DATASET__|glm53|g; s|__ARGS__||' k8s/score.yaml | fl run -f -
sed 's|__DATASET__|deepseek_v4_pro|g; s|__ARGS__|--yarn|' k8s/score.yaml | fl run -f -
```

## Layout

- `scripts/download.py`: fetch a LinuxArena run's raw Inspect samples
- `scripts/build_dataset.py`: raw samples → `data/<dataset>.jsonl` (prompts per condition, side task block, cleaned transcript)
- `scripts/score.py`: per-condition transcript NLL under T
- `scripts/analyze.py`: paired comparison across conditions
- `k8s/`: Flamingo job and sync-pod manifests

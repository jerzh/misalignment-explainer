# Resampled dataset: repeat the analysis with many samples of the same prompt (plan, 2026-11-19)

**Why.** On DeepSeek each main + side task pair has one trajectory, so GEPA's best instructions can win by summarizing that one transcript's steps and copying its literals (`gepa.md`, Content constraints). With many resamples of the *same* prompt (same main task and side task), one instruction has to explain all of them. Trajectory-specific detail should stop paying off, while the side task, the one thing every resample shares, should still pay off. The question we care about is whether the lowest-loss instruction makes the malice obvious. Overfitting to the resamples is fine; there is no train/test split (decided 2026-11-19).

Note on GLM: `glm53` repeats side tasks (85 side tasks, up to 6 trajectories each) but never a main + side pair (203 pairs, 1 each). So it is not a resampled dataset in this sense.

Steps 1–3 can run without asking. Step 4 needs the design decisions below, and step 5 runs only if step 4's results meet its condition.

## 1. Find the dataset and check it is usable

Look on LinuxArena (data.linuxarena.ai/runs) for a run that samples each main + side task pair many times. Then run the same checks as for DeepSeek (`todo.md`, "Generation sanity check and monitor subset"):
- **Resampling:** how many pairs, how many samples per pair (want ≥ 8–10 per pair, ideally several pairs across environments), and whether samples differ only by sampling seed (same prompt, model, policy and settings).
- **Generation:** which U and policy. Want a current `simple-attack`-style policy where U is told to stay unnoticed and doesn't narrate the attack. GLM narrates and is only a positive control. Also check whether U ran monitored or unmonitored, and the honest/attack mix.
- **Side-task success:** keep only successful samples. Check how success varies by pair, since keeping successes biases things (on DeepSeek, cover-task pairings always succeeded).
- **Black-box monitor:** there must be a trajectory monitor with recorded scores (and its prompt, so the instruction monitor can reuse it) that is **fooled at least some of the time**. Requirement: a meaningful share of successful samples score low (e.g. ≤ 25 on 0–100), so the data isn't just obvious attacks. Report the score distribution per pair; on DeepSeek, 17 of 49 scored ≤ 25.
- **Leaks in visible text:** count samples whose visible messages name or narrate the side task (DeepSeek: 0 after cleaning; GLM: most). Too many means the data is positive-control only.
- **Length:** fits Qwen3-32B with YaRN ×4 (128k) after cleaning.
- **Provenance:** run id, date, creator, task set. Note whether it's from the LinuxArena paper or someone else's run.

## 2. Preprocessing

As for `data/<ds>_clean.jsonl` (`build_dataset.py --clean`):
- **Remove side-task harness calls and their replies,** e.g. `run_side_task_test`, `set_side_task_submission_args` and whatever this run's policy has.
- **Remove those tools from the schemas.**
- **Remove harness paragraphs** from the side-task block.
- **Drop reasoning/thinking.**

Report how many messages and samples are affected.

## 3. Baselines

Per sample, scored on U's tokens with tool schemas in the prompt, using the instruction slot on the honest prompt (as in the instruction-conditions run):
- **none**
- **verbatim block**
- **gist**
- **random gist:** another side task's gist from the same environment, seeded.

Report win rates plus Wilcoxon (verbatim − random gist, gist − random gist) and relative gain. Also report per pair (spread across resamples) and for the monitor ≤ 25 subset.

## 4. GEPA Part 1a (lowest loss, ≤ 300 characters, no content constraints)

**Hypothesis:** with one instruction per pair scored over many resamples, the best instruction won't be a summary of one trajectory's steps, because a summary doesn't transfer across resamples. Main question: is the malice obvious in the lowest-loss instruction (instruction monitor, hand check)?

Proposed design, for review before running:
- **Unit of optimization:** one instruction per main + side task pair, scored on the sum of its gain over all the pair's resamples. No train/test split; overfitting is fine.
- **Pareto instances:** the resamples, or their chunks, instead of one transcript's 8 chunks. Feedback batches mix resamples.
- **Proposer:** Opus 4.8 (OpenRouter, effort medium), as in `gepa_part1.py`. It sees the coding task and the action lists of a few resamples; showing all of them may not fit, so this needs a decision.
- **Seeds:** none, vague, 3 zero-shot guesses (from the resamples' actions).
- **References:** gist, verbatim block, random gist.
- **Judging the finals:**
  - instruction monitor suspicion (is the malice obvious?) and a hand check;
  - robust gain;
  - side-task identification;
  - category label: side task / step summary / environment or style / unrelated / jailbreak;
  - comparison with a per-trajectory GEPA run on a few single resamples, to show how much single-trajectory GEPA overfits.
- **Cost:** one T pass per candidate per resample, so T compute grows with the number of resamples. GEPA proposals stay ~$4–8 per 5 pairs. Plan GPUs (one 4-GPU pod scores ~1 pass/s).

## 5. GEPA Part 2 (benign-looking instructions), only if step 4 is interesting

**Condition:** step 4's best instructions mostly state a goal rather than a step summary (by category label and hand check).

If so, run Part 2 as in `gepa.md`:
- instruction monitor = the dataset's monitor prompt, edited as for monitor v1 (Qwen3-32B);
- 2D Pareto on (gain over all resamples, suspicion);
- optionally with the one-goal constraint.

Report gain and suspicion of the zero-suspicion frontier, and hand-check whether it still carries the side task. Lessons from the DeepSeek runs to carry over:
- compare monitors at a matched false-positive rate;
- the leak check needs validating before it is used (it misjudged the benign-twin candidates);
- a monitor-gaming result needs a side-task-aware check of whether the instruction still carries the side task.

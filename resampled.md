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

# Results / Log

## Step 1: dataset search (2026-11-19, autonomous session). Result: no suitable dataset, stopped here

**Verdict: LinuxArena has no run that resamples main + side task pairs with a fixed prompt, so steps 2–5 were not run.** No near-miss meets the hard requirement either: a recorded black-box trajectory monitor that is fooled a meaningful share of the time. Spend: $0 OpenRouter, no cluster jobs. Scan scripts and tables are in `results/resampled/step1_scan/` (gitignored).

### What was scanned
- All 642 runs listed by `data.linuxarena.ai/api/runs` (as of 2026-11-19). The trajectory listings were pulled for all 615 non-BashArena runs with ≥ 2 trajectories: environment, main task, side task, success, monitors, `time_started`, `actions_count`.
- **Within a run**, the only runs with more than 2 trajectories per main + side pair are the human-written sabotage-trajectory ("straj") runs: `* monitor on straj`, 74 trajectories, 35–36 pairs, up to 9 per pair. Those are different hand-built attacks re-scored by 12 monitors. There is no model U and no single prompt, so they don't count as resamples. Every model run (paper, board, sabotage-monitor, sandboxing) has ≤ 2 trajectories per pair, nearly always 1. No run uses Inspect epochs > 1.
- **Duplicate runs are not resamples.** Each board run (`board-models-*`, `board-protocols-*`) appears 4–5 times on the site, but they hold the same 60 generations (identical `time_started`), re-monitored or not. The same goes for the paper runs and their "GPT-5 Nano monitor on …" / "LinuxArena paper: …" copies (identical action counts and trajectories).
- **Across runs**, I pooled successful generations per pair and per U model. Only two model families repeat a pair ≥ 8 times, and neither does it with a fixed prompt:

| candidate | pairs × successful samples | same prompt? | monitor | fooled (≤ 25)? | other problems |
|---|---|---|---|---|---|
| **Claude Sonnet 4.5, `big_sonnet_attack_*`** (sandboxing-dynamics study, 17 runs, 2026-06-08, `research.ram.260604_sandboxing_dynamics`) | 5 pairs with 11–17 successes over the 17 variants (suppress_warnings 17, delete_blood_tests_and_backups 16, suppress_workflow_audits 16, expose_secret 11, hijack_cross_team_workflows 11). The other 71 pairs have ≤ 7 | No. Each run is a different sandbox/permission protocol (no_sandbox, no_internet, editor_only, per_action, per_task, request_permission_*, …; "noreason/seereason" pairs differ only in the protocol's gate) | **None recorded** (`monitors` and `audit_scores` empty) | n/a | Reconstructed format: the system message is the main-task spec, the user message is the one-line side-task goal, and neither U's real attack prompt nor its tool schemas are stored. Sandbox events (`[SandboxStart …]`, `[JITGate …]`) are interleaved as assistant messages. Haiku's sibling runs (`big_haiku_attack_*`) almost never succeed (62 of 76 pairs 0/17) |
| **Claude Haiku 4.5, LinuxArena paper attack family** (2026-04-12…18): `Haiku attack` + `thinking=off/4k/16k/64k` (+ `simple-insert`, `no-ST-test`) | Same-prompt variants (attack + 4 thinking budgets): ≤ 5 per pair, 18 pairs at 5. With simple-insert and no-ST-test: ≤ 7, 9 pairs at 7 | Nearly: the 5 core variants differ only in the thinking budget; simple-insert and no-ST-test change the attack prompt. Pooling more (Goal-only, Goal-plus-details, No Internet, Editor Only, obvious-attack, sandbox) reaches 65 pairs with ≥ 8, but those change the side-task text or the policy | Per-action GPT-5 Nano (`post_hoc`, max over actions) on every variant. A trajectory-level "quick" monitor (`mean-quick@gpt-5-nano`) exists only on the single default-attack generation | **Barely: 12 of 374 successful samples (3%) have max per-action score ≤ 25**. The 15 best-covered pairs are mostly 75–95 | The paper attack policy has an explicit plan stage and narrates ("I'll start by exploring the codebase to understand … both the main task and side task"), so it would be GLM-style positive-control data |

- **Other U models** have at most 1–5 successful generations per pair across all their runs (Opus 4.6 ≤ 5, Kimi K2.5 ≤ 4, DeepSeek V4 Flash ≤ 4, GLM-5.3 ≤ 3, DeepSeek V4 Pro 1), always from different policies or protocols.

### Why I stopped instead of relaxing the criteria
- The hard requirement (recorded black-box trajectory monitor, fooled a meaningful share of the time) fails for both near-misses. Sonnet has no monitor at all. Haiku's monitor is per-action, catches 97% of successes, and its only trajectory-level monitor covers one generation per pair.
- Neither is a resample in the plan's sense (same prompt, model, policy and settings; only the seed differs). Sonnet varies the protocol and tools across samples. Haiku's same-prompt core gives ≤ 5 samples per pair (the plan wants ≥ 8–10) and varies the thinking budget.
- Sonnet's stored transcripts lack U's real prompt and tool schemas. `verbatim`, the honest base prompt and the tool-schema conditions couldn't be rebuilt the way `build_dataset.py` does for the board runs.

### Options for the user (not started; each needs a decision)
1. **Generate a resampled run ourselves (recommended if the question is still wanted).** Use control-tower on the LinuxArena eval box, with the same policy and monitor as the DeepSeek board run: DeepSeek V4 Pro, current `simple-attack` (`ct/attack`), null blue team, post-hoc `trajectory@gpt-5.4-nano` monitor. Run Inspect epochs ≈ 10–12 on ~5 main + side pairs. Pick pairs where the DeepSeek board trajectory both succeeded and scored ≤ 25 (e.g. suppress_workflow_audits, erase_history, mp_run_malware_live, or Part 1 extension pairs), plus 1–2 high-suspicion controls. The `7b175cea` run cost ~$60 for 105 samples (DeepSeek $38, gpt-5.4-nano monitor $21), so 60 attack samples would cost roughly $35–50 plus EC2 time (~$17.60/day). `build_dataset.py --format board --clean` would then work unchanged.
2. **Use Sonnet's sandboxing runs without a monitor.** 5 pairs × 11–17 successes. Score our own monitor with Qwen3-32B and the DeepSeek `monitor2` prompt, and accept that protocol and tools vary across samples and that the base prompt has to be reconstructed. This needs a new `--format` in the builder.
3. **Use Haiku's same-prompt family as a positive control only** (≤ 5 per pair, narrated, monitor rarely fooled).

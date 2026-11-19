"""Build the resampled dataset (resampled.md, step 2) from several board runs of the same U setup.

    python scripts/build_resampled.py RAW_RUN_DIR [RAW_RUN_DIR ...] --out data/flash_resampled_clean.jsonl

Like `build_dataset.py --format board --clean`, except that side-task FAILURES are kept too (flag
`success`), because a failed sample still counts if U attempted the side task (decided 2026-11-19; the
attempt check is scripts/attempt_judge.py). Errored samples without a transcript are dropped.
Adds: `pair` (environment|main_task|side_task), `run`, `monitor2` (gpt-5.4-nano trajectory monitor, 0-100).
Reports how many harness messages were removed.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import convert_board, first, remove_harness, side_success  # noqa: E402


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
                ex = convert_board(sample) if side else None
                if ex is None:
                    dropped += 1
                    continue
                env = first(meta["environment"])["id"]
                main_task = first(meta.get("main_task") or meta.get("main_tasks"))["id"]
                m2 = (sample["scores"].get("monitor2") or {}).get("value") or {}
                ex = remove_harness({"id": path.stem, "run": Path(d).name[:8], "environment": env, "main_task": main_task,
                                     "side_task": side["id"], "pair": f"{env}|{main_task}|{side['id']}",
                                     "success": side_success(sample, side["id"]), "monitor2": next(iter(m2.values()), None), **ex})
                removed += ex["harness_messages_removed"]
                affected += ex["harness_messages_removed"] > 0
                f.write(json.dumps(ex) + "\n")
                kept += 1
    print(f"kept {kept}, dropped {dropped}; harness messages removed: {removed} in {affected} samples -> {a.out}")


if __name__ == "__main__":
    main()

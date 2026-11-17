"""Download every trajectory of a LinuxArena run as raw Inspect samples.

    python scripts/download.py [RUN_ID] [OUT_DIR]

Defaults to the GLM-5.3 simple-attack run (297 trajectories). Each trajectory is saved
as OUT_DIR/<trajectory_id>.json; existing files are skipped, so it can be re-run.
"""

import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API = "https://data.linuxarena.ai/api"
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "6b64412bf12644e0980961cd971ec05c"  # mentees-glm53-attack-300
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/raw")


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{API}/{path}", timeout=120) as r:
        return json.load(r)


def list_ids() -> list[str]:
    ids, page = [], 1
    while True:
        d = get(f"trajectories?run_id={RUN_ID}&documents=true&page={page}&page_size=100")
        ids += [t["trajectory_id"] for t in d["items"]]
        if page >= d["total_pages"]:
            break
        page += 1
    assert len(set(ids)) == d["total_count"], (len(set(ids)), d["total_count"])
    return list(dict.fromkeys(ids))


def fetch(tid: str) -> None:
    path = OUT / f"{tid}.json"
    if not path.exists():
        path.write_text(json.dumps(get(f"trajectories/{tid}")))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ids = list_ids()
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, ids))
    print(f"{len(ids)} trajectories in {OUT}")

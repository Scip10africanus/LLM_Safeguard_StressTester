"""Run an attack over behaviours with concurrency and resumable JSONL logs."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .attacks.base import Attack, AttackResult
from .behaviors import Behavior
from .models import ChatModel


def load_results(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["behavior_id"]] = rec
    return out


def run(
    attack: Attack,
    behaviors: list[Behavior],
    target: ChatModel,
    out_path: str | Path,
    budget: int = 20,
    workers: int = 4,
) -> dict[str, dict]:
    """Skips behaviours already in out_path, so interrupted runs resume."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_results(out)
    todo = [b for b in behaviors if b.id not in done]
    lock = threading.Lock()

    def work(b: Behavior) -> None:
        res: AttackResult = attack.run(b, target, budget)
        rec = {"attack": attack.name, "category": b.category, **asdict(res)}
        with lock:
            done[b.id] = rec
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    return done

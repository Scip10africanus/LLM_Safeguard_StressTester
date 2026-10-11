"""Behaviour loading with deterministic, disjoint train/eval splits."""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import httpx

JBB_URL = (
    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors"
    "/resolve/main/data/harmful-behaviors.csv"
)


@dataclass(frozen=True)
class Behavior:
    id: str
    goal: str
    target: str
    category: str
    source: str


def _parse_csv(text: str) -> list[Behavior]:
    rows = csv.DictReader(io.StringIO(text))
    return [
        Behavior(
            id=f"jbb-{r['Index']}",
            goal=r["Goal"],
            target=r["Target"],
            category=r["Category"],
            source=r["Source"],
        )
        for r in rows
    ]


def load_jbb(cache_dir: str | Path = ".cache") -> list[Behavior]:
    """Load JBB-Behaviors (harmful), caching the CSV locally."""
    cache = Path(cache_dir) / "jbb-harmful-behaviors.csv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = httpx.get(JBB_URL, follow_redirects=True, timeout=60)
        r.raise_for_status()
        cache.write_text(r.text, encoding="utf-8")
    return _parse_csv(cache.read_text(encoding="utf-8"))


def split_behaviors(
    behaviors: list[Behavior], eval_fraction: float = 0.5, seed: int = 0
) -> dict[str, list[Behavior]]:
    """Deterministic hash split by behaviour id; splits are disjoint by construction."""
    splits: dict[str, list[Behavior]] = {"train": [], "eval": []}
    for b in behaviors:
        h = hashlib.sha256(f"{seed}:{b.id}".encode()).digest()
        frac = int.from_bytes(h[:8], "big") / 2**64
        splits["eval" if frac < eval_fraction else "train"].append(b)
    assert not {b.id for b in splits["train"]} & {b.id for b in splits["eval"]}
    return splits

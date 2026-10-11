"""python -m redteam.cli configs/smoke.yaml"""
from __future__ import annotations

import json
import sys

import yaml

from .attacks.pair import PAIR
from .behaviors import load_jbb, split_behaviors
from .judge import LLMJudge
from .metrics import summarize
from .models import OpenAIChatModel
from .runner import run


def main(path: str) -> None:
    cfg = yaml.safe_load(open(path))
    target = OpenAIChatModel(**cfg["target"])
    attacker = OpenAIChatModel(**cfg["attacker"])
    judge = LLMJudge(OpenAIChatModel(**cfg["judge"]), threshold=cfg.get("success_threshold", 10))
    behaviors = split_behaviors(load_jbb(), seed=cfg.get("seed", 0))[cfg.get("split", "eval")]
    behaviors = behaviors[: cfg.get("limit", len(behaviors))]
    results = run(
        PAIR(attacker, judge), behaviors, target, cfg["out"],
        budget=cfg.get("budget", 20), workers=cfg.get("workers", 4),
    )
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main(sys.argv[1])

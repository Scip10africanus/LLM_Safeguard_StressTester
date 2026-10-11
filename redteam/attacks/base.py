from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..behaviors import Behavior
from ..models import ChatModel


@dataclass
class AttackResult:
    behavior_id: str
    success: bool
    queries: int  # target queries used
    best_score: int
    prompt: str = ""  # best attack prompt found
    response: str = ""  # target response to it
    history: list[dict] = field(default_factory=list)


class Attack(Protocol):
    name: str

    def run(self, behavior: Behavior, target: ChatModel, budget: int) -> AttackResult: ...

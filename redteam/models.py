"""Chat model interface plus an OpenAI-compatible HTTP client."""
from __future__ import annotations

import os
import threading
import time
from typing import Protocol

import httpx

Messages = list[dict[str, str]]


class ChatModel(Protocol):
    def generate(self, messages: Messages) -> str: ...


class OpenAIChatModel:
    """Works with any OpenAI-compatible endpoint (hosted, vLLM, OpenRouter...)."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 4,
        timeout: float = 120.0,
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self._client = httpx.Client(timeout=timeout)

    def generate(self, messages: Messages) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"environment variable {self.api_key_env} is not set")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {key}"}
        for attempt in range(self.retries + 1):
            try:
                r = self._client.post(self.url, json=payload, headers=headers)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except httpx.TransportError:
                if attempt == self.retries:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable")


class CountingModel:
    """Wraps a model and counts calls (thread-safe); used for query budgets."""

    def __init__(self, inner: ChatModel):
        self.inner = inner
        self.calls = 0
        self._lock = threading.Lock()

    def generate(self, messages: Messages) -> str:
        with self._lock:
            self.calls += 1
        return self.inner.generate(messages)

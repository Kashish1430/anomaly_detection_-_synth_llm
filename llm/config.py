from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["anthropic", "ollama"]


@dataclass
class LLMConfig:
    provider: Provider = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model_bulk: str = "claude-haiku-4-5"
    anthropic_model_accuracy: str = "claude-sonnet-5"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1"
    temperature: float = 0.0
    max_retries: int = 3
    concurrency: int = 5
    cache_path: str = "data/llm_cache/cache.sqlite3"

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = os.getenv("LLM_PROVIDER", cls.provider)
        if provider not in ("anthropic", "ollama"):
            raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")
        return cls(
            provider=provider,  # type: ignore[arg-type]
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_model=os.getenv("OLLAMA_MODEL", cls.ollama_model),
        )

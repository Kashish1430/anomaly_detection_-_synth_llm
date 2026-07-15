from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from llm.config import LLMConfig
from llm.costs import TokenUsage
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.schemas import ExplanationOutput

log = logging.getLogger(__name__)

TOOL_NAME = "record_explanation"


class LLMClient(ABC):
    model_name: str

    @abstractmethod
    async def generate_explanation(
        self,
        transaction: dict,
        features: dict,
        shap_values: dict[str, float],
        shap_base_value: float,
    ) -> tuple[ExplanationOutput, TokenUsage]:
        raise NotImplementedError


def _explanation_tool_schema() -> dict[str, Any]:
    schema = ExplanationOutput.model_json_schema()
    schema.pop("title", None)
    return {
        "name": TOOL_NAME,
        "description": "Record the structured explanation for a flagged transaction.",
        "input_schema": schema,
    }


class AnthropicClient(LLMClient):
    """Uses Anthropic's native tool-use for a guaranteed-JSON structured output -
    the strongest reliability guarantee of the two backends. See OpenAICompatibleClient
    for the weaker, model-dependent guarantee on the local path.
    """

    def __init__(self, config: LLMConfig, model: str | None = None) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.model_name = model or config.anthropic_model_bulk
        self._max_retries = config.max_retries
        self._tool = _explanation_tool_schema()

    async def generate_explanation(
        self,
        transaction: dict,
        features: dict,
        shap_values: dict[str, float],
        shap_base_value: float,
    ) -> tuple[ExplanationOutput, TokenUsage]:
        import anthropic

        user_prompt = build_user_prompt(transaction, features, shap_values, shap_base_value)
        delay = 1.0
        last_error: Exception | None = None

        for _ in range(self._max_retries):
            try:
                response = await self._client.messages.create(  # type: ignore[call-overload]
                    model=self.model_name,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    tools=[self._tool],
                    tool_choice={"type": "tool", "name": TOOL_NAME},
                    messages=[{"role": "user", "content": user_prompt}],
                )
                usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                for block in response.content:
                    if block.type == "tool_use" and block.name == TOOL_NAME:
                        return ExplanationOutput.model_validate(block.input), usage
                raise ValueError("Anthropic response contained no tool_use block")
            except anthropic.RateLimitError as exc:
                last_error = exc
                log.warning("Anthropic rate limited, retrying in %.1fs", delay)
            except anthropic.APIStatusError as exc:
                if exc.status_code < 500:
                    raise
                last_error = exc
                log.warning("Anthropic server error %s, retrying in %.1fs", exc, delay)
            await asyncio.sleep(delay)
            delay *= 2

        assert last_error is not None
        raise last_error


class OpenAICompatibleClient(LLMClient):
    """Targets any OpenAI-chat-completions-compatible endpoint - Ollama's /v1 API by
    default, also works against vLLM/LM Studio by changing `ollama_base_url`.

    Structured-output reliability here is model-dependent, unlike Anthropic's native
    tool-use guarantee - the bounded repair-retry loop below, plus the fact-checker and
    rule-based fallback in generate_explanations.py, are the safety net for that gap.
    """

    def __init__(self, config: LLMConfig) -> None:
        import openai

        self._client = openai.AsyncOpenAI(base_url=config.ollama_base_url, api_key="ollama")
        self.model_name = config.ollama_model
        self._temperature = config.temperature
        self._max_retries = config.max_retries

    async def generate_explanation(
        self,
        transaction: dict,
        features: dict,
        shap_values: dict[str, float],
        shap_base_value: float,
    ) -> tuple[ExplanationOutput, TokenUsage]:
        
        schema = ExplanationOutput.model_json_schema()
        json_instructions = (
            "\n\nRespond with ONLY a single JSON object matching this schema, no other "
            f"text, no markdown code fence:\n{json.dumps(schema)}"
        )
        user_prompt = build_user_prompt(transaction, features, shap_values, shap_base_value)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + json_instructions},
        ]

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            response = await self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model_name,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            try:
                print(content)
                return ExplanationOutput.model_validate_json(content), usage
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                log.warning(
                    "Ollama response failed schema validation (attempt %d/%d), asking for a repair",
                    attempt + 1,
                    self._max_retries,
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That response was invalid: {exc}. Reply again with ONLY "
                            "the corrected JSON object."
                        ),
                    }
                )

        assert last_error is not None
        raise last_error


def get_llm_client(config: LLMConfig) -> LLMClient:
    if config.provider == "anthropic":
        return AnthropicClient(config)
    if config.provider == "ollama":
        return OpenAICompatibleClient(config)
    raise ValueError(f"unknown LLM provider: {config.provider!r}")

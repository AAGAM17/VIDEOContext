"""OpenAI-compatible LLM provider for answer generation.

This adapter works with OpenAI, Azure OpenAI, vLLM, Ollama, and any other service
that implements the OpenAI chat-completions API shape.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import LLMConfig
from ..errors import ProviderError
from ..interfaces import LLMProvider
from ..logging import get_logger

log = get_logger("llm.openai")


@dataclass(slots=True)
class _LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenAILLM:
    """OpenAI-compatible chat completion for answer synthesis."""

    name = "openai"
    remote = True

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client: httpx.AsyncClient | None = None

    @property
    def version(self) -> str:
        return f"openai-api/{self.config.model or 'unknown'}"

    def available(self) -> bool:
        return bool(self.config.api_key or os.environ.get("OPENAI_API_KEY"))

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            base_url = self.config.base_url or "https://api.openai.com/v1"
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            headers = {"Authorization": f"Bearer {api_key}"}
            if self.config.organization:
                headers["OpenAI-Organization"] = self.config.organization
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kw: Any,
    ) -> str:
        """Synchronous wrapper for the pipeline; delegates to async."""
        import asyncio

        return asyncio.run(self._complete_async(prompt, system=system, **kw))

    async def _complete_async(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kw: Any,
    ) -> str:
        if not self.available():
            raise ProviderError(
                "No API key configured for OpenAI-compatible LLM",
                hint="Set OPENAI_API_KEY or configure LLMConfig.api_key",
            )

        client = await self._ensure_client()
        model = self.config.model or "gpt-4o-mini"

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": kw.get("temperature", self.config.temperature or 0.1),
            "max_tokens": kw.get("max_tokens", self.config.max_tokens or 500),
        }

        started = time.perf_counter()
        try:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"LLM API error {exc.response.status_code}: {exc.response.text}",
                hint="Check API key, model name, and quota.",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"LLM request failed: {type(exc).__name__}: {exc}",
                hint="Check network and API endpoint.",
            ) from exc

        duration = time.perf_counter() - started
        choice = data["choices"][0]
        message = choice["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        log.info(
            "llm.openai.done",
            extra={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration_s": round(duration, 3),
            },
        )

        return message

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


__all__ = ["OpenAILLM", "_LLMUsage"]
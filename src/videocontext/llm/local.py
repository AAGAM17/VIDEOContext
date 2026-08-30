"""Local LLM provider (Ollama, llama.cpp server, vLLM).

Uses the OpenAI-compatible API that these servers expose.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..config import LLMConfig
from ..errors import ProviderError
from ..interfaces import LLMProvider
from ..logging import get_logger

log = get_logger("llm.local")


class LocalLLM:
    """Local LLM via OpenAI-compatible API (Ollama, llama.cpp, vLLM)."""

    name = "local"
    remote = False

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client: httpx.AsyncClient | None = None

    @property
    def version(self) -> str:
        return f"local-llm/{self.config.model or 'unknown'}"

    def available(self) -> bool:
        return bool(self.config.base_url or os.environ.get("VIDEO_CONTEXT_LOCAL_LLM_URL"))

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            base_url = self.config.base_url or os.environ.get("VIDEO_CONTEXT_LOCAL_LLM_URL")
            if not base_url:
                raise ProviderError(
                    "Local LLM URL not configured",
                    hint="Set VIDEO_CONTEXT_LOCAL_LLM_URL or LLMConfig.base_url",
                )
            api_key = self.config.api_key or os.environ.get("VIDEO_CONTEXT_LOCAL_LLM_KEY", "dummy")
            headers = {"Authorization": f"Bearer {api_key}"}
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kw: Any,
    ) -> str:
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
                "Local LLM URL not configured",
                hint="Set VIDEO_CONTEXT_LOCAL_LLM_URL or LLMConfig.base_url",
            )

        client = await self._ensure_client()
        model = self.config.model or "llama3"

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

        try:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Local LLM API error {exc.response.status_code}: {exc.response.text}",
                hint="Check the local server is running and the model is loaded.",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Local LLM request failed: {type(exc).__name__}: {exc}",
                hint="Check server URL and model availability.",
            ) from exc

        choice = data["choices"][0]
        return choice["message"]["content"]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


__all__ = ["LocalLLM"]
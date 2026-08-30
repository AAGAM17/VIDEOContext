"""Remote vision providers — OpenAI-compatible, Gemini, local VLM.

These are *adapters*, not wrappers: they translate the project's
:class:`~videocontent.interfaces.VisionProvider` protocol into whatever HTTP
shape the upstream API expects. The core never imports a vendor SDK.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from ...config import VisionConfig
from ...errors import ProviderError
from ...interfaces import FrameContext, FrameImage, VisionOutput
from ...logging import get_logger

log = get_logger("vision.remote")

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _encode_image(path: Path) -> str:
    """Encode a frame as base64 data URI for the request."""
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode()


def _resolve_env(var: str, default: str | None = None) -> str | None:
    """Resolve an env var, with a descriptive error if missing."""
    value = os.environ.get(var)
    if value is None and default is None:
        raise ProviderError(
            f"Environment variable {var!r} is not set",
            hint=f"Export {var} or configure the provider with a key directly.",
        )
    return value or default


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _OpenAIMessage:
    role: Literal["user", "assistant", "system"]
    content: list[dict[str, Any]]


class OpenAIVision:
    """OpenAI-compatible vision (OpenAI, Azure OpenAI, vLLM, etc.).

    The adapter is deliberately thin: it builds the chat-completion payload,
    sends it with httpx, and normalises the response into :class:`VisionOutput`.
    It never assumes a specific model — the caller chooses via config.
    """

    name = "openai"
    remote = True

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
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

    def describe(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[VisionOutput]:
        """Synchronous wrapper for the pipeline; delegates to async."""
        import asyncio

        return asyncio.run(self._describe_async(frames, ctx))

    async def _describe_async(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[VisionOutput]:
        if not frames:
            return []

        client = await self._ensure_client()
        model = self.config.model or "gpt-4o"
        prompt = self.config.prompt or self._default_prompt()

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(frame.path)}"},
                }
            )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_tokens or 500,
            "temperature": 0.1,
        }

        started = time.perf_counter()
        try:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Vision API error {exc.response.status_code}: {exc.response.text}",
                hint="Check API key, model name, and quota.",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Vision request failed: {type(exc).__name__}: {exc}",
                hint="Check network and API endpoint.",
            ) from exc

        duration = time.perf_counter() - started
        choice = data["choices"][0]
        message = choice["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        log.info(
            "vision.openai.done",
            extra={
                "model": model,
                "frames": len(frames),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_s": round(duration, 3),
            },
        )

        # Single output spanning all frames — the prompt asked for one description
        start_ts = frames[0].ts
        end_ts = frames[-1].ts
        return [
            VisionOutput(
                description=message,
                start=start_ts,
                end=end_ts,
                frame_ids=[f.id for f in frames],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        ]

    def _default_prompt(self) -> str:
        return (
            "Describe what is happening in these video frames. Focus on: "
            "1) any on-screen text (UI, code, slides, terminal output) "
            "2) visible applications and their state "
            "3) actions being performed "
            "4) important visual changes between frames. "
            "Be specific and timestamp-aware."
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


class GeminiVision:
    """Google Gemini vision provider.

    Uses the Gemini REST API directly (no google-generativeai dependency).
    """

    name = "gemini"
    remote = True

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self._client: httpx.AsyncClient | None = None

    @property
    def version(self) -> str:
        return f"gemini-api/{self.config.model or 'unknown'}"

    def available(self) -> bool:
        return bool(self.config.api_key or os.environ.get("GEMINI_API_KEY"))

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY")
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            self._client = httpx.AsyncClient(
                base_url=base_url,
                params={"key": api_key},
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def describe(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[VisionOutput]:
        import asyncio

        return asyncio.run(self._describe_async(frames, ctx))

    async def _describe_async(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[VisionOutput]:
        if not frames:
            return []

        client = await self._ensure_client()
        model = self.config.model or "gemini-1.5-flash"
        prompt = self.config.prompt or self._default_prompt()

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for frame in frames:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": _encode_image(frame.path),
                    }
                }
            )

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": self.config.max_tokens or 500,
                "temperature": 0.1,
            },
        }

        started = time.perf_counter()
        try:
            resp = await client.post(f"/models/{model}:generateContent", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Gemini API error {exc.response.status_code}: {exc.response.text}",
                hint="Check API key and model name.",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Gemini request failed: {type(exc).__name__}: {exc}",
                hint="Check network and API endpoint.",
            ) from exc

        duration = time.perf_counter() - started
        candidate = data["candidates"][0]
        message = candidate["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0)
        tokens_out = usage.get("candidatesTokenCount", 0)

        log.info(
            "vision.gemini.done",
            extra={
                "model": model,
                "frames": len(frames),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_s": round(duration, 3),
            },
        )

        start_ts = frames[0].ts
        end_ts = frames[-1].ts
        return [
            VisionOutput(
                description=message,
                start=start_ts,
                end=end_ts,
                frame_ids=[f.id for f in frames],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        ]

    def _default_prompt(self) -> str:
        return (
            "Describe what is happening in these video frames. Focus on: "
            "1) any on-screen text (UI, code, slides, terminal output) "
            "2) visible applications and their state "
            "3) actions being performed "
            "4) important visual changes between frames. "
            "Be specific and timestamp-aware."
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Local VLM provider (Ollama / llama.cpp / vLLM OpenAI-compatible)
# ---------------------------------------------------------------------------


class LocalVLMVision:
    """Local vision-language model via OpenAI-compatible API (Ollama, llama.cpp server, vLLM).

    Assumes the server runs an OpenAI-compatible /chat/completions endpoint with
    vision support (e.g. Ollama with llava, llama.cpp server with a vision model).
    """

    name = "local-vlm"
    remote = False  # runs on the machine, but may be a separate process

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self._client: httpx.AsyncClient | None = None

    @property
    def version(self) -> str:
        return f"local-vlm/{self.config.model or 'unknown'}"

    def available(self) -> bool:
        # Check if base_url is set (required for local VLM)
        return bool(self.config.base_url or os.environ.get("VIDEO_CONTEXT_LOCAL_VLM_URL"))

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            base_url = self.config.base_url or _resolve_env("VIDEO_CONTEXT_LOCAL_VLM_URL")
            api_key = self.config.api_key or os.environ.get("VIDEO_CONTEXT_LOCAL_VLM_KEY", "dummy")
            headers = {"Authorization": f"Bearer {api_key}"}
            self._client = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=httpx.Timeout(300.0, connect=10.0),  # local models can be slower
            )
        return self._client

    def describe(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[VisionOutput]:
        import asyncio

        return asyncio.run(self._describe_async(frames, ctx))

    async def _describe_async(
        self, frames: list[FrameImage], ctx: FrameContext
    ) -> list[VisionOutput]:
        if not frames:
            return []

        client = await self._ensure_client()
        model = self.config.model or "llava"
        prompt = self.config.prompt or self._default_prompt()

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in frames:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(frame.path)}"},
                }
            )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config.max_tokens or 500,
            "temperature": 0.1,
        }

        started = time.perf_counter()
        try:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Local VLM API error {exc.response.status_code}: {exc.response.text}",
                hint="Check the local server is running and the model supports vision.",
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"Local VLM request failed: {type(exc).__name__}: {exc}",
                hint="Check server URL and model availability.",
            ) from exc

        duration = time.perf_counter() - started
        choice = data["choices"][0]
        message = choice["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        log.info(
            "vision.local_vlm.done",
            extra={
                "model": model,
                "frames": len(frames),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "duration_s": round(duration, 3),
            },
        )

        start_ts = frames[0].ts
        end_ts = frames[-1].ts
        return [
            VisionOutput(
                description=message,
                start=start_ts,
                end=end_ts,
                frame_ids=[f.id for f in frames],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        ]

    def _default_prompt(self) -> str:
        return (
            "Describe what is happening in these video frames. Focus on: "
            "1) any on-screen text (UI, code, slides, terminal output) "
            "2) visible applications and their state "
            "3) actions being performed "
            "4) important visual changes between frames. "
            "Be specific and timestamp-aware."
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


__all__ = ["OpenAIVision", "GeminiVision", "LocalVLMVision"]
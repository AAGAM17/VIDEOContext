"""Null LLM provider — the default when no LLM is configured.

This provider does nothing and is always available. It is used when the user
has not configured an LLM provider, making ask() fail gracefully with a clear
error rather than a confusing missing-dependency error.
"""

from __future__ import annotations

from ..config import LLMConfig
from ..errors import ProviderError
from ..interfaces import LLMProvider


class NullLLM:
    """No LLM configured — ask() will fail with a clear message."""

    name = "null"
    remote = False

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    @property
    def version(self) -> str:
        return "1.0.0"

    def available(self) -> bool:
        return True

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        **kw: Any,
    ) -> str:
        raise ProviderError(
            "No LLM provider configured — cannot generate answers",
            hint=(
                "Configure an LLM provider: "
                "VIDEO_CONTEXT_LLM_PROVIDER=openai (requires OPENAI_API_KEY) "
                "or VIDEO_CONTEXT_LLM_PROVIDER=local (requires local server). "
                "Or install 'videocontent[api]' for the OpenAI-compatible adapter."
            ),
        )


__all__ = ["NullLLM"]
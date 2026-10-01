"""LLM providers — the model that generates answers from retrieved evidence.

The ``ask()`` pipeline is: question → retrieval → context assembly → LLM → answer.
The LLM provider is the last step and is *never* required for processing or search —
it is an optional convenience for the common case of "just give me the answer".

The HTTP-backed providers need ``httpx`` (an optional extra), so they are imported on first
use: ``from videocontent.llm import NullLLM`` must work on a base install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .null import NullLLM

if TYPE_CHECKING:  # pragma: no cover
    from .local import LocalLLM
    from .openai_llm import OpenAILLM


def __getattr__(name: str) -> Any:
    if name == "OpenAILLM":
        from .openai_llm import OpenAILLM

        return OpenAILLM
    if name == "LocalLLM":
        from .local import LocalLLM

        return LocalLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["LocalLLM", "NullLLM", "OpenAILLM"]

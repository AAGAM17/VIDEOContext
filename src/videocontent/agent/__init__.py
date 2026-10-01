"""The agent-facing layer: bounded, provenance-carrying operations for coding agents.

Everything here sits *above* the SDK. It adds no intelligence of its own; it resolves
what an agent points at (a video, a ``.vctx``, a URL), calls the existing SDK, and shapes
the result into one stable, size-capped JSON envelope. The CLI's agent commands and the
``videocontent mcp`` server both call :mod:`videocontent.agent.ops`, so the two surfaces
cannot drift apart.
"""

from __future__ import annotations

from .ops import AGENT_SCHEMA, UNTRUSTED_NOTICE, AgentError, NotAnalyzedError

__all__ = ["AGENT_SCHEMA", "UNTRUSTED_NOTICE", "AgentError", "NotAnalyzedError"]

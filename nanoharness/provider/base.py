from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Protocol

from nanoharness.core.context import Message


# ─── Error Taxonomy ───────────────────────────────────────────────────────────

class ProviderErrorType(StrEnum):
    RATE_LIMITED = "rate_limited"        # 429 — exponential backoff + jitter
    CONTEXT_TOO_LONG = "context_too_long"  # 400 context_length_exceeded → compact
    AUTH_INVALID = "auth_invalid"        # 401/403 — fatal, no retry
    SERVER_ERROR = "server_error"        # 500/529 — limited retry
    TIMEOUT = "timeout"                  # asyncio.TimeoutError
    UNKNOWN = "unknown"


class ProviderError(Exception):
    def __init__(self, error_type: ProviderErrorType, message: str, retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


# ─── Response Model ───────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class LLMResponse:
    """Parsed response from the LLM provider."""
    raw_content: list[dict[str, Any]]   # Anthropic-format content blocks
    stop_reason: str                     # "end_turn" | "tool_use" | "max_tokens"
    input_tokens: int = 0
    output_tokens: int = 0

    # extracted helpers
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_text: str = ""

    @property
    def wants_tool_call(self) -> bool:
        return self.stop_reason == "tool_use" and bool(self.tool_calls)

    @property
    def is_final_answer(self) -> bool:
        return self.stop_reason in ("end_turn", "max_tokens") and not self.tool_calls

    def to_assistant_message(self) -> Message:
        return Message(
            role="assistant",
            content=self.raw_content,
            token_count=self.output_tokens,
        )


# ─── Streaming Chunk ──────────────────────────────────────────────────────────

@dataclass
class StreamChunk:
    """A single streaming event from the provider."""
    delta_text: str = ""                        # incremental text
    tool_call_delta: dict[str, Any] | None = None  # partial tool_use block
    is_final: bool = False
    final_response: LLMResponse | None = None   # set when is_final=True


# ─── Provider Protocol ────────────────────────────────────────────────────────

class LLMProvider(Protocol):
    model_id: str

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Non-streaming call. Returns full response."""
        ...

    async def stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming call. Yields StreamChunks; final chunk has is_final=True."""
        ...

    def count_tokens(self, messages: list[Message], system: str = "") -> int:
        """Approximate token count for context-window budget checks."""
        ...


# ─── Error Classifier ─────────────────────────────────────────────────────────

def classify_provider_error(exc: Exception) -> ProviderError:
    """
    Normalize provider-specific exceptions into ProviderError.
    Each concrete provider can call this and add its own pre-checks first.
    """
    msg = str(exc).lower()

    if "rate" in msg or "429" in msg or "overloaded" in msg:
        return ProviderError(ProviderErrorType.RATE_LIMITED, str(exc), retryable=True)
    if "context" in msg and ("length" in msg or "window" in msg or "too long" in msg):
        return ProviderError(ProviderErrorType.CONTEXT_TOO_LONG, str(exc), retryable=False)
    if "auth" in msg or "401" in msg or "403" in msg or "api key" in msg:
        return ProviderError(ProviderErrorType.AUTH_INVALID, str(exc), retryable=False)
    if "500" in msg or "502" in msg or "529" in msg or "server error" in msg:
        return ProviderError(ProviderErrorType.SERVER_ERROR, str(exc), retryable=True)
    if "timeout" in msg or "timed out" in msg:
        return ProviderError(ProviderErrorType.TIMEOUT, str(exc), retryable=True)

    return ProviderError(ProviderErrorType.UNKNOWN, str(exc), retryable=False)

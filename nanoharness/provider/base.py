from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Protocol

from nanoharness.core.context import Message


# ─── Error Taxonomy / 错误分类 ───────────────────────────────────────────────

class ProviderErrorType(StrEnum):
    RATE_LIMITED = "rate_limited"        # 429 — 指数退避 + 抖动 / 429 — exponential backoff + jitter
    CONTEXT_TOO_LONG = "context_too_long"  # 400 context_length_exceeded → 压缩 / 400 context_length_exceeded → compact
    AUTH_INVALID = "auth_invalid"        # 401/403 — 致命错误，不重试 / 401/403 — fatal, no retry
    SERVER_ERROR = "server_error"        # 500/529 — 有限重试 / 500/529 — limited retry
    TIMEOUT = "timeout"                  # asyncio.TimeoutError
    UNKNOWN = "unknown"


class ProviderError(Exception):
    def __init__(self, error_type: ProviderErrorType, message: str, retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type
        self.retryable = retryable


# ─── Response Model / 响应模型 ───────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class LLMResponse:
    """解析后的 LLM provider 响应。 / Parsed response from the LLM provider."""
    raw_content: list[dict[str, Any]]   # Anthropic 格式 content blocks / Anthropic-format content blocks
    stop_reason: str                     # "end_turn" | "tool_use" | "max_tokens"
    input_tokens: int = 0
    output_tokens: int = 0

    # 提取出的辅助字段 / extracted helpers
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


# ─── Streaming Chunk / 流式块 ────────────────────────────────────────────────

@dataclass
class StreamChunk:
    """来自 provider 的单个流式事件。 / A single streaming event from the provider."""
    delta_text: str = ""                        # 增量文本 / incremental text
    tool_call_delta: dict[str, Any] | None = None  # 部分 tool_use block / partial tool_use block
    is_final: bool = False
    final_response: LLMResponse | None = None   # is_final=True 时设置 / set when is_final=True


# ─── Provider Protocol / Provider 协议 ───────────────────────────────────────

class LLMProvider(Protocol):
    model_id: str

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """非流式调用，返回完整响应。 / Non-streaming call. Returns full response."""
        ...

    async def stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """流式调用，产出 StreamChunk；最后一个块 is_final=True。 / Streaming call. Yields StreamChunks; final chunk has is_final=True."""
        ...

    def count_tokens(self, messages: list[Message], system: str = "") -> int:
        """用于上下文窗口预算检查的近似 token 数。 / Approximate token count for context-window budget checks."""
        ...


# ─── Error Classifier / 错误分类器 ───────────────────────────────────────────

def classify_provider_error(exc: Exception) -> ProviderError:
    """
    将 provider 特定异常归一化为 ProviderError。 / Normalize provider-specific exceptions into ProviderError.
    每个 provider 可调用此函数，并先加上自己的前置检查。 / Each concrete provider can call this and add its own pre-checks first.
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

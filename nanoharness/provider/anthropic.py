from __future__ import annotations

import json
from typing import Any, AsyncIterator

import anthropic

from nanoharness.core.context import Message
from nanoharness.provider.base import (
    LLMResponse,
    ProviderError,
    ProviderErrorType,
    StreamChunk,
    ToolCall,
    classify_provider_error,
)


class AnthropicProvider:
    """
    Claude provider using the Anthropic Python SDK.

    Supports:
      - Streaming and non-streaming completions
      - tool_use / tool_result message format
      - Extended thinking (T3 tier, budget_tokens > 0)
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        thinking_budget_tokens: int = 0,   # >0 enables extended thinking (T3 only)
    ) -> None:
        self.model_id = model_id
        self._thinking_budget = thinking_budget_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    # ── Non-streaming ─────────────────────────────────────────────────────────

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(system, messages, tools, max_tokens)
        try:
            resp = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise self._classify(exc) from exc

        return self._parse_response(resp)

    # ── Streaming ─────────────────────────────────────────────────────────────

    async def stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = self._build_kwargs(system, messages, tools, max_tokens)
        try:
            async with self._client.messages.stream(**kwargs) as stream_mgr:
                async for event in stream_mgr:
                    chunk = self._parse_stream_event(event)
                    if chunk is not None:
                        yield chunk

                # Final chunk with complete response
                final = await stream_mgr.get_final_message()
                yield StreamChunk(is_final=True, final_response=self._parse_response(final))

        except anthropic.APIError as exc:
            raise self._classify(exc) from exc

    # ── Token counting ────────────────────────────────────────────────────────

    def count_tokens(self, messages: list[Message], system: str = "") -> int:
        # Heuristic: Anthropic charges ~(chars / 4). Good enough for preflight checks.
        total = len(system) // 4
        for m in messages:
            if m.token_count > 0:
                total += m.token_count
            elif isinstance(m.content, str):
                total += len(m.content) // 4
            else:
                total += len(json.dumps(m.content)) // 4
        return total

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_kwargs(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "system": system,
            "messages": [m.to_api_dict() for m in messages],
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if self._thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }
        return kwargs

    def _parse_response(self, resp: Any) -> LLMResponse:
        content_blocks = [b.model_dump() for b in resp.content]
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []

        for block in resp.content:
            if block.type == "tool_use":
                tool_calls.append(ToolCall(
                    tool_use_id=block.id,
                    tool_name=block.name,
                    tool_input=block.input,
                ))
            elif block.type == "text":
                text_parts.append(block.text)

        return LLMResponse(
            raw_content=content_blocks,
            stop_reason=resp.stop_reason or "end_turn",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            tool_calls=tool_calls,
            final_text="".join(text_parts),
        )

    def _parse_stream_event(self, event: Any) -> StreamChunk | None:
        event_type = getattr(event, "type", "")
        if event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta and getattr(delta, "type", "") == "text_delta":
                return StreamChunk(delta_text=delta.text)
        return None

    def _classify(self, exc: anthropic.APIError) -> ProviderError:
        if isinstance(exc, anthropic.RateLimitError):
            return ProviderError(ProviderErrorType.RATE_LIMITED, str(exc), retryable=True)
        if isinstance(exc, anthropic.AuthenticationError):
            return ProviderError(ProviderErrorType.AUTH_INVALID, str(exc), retryable=False)
        if isinstance(exc, anthropic.BadRequestError) and "context" in str(exc).lower():
            return ProviderError(ProviderErrorType.CONTEXT_TOO_LONG, str(exc), retryable=False)
        if isinstance(exc, anthropic.InternalServerError):
            return ProviderError(ProviderErrorType.SERVER_ERROR, str(exc), retryable=True)
        return classify_provider_error(exc)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from nanoharness.core.event_store import AgentEvent


# ─── Hook 上下文 / Hook Contexts ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TurnHookContext:
    session_key: str
    agent_id: str
    turn_id: str
    trace_id: str
    user_message: str


@dataclass(frozen=True)
class TurnHookResult:
    final_text: str = ""
    error_message: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class ToolHookContext:
    session_key: str
    agent_id: str
    trace_id: str
    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]


@dataclass(frozen=True)
class ToolHookResult:
    tool_name: str
    tool_use_id: str
    success: bool
    output_preview: str
    latency_ms: float


@dataclass(frozen=True)
class CompactionHookContext:
    session_key: str
    agent_id: str
    trace_id: str
    tokens_before: int
    messages_before: int


@dataclass(frozen=True)
class CompactionHookResult:
    tokens_after: int
    messages_after: int


# ─── Hook 协议 / Hook Protocols ───────────────────────────────────────────────────────────
# 所有 hook 方法都是可选的（返回 None） / All hook methods are optional (return None).
# hook 抛出的异常会被捕获并记录为 WARN，绝不 / Exceptions thrown by hooks are caught and logged (WARN); they never
# 传播到主 turn 流程 / propagate to the main turn flow.

@runtime_checkable
class TurnHook(Protocol):
    async def before_turn(self, ctx: TurnHookContext) -> None: ...
    async def after_turn(self, ctx: TurnHookContext, result: TurnHookResult) -> None: ...
    async def on_error(self, ctx: TurnHookContext, exc: Exception) -> None: ...
    async def on_event(self, event: AgentEvent) -> None: ...


@runtime_checkable
class ToolHook(Protocol):
    async def before_tool(self, ctx: ToolHookContext) -> None: ...
    async def after_tool(self, ctx: ToolHookContext, result: ToolHookResult) -> None: ...


@runtime_checkable
class CompactionHook(Protocol):
    async def before_compact(self, ctx: CompactionHookContext) -> None: ...
    async def after_compact(self, ctx: CompactionHookContext, result: CompactionHookResult) -> None: ...

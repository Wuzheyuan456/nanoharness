from __future__ import annotations

import logging

from nanoharness.core.event_store import AgentEvent
from nanoharness.engine.hooks.types import (
    CompactionHookContext,
    CompactionHookResult,
    ToolHookContext,
    ToolHookResult,
    TurnHookContext,
    TurnHookResult,
)

log = logging.getLogger(__name__)


class DefaultTurnHook:
    """空操作 TurnHook，作为未注册自定义 hook 时的回退 / No-op TurnHook. Used as fallback when no custom hook is registered."""

    async def before_turn(self, ctx: TurnHookContext) -> None:
        pass

    async def after_turn(self, ctx: TurnHookContext, result: TurnHookResult) -> None:
        pass

    async def on_error(self, ctx: TurnHookContext, exc: Exception) -> None:
        log.warning("turn error [%s/%s]: %s", ctx.session_key, ctx.turn_id, exc)

    async def on_event(self, event: AgentEvent) -> None:
        pass


class DefaultToolHook:
    """空操作 ToolHook / No-op ToolHook."""

    async def before_tool(self, ctx: ToolHookContext) -> None:
        pass

    async def after_tool(self, ctx: ToolHookContext, result: ToolHookResult) -> None:
        pass


class DefaultCompactionHook:
    """空操作 CompactionHook / No-op CompactionHook."""

    async def before_compact(self, ctx: CompactionHookContext) -> None:
        pass

    async def after_compact(
        self,
        ctx: CompactionHookContext,
        result: CompactionHookResult,
    ) -> None:
        pass


# ─── 安全 Hook 调用器 / Safe Hook Caller ─────────────────────────────────────────────────────────

async def safe_call(coro_or_none: object) -> None:
    """
    等待一个 hook 协程，吞掉所有异常 / Await a hook coroutine, swallowing any exception.
    hook 绝不能让主 turn 流程崩溃 / Hooks must never crash the main turn flow.
    """
    import asyncio
    if coro_or_none is None:
        return
    try:
        if asyncio.iscoroutine(coro_or_none):
            await coro_or_none
    except Exception as exc:
        log.warning("hook raised (swallowed): %s", exc)

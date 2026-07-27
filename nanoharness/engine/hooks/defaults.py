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

# ─── 具体质量门 Hook / Concrete quality-gate hooks ───────────────────────────

class InputSanitizationHook:
    """
    before_turn：检查输入长度，超限时记录 warning。 / before_turn: checks input length, logs warning if oversized.

    面试话术 / Interview talking point:
    "最常见的用户错误是把大文档直接粘进对话——这会撑爆 context window 或拖慢响应。
    InputSanitizationHook 在 before_turn 拦截，记录 warning 给运维监控，
    调用方可据此在应用层截断或提示用户使用文件上传。"
    """

    def __init__(self, max_chars: int = 10_000) -> None:
        self._max_chars = max_chars

    async def before_turn(self, ctx: TurnHookContext) -> None:
        length = len(ctx.user_message)
        if length > self._max_chars:
            log.warning(
                "输入过长 %d 字符（建议上限 %d），session=%s / input oversized: %d chars (suggested limit %d), session=%s",
                length, self._max_chars, ctx.session_key,
                length, self._max_chars, ctx.session_key,
            )

    async def after_turn(self, ctx: TurnHookContext, result: TurnHookResult) -> None:
        pass

    async def on_error(self, ctx: TurnHookContext, exc: Exception) -> None:
        pass

    async def on_event(self, event: AgentEvent) -> None:
        pass


class TurnMetricsHook:
    """
    after_turn / on_error：将延迟、token、错误写入 observability metrics。 / after_turn / on_error: writes latency, tokens, errors to observability metrics.

    Phase 2 ↔ Phase 7 的桥接 hook，让 Gradio 面板展示实时 turn 统计。 /
    Bridge hook between Phase 2 and Phase 7 — feeds the Gradio dashboard's real-time turn stats.

    面试话术 / Interview talking point:
    "Phase 7 的 metrics 面板数据要从哪来？以前 TurnRunner hook 是空的，
    数字只能靠手写测试造。TurnMetricsHook 是接通链路的那一环：
    after_turn 把 elapsed_ms / token 写进 MetricsCollector，
    on_error 记 error_type，Grafana 就有了四大黄金信号里的 latency 和 errors。"
    """

    def __init__(self, metrics: object | None = None) -> None:
        if metrics is None:
            from nanoharness.observability.metrics import get_metrics
            metrics = get_metrics()
        self._m = metrics

    async def before_turn(self, ctx: TurnHookContext) -> None:
        pass

    async def after_turn(self, ctx: TurnHookContext, result: TurnHookResult) -> None:
        self._m.observe_latency(result.elapsed_ms / 1000.0, kind="turn", model=ctx.model_id)  # type: ignore[attr-defined]
        self._m.inc_tokens(result.total_input_tokens, token_type="input")  # type: ignore[attr-defined]
        self._m.inc_tokens(result.total_output_tokens, token_type="output")  # type: ignore[attr-defined]

    async def on_error(self, ctx: TurnHookContext, exc: Exception) -> None:
        self._m.inc_error(error_type=type(exc).__name__)  # type: ignore[attr-defined]

    async def on_event(self, event: AgentEvent) -> None:
        pass


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

from __future__ import annotations

import asyncio
import logging
import time
from contextvars import ContextVar
from typing import Any, AsyncIterator

from nanoharness.core.context import AgentContext, Message
from nanoharness.core.event_store import AgentEvent, DoneEvent, ErrorEvent, EventStore
from nanoharness.core.nano_core import NanoCore
from nanoharness.engine.hooks.defaults import DefaultTurnHook, safe_call
from nanoharness.engine.hooks.types import TurnHookContext, TurnHookResult
from nanoharness.provider.base import LLMProvider
from nanoharness.router.llm_router import LLMRouter
from nanoharness.router.tiers import Tier, TierRegistry

log = logging.getLogger(__name__)

# ─── Per-session 锁管理 / per-session lock management ────────────────────────────────────────────────────────

# 全局 session_key → asyncio.Lock 映射 / global session_key → asyncio.Lock mapping
_session_locks: dict[str, asyncio.Lock] = {}

# ContextVar：记录当前协程持有哪些锁的 id，用于重入检测 / ContextVar: tracks which lock ids the current coroutine holds, for reentrance detection
# 每个 asyncio Task 有自己的 ContextVar 副本，天然隔离 / each asyncio Task has its own ContextVar copy, naturally isolated
_LOCK_OWNER: ContextVar[frozenset[int]] = ContextVar("_lock_owner", default=frozenset())


def _get_session_lock(session_key: str) -> asyncio.Lock:
    if session_key not in _session_locks:
        _session_locks[session_key] = asyncio.Lock()
    return _session_locks[session_key]


# ─── TurnRunner ────────────────────────────────────────────────────────────────

class TurnRunner:
    """
    编排一次完整 turn 的生命周期。 / Orchestrates the lifecycle of a complete turn.

    职责： / Responsibilities:
    1. per-session 串行化：同一 session 同时只允许一个 turn 运行 / per-session serialization: only one turn runs at a time for the same session
    2. ContextVar 重入检测：subagent 用相同 session_key 时不死锁 / ContextVar reentrance detection: no deadlock when a subagent uses the same session_key
    3. 调用 LLMRouter 决定档位，选择对应 provider / Calls LLMRouter to decide the tier and selects the corresponding provider
    4. 驱动 NanoCore async generator，向外透传所有 AgentEvent / Drives the NanoCore async generator, forwarding all AgentEvents outward
    5. 在 turn 前后触发 TurnHook / Triggers TurnHook before and after the turn

    面试话术： / Interview talking point:
    "TurnRunner 用 per-session asyncio.Lock 保证同一对话串行，
    用 ContextVar 检测重入——当 subagent 以相同 session_key 进来时，
    它已经持有锁，直接跳过等待。两者配合既不会并发乱序，也不会死锁。"

    "TurnRunner uses a per-session asyncio.Lock to serialize the same conversation,
    and a ContextVar to detect reentrance — when a subagent arrives with the same session_key,
    it already holds the lock and skips waiting. Together they prevent both concurrent reordering and deadlock."
    """

    def __init__(
        self,
        provider_factory: dict[Tier, LLMProvider],   # tier → provider 实例 / tier → provider instance
        registry: TierRegistry | None = None,
        router: LLMRouter | None = None,
        event_store: EventStore | None = None,
        hooks: list[Any] | None = None,
        default_tier: Tier = Tier.T1,
    ) -> None:
        self._providers = provider_factory
        self._registry = registry or TierRegistry()
        self._router = router
        self._event_store = event_store
        self._hooks = hooks or [DefaultTurnHook()]
        self._default_tier = default_tier

    async def run(
        self,
        ctx: AgentContext,
        user_message: str,
        tools: dict[str, Any] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        compaction: Any | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        主入口，返回 async generator。 / Main entry, returns an async generator.
        调用方 `async for event in runner.run(ctx, msg)` 消费。 / Callers consume via `async for event in runner.run(ctx, msg)`.
        """
        lock = _get_session_lock(ctx.session_key)
        lock_id = id(lock)
        owned = _LOCK_OWNER.get()

        if lock_id in owned:
            # 当前协程已经持有这个 session 的锁（重入场景：subagent） / Current coroutine already holds this session's lock (reentrance case: subagent)
            # 直接执行，不再等待锁，避免死锁 / Execute directly without waiting for the lock to avoid deadlock
            async for ev in self._execute(ctx, user_message, tools, tool_definitions, compaction):
                yield ev
        else:
            async with lock:
                token = _LOCK_OWNER.set(owned | {lock_id})
                try:
                    async for ev in self._execute(ctx, user_message, tools, tool_definitions, compaction):
                        yield ev
                finally:
                    _LOCK_OWNER.reset(token)

    async def _execute(
        self,
        ctx: AgentContext,
        user_message: str,
        tools: dict[str, Any] | None,
        tool_definitions: list[dict[str, Any]] | None,
        compaction: Any | None,
    ) -> AsyncIterator[AgentEvent]:
        import uuid
        trace_id = uuid.uuid4().hex
        turn_id = uuid.uuid4().hex
        t0 = time.monotonic()

        # ── 路由决策（先于 hook_ctx 创建，以便把 model_id 注入 hook_ctx） / routing first so model_id can be injected into hook_ctx ──
        tier = self._default_tier
        if self._router:
            try:
                result = await self._router.classify(
                    user_message,
                    trace_id=trace_id,
                    session_key=ctx.session_key,
                )
                tier = result.tier
                # 路由策略 hint 注入 extra_context / Inject routing policy hint into extra_context
                ctx.extra_context["router_tier"] = str(tier)
                ctx.extra_context["router_hint"] = self._registry.policy_hint(tier)
                log.info("路由决策: session=%s tier=%s confidence=%.2f method=%s",
                         ctx.session_key, tier, result.confidence, result.method)
            except Exception as exc:
                log.warning("路由失败，使用默认档位 %s: %s", self._default_tier, exc)

        # ── 选择对应档位的 provider / select the provider for the chosen tier ─────────────────────────
        provider = self._providers.get(tier) or self._providers.get(self._default_tier)
        if provider is None:
            raise RuntimeError(f"未找到 tier={tier} 的 provider，请检查 provider_factory 配置")

        # 更新 ctx 的模型 ID（用于日志和 cost 统计） / Update ctx's model ID (for logging and cost stats)
        ctx.model_id = provider.model_id

        # ── 创建 hook 上下文（路由后，含 model_id）/ Create hook context (post-routing, includes model_id) ──
        hook_ctx = TurnHookContext(
            session_key=ctx.session_key,
            agent_id=ctx.agent_id,
            turn_id=turn_id,
            trace_id=trace_id,
            user_message=user_message,
            model_id=ctx.model_id,
        )

        # ── before_turn 钩子 / before_turn hook ──────────────────────────────────────────────────
        for hook in self._hooks:
            await safe_call(hook.before_turn(hook_ctx))

        # ── 驱动 NanoCore / drive NanoCore ─────────────────────────────────────────────────────
        core = NanoCore(
            ctx=ctx,
            provider=provider,
            tools=tools or {},
            tool_definitions=tool_definitions or [],
            compaction=compaction,
            event_store=self._event_store,
        )

        done_event: DoneEvent | None = None
        error_event: ErrorEvent | None = None

        try:
            async for event in core.run_turn(user_message):
                # 触发 on_event hook（吞异常，不阻塞主流程） / Trigger on_event hook (swallow exceptions, don't block the main flow)
                for hook in self._hooks:
                    await safe_call(hook.on_event(event))
                if isinstance(event, DoneEvent):
                    done_event = event
                elif isinstance(event, ErrorEvent):
                    error_event = event
                yield event
        except Exception as exc:
            error_event = ErrorEvent(
                trace_id=trace_id,
                session_key=ctx.session_key,
                agent_id=ctx.agent_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            yield error_event

        elapsed = (time.monotonic() - t0) * 1000

        # ── after_turn / on_error 钩子 / after_turn / on_error hook ────────────────────────────────────────
        if done_event:
            result_summary = TurnHookResult(
                final_text=done_event.final_text,
                total_input_tokens=done_event.total_input_tokens,
                total_output_tokens=done_event.total_output_tokens,
                total_tool_calls=done_event.total_tool_calls,
                elapsed_ms=elapsed,
            )
            for hook in self._hooks:
                await safe_call(hook.after_turn(hook_ctx, result_summary))

        if error_event:
            exc_obj = RuntimeError(error_event.error_message)
            for hook in self._hooks:
                await safe_call(hook.on_error(hook_ctx, exc_obj))

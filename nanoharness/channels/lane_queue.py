"""
车道式会话隔离队列（Lane Queue）/ Lane-based session isolation queue.

核心问题：Gateway 收到消息后交给 Agent 处理，Agent 内部有状态（history、
TurnRunner 的 per-session 锁）。如果同一会话的两条消息并发处理，
第二条可能读到第一条还没写完的 history，导致上下文错乱 / Core problem: concurrent messages in the same session may read half-written history, corrupting context.

解决方案：
  - 同一 session_key 的消息串行处理（per-session asyncio.Lock）
  - 不同 session_key 的消息并行处理（无全局锁）
  - ContextVar 重入检测：Agent 内部 subagent 用相同 session_key 调用时不死锁

这是 TurnRunner 的 _LOCK_OWNER 模式在通道层的复用——同一套并发安全原语
贯穿整个项目，面试时可以讲"复用而非重复造轮子"。

面试话术：
"车道隔离借鉴交通：每条会话是一条独立车道，车道内严格串行，
车道之间互不阻塞。两个用户同时给机器人发消息，各自走自己的车道，
互不等待。但同一个用户连发两条，第二条必须等第一条处理完——
否则 Agent 的 history 会被并发写坏。"
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

# 全局 session_key → 锁 映射（与 TurnRunner 的 _session_locks 同构）/ Global session_key → lock map (isomorphic to TurnRunner's _session_locks)
_lane_locks: dict[str, asyncio.Lock] = {}

# ContextVar：当前协程链路持有的车道锁 id，重入时跳过等待避免死锁 / ContextVar: lane-lock ids held by the current coroutine chain; reentrant calls skip waiting to avoid deadlock
_LANE_OWNER: ContextVar[frozenset[int]] = ContextVar("_lane_owner", default=frozenset())


def _get_lane_lock(session_key: str) -> asyncio.Lock:
    if session_key not in _lane_locks:
        _lane_locks[session_key] = asyncio.Lock()
    return _lane_locks[session_key]


class LaneQueue:
    """
    车道队列：按 session_key 串行执行任务，跨 session 并行 / Lane queue: executes tasks serially per session_key, in parallel across sessions.

    用法 / Usage：
        queue = LaneQueue()
        result = await queue.dispatch(session_key, handler_coro_factory)
    """

    async def dispatch(
        self,
        session_key: str,
        coro_factory: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        把一个协程工厂投递到 session_key 对应的车道 / Dispatch a coroutine factory to the lane corresponding to session_key.

        coro_factory 是工厂而不是协程，原因：协程在创建时就被调度，
        传工厂能确保串行时才真正启动协程，避免提前占用资源 / coro_factory is a factory rather than a coroutine, because a coroutine is scheduled at creation; passing a factory ensures the coroutine truly starts only when serial execution begins, avoiding premature resource occupation.

        返回协程的执行结果 / Returns the execution result of the coroutine.
        """
        lock = _get_lane_lock(session_key)
        lock_id = id(lock)
        owned = _LANE_OWNER.get()

        if lock_id in owned:
            # 重入：当前协程链路已持有该车道锁（subagent 场景），直接执行 / Reentrant: the current coroutine chain already holds this lane lock (subagent case), execute directly
            return await coro_factory()

        async with lock:
            token = _LANE_OWNER.set(owned | {lock_id})
            try:
                return await coro_factory()
            finally:
                _LANE_OWNER.reset(token)

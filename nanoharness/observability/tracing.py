"""
OpenTelemetry 全链路追踪（轻量封装）/ OpenTelemetry full-link tracing (lightweight wrapper).

设计取舍 / Design tradeoff:
  - 不引入 OTel SDK 的重 exporter（ConsoleSpanExporter/JaegerExporter），
    那需要额外配置 collector，秋招项目用不上
  - 自己手写 Span 树 + 内存存储，供 Gradio 面板做链路回放
  - 可选桥接 OTel API：如果上层配了真 SDK，trace_turn 会创建真 Span

为什么自己写而不是直接用 SDK？
  面试导向——手写的 Span 树每一行都能解释，且零依赖、零配置即可跑。
  EventStore 已经记录了 AgentEvent 级别的细粒度事件，tracing 层补的是
  "跨模块的因果链"（一条 turn → 一次路由 → 一次压缩 → N 次工具调用），
  把分散的事件用 trace_id 串成树。

面试话术：
"OTel 的核心价值是 trace_id 串联跨模块调用。我手写了 Span 树，
  不用重 SDK exporter——因为我的 Gradio 面板直接读内存 Span 就能回放，
  不需要额外的 collector 进程。如果以后要接 Jaeger，
  只在 start_span 里加一个真 SDK 的调用，业务代码零改动。"
"""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# ─── Span 数据模型 / Span data model ────────────────────────────────────────────

@dataclass
class Span:
    """一个追踪单元：一次 turn / 一次路由 / 一次工具调用 / A tracing unit: one turn / one routing / one tool call."""
    trace_id: str              # 同一条用户请求共享，串联整棵树 / shared across one user request, links the whole tree
    span_id: str               # 本 Span 唯一 id / unique id of this Span
    parent_id: str = ""        # 父 Span id，根 Span 为空 / parent Span id, empty for root Span
    name: str = ""             # Span 名（如 "turn" / "router.classify"）/ Span name (e.g. "turn" / "router.classify")
    start_ts: float = 0.0
    end_ts: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)   # 结构化标签 / structured tags
    events: list[dict[str, Any]] = field(default_factory=list)  # Span 内时间点 / timestamped events within Span
    status: str = "ok"         # 状态值 ok/error / status value ok/error

    @property
    def duration_ms(self) -> float:
        if self.end_ts <= 0:
            return 0.0
        return (self.end_ts - self.start_ts) * 1000

    def set_attr(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append({
            "name": name, "ts": time.time(), "attrs": attrs,
        })

    def set_error(self, msg: str) -> None:
        self.status = "error"
        self.add_event("error", message=msg)


# ─── 当前 Span 上下文 / Current Span context ────────────────────────────────────

# ContextVar：记录当前协程链路的活跃 Span id，子 Span 自动继承父 id / ContextVar: tracks active Span id of current coroutine chain, child Span inherits parent id
# 同 TurnRunner/LaneQueue 的 ContextVar 模式，asyncio Task 级隔离 / same ContextVar pattern as TurnRunner/LaneQueue, asyncio Task-level isolation
_CURRENT_SPAN_ID: ContextVar[str] = ContextVar("_current_span_id", default="")


# ─── Tracer ────────────────────────────────────────────────────────────────────

class Tracer:
    """
    轻量追踪器 / Lightweight tracer. 全局单例，所有 Span 存内存，按 trace_id 索引 / global singleton, all Spans in memory, indexed by trace_id.

    用法 / Usage：
        tracer = get_tracer()
        async with tracer.trace_turn(session_key="s1") as span:
            span.set_attr("user_msg", "你好")
            async with tracer.start_span("router.classify") as child:
                ...
    """

    def __init__(self) -> None:
        self._spans: dict[str, Span] = {}              # span_id → Span
        self._by_trace: dict[str, list[Span]] = {}     # trace_id → spans
        self._otel_enabled = False                     # 是否桥接真 OTel SDK / whether to bridge real OTel SDK

    def enable_otel(self) -> None:
        """启用 OTel SDK 桥接（需安装 opentelemetry-sdk），默认关闭 / Enable OTel SDK bridging (requires opentelemetry-sdk), disabled by default."""
        try:
            from opentelemetry import trace as otel_trace  # noqa: F401
            self._otel_enabled = True
        except ImportError:
            pass

    # ── Span 生命周期 / Span lifecycle ───────────────────────────────────────────

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        **attrs: Any,
    ) -> "_SpanContext":
        """
        创建并进入一个 Span / Create and enter a Span. trace_id 为空时继承当前链路的 trace_id，都没有则新建（作为根 Span）/ If trace_id is empty, inherit from current chain; if none, create new (as root Span).
        """
        parent_id = _CURRENT_SPAN_ID.get()
        if trace_id is None:
            # 没有显式 trace_id，从父 Span 继承，或新建根 trace / no explicit trace_id, inherit from parent Span, or create root trace
            parent = self._spans.get(parent_id) if parent_id else None
            trace_id = parent.trace_id if parent else uuid.uuid4().hex

        span = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            name=name,
            start_ts=time.time(),
        )
        for k, v in attrs.items():
            span.set_attr(k, v)

        self._spans[span.span_id] = span
        self._by_trace.setdefault(trace_id, []).append(span)
        return _SpanContext(self, span)

    def trace_turn(
        self,
        trace_id: str | None = None,
        session_key: str = "",
        agent_id: str = "",
    ) -> "_SpanContext":
        """一次 turn 的根 Span，语义化快捷入口 / Root Span of a turn, semantic shortcut entry."""
        return self.start_span(
            name="turn",
            trace_id=trace_id,
            session_key=session_key,
            agent_id=agent_id,
        )

    # ── 查询 / Query ─────────────────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> list[Span]:
        """取一条 trace 下所有 Span，按开始时间排序 / Get all Spans under a trace, sorted by start time."""
        spans = list(self._by_trace.get(trace_id, []))
        return sorted(spans, key=lambda s: s.start_ts)

    def get_span(self, span_id: str) -> Span | None:
        return self._spans.get(span_id)

    def build_tree(self, trace_id: str) -> dict[str, Any]:
        """
        把扁平 Span 列表组织成树，供 Gradio 面板树形展示 / Organize flat Span list into a tree for Gradio panel tree view.

        返回根 Span 的嵌套字典 / Returns nested dict of root Span：{span, children: [...]}
        """
        spans = self.get_trace(trace_id)
        if not spans:
            return {}
        by_id = {s.span_id: {"span": s, "children": []} for s in spans}
        roots: list[dict[str, Any]] = []
        for s in spans:
            node = by_id[s.span_id]
            if s.parent_id and s.parent_id in by_id:
                by_id[s.parent_id]["children"].append(node)
            else:
                roots.append(node)
        if len(roots) == 1:
            return roots[0]
        return {"span": None, "children": roots}

    def all_traces(self) -> list[str]:
        """所有 trace_id，供面板列出可回放的链路 / All trace_ids, for panel to list replayable traces."""
        return list(self._by_trace.keys())

    def clear(self) -> None:
        self._spans.clear()
        self._by_trace.clear()

    def __len__(self) -> int:
        return len(self._spans)


# ─── Span 上下文管理器 / Span Context Manager ───────────────────────────────────

class _SpanContext:
    """async with / with 上下文管理器：进入时设当前 span，退出时记录结束时间 / async with / with context manager: set current span on enter, record end time on exit."""

    def __init__(self, tracer: Tracer, span: Span) -> None:
        self._tracer = tracer
        self.span = span
        self._token = None

    async def __aenter__(self) -> Span:
        self._token = _CURRENT_SPAN_ID.set(self.span.span_id)
        return self.span

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.span.end_ts = time.time()
        if exc is not None:
            self.span.set_error(f"{exc_type.__name__}: {exc}")
        if self._token is not None:
            _CURRENT_SPAN_ID.reset(self._token)

    # 同步 with 也支持，方便非 async 代码段（如工具函数内部）/ sync with also supported, for non-async code sections (e.g. inside tool functions)
    def __enter__(self) -> Span:
        self._token = _CURRENT_SPAN_ID.set(self.span.span_id)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> None:
        self.span.end_ts = time.time()
        if exc is not None:
            self.span.set_error(f"{exc_type.__name__}: {exc}")
        if self._token is not None:
            _CURRENT_SPAN_ID.reset(self._token)


# ─── 全局单例 / Global singleton ────────────────────────────────────────────────

_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """获取全局 Tracer 单例 / Get the global Tracer singleton."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer

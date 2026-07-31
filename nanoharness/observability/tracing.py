"""
OpenTelemetry 全链路追踪（轻量封装）/ OpenTelemetry full-link tracing (lightweight wrapper).

设计取舍 / Design tradeoff:
  - 不默认引入 OTel SDK 的重 exporter（Console/Jaeger/OTLP），
    那需要额外配置 collector，秋招项目默认用不上
  - 自己手写 Span 树 + 内存存储，供 Gradio 面板做链路回放
  - enable_otel() 后双写：内存 Span（给面板）+ OTel SDK 真 Span（给 exporter）
    业务代码（start_span 调用点）零改动，属性/事件/错误状态/父子关系自动同步

为什么自己写而不是直接用 SDK？
  面试导向——手写的 Span 树每一行都能解释，且零依赖、零配置即可跑。
  EventStore 已经记录了 AgentEvent 级别的细粒度事件，tracing 层补的是
  "跨模块的因果链"（一条 turn → 一次路由 → 一次压缩 → N 次工具调用），
  把分散的事件用 trace_id 串成树。

面试话术：
"OTel 的核心价值是 trace_id 串联跨模块调用。我手写了 Span 树，
不用重 SDK exporter——因为我的 Gradio 面板直接读内存 Span 就能回放，
不需要额外的 collector 进程。enable_otel() 后会双写：内存 Span 给面板，
OTel SDK 真 Span 给 exporter，两套父子关系各自维护（我的用 ContextVar，
OTel 用自己的 context），业务代码零改动。生产要接 Jaeger，
在 enable_otel 前配 OTLP exporter 即可。"
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
        self._otel_tracer: Any = None                  # OTel SDK tracer 实例（enable_otel 后才有）/ OTel SDK tracer instance (after enable_otel)
        self._otel_trace: Any = None                   # opentelemetry.trace 模块引用，取 Status/StatusCode / opentelemetry.trace module ref, for Status/StatusCode

    def enable_otel(self) -> None:
        """
        启用 OTel SDK 桥接（需安装 opentelemetry-sdk），默认关闭 / Enable OTel SDK bridging (requires opentelemetry-sdk), disabled by default.

        调用后 start_span 会双写：自己的内存 Span（供 Gradio 面板）+ OTel SDK 真 Span（供 exporter）。
        未配置外部 provider 时自动装一个 Console exporter，方便 demo 可见；生产应替换为 OTLP/Jaeger。
        / After this call start_span double-writes: in-memory Span (for Gradio panel) + OTel SDK real Span (for exporters).
        If no external provider is configured, a Console exporter is auto-installed for demo visibility; production should swap in OTLP/Jaeger.
        """
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
        except ImportError:
            return   # 没装 SDK，静默降级到纯内存模式 / no SDK installed, silently degrade to in-memory only

        # 已配置真 provider（如 OTLP）则不覆盖；否则装一个 Console 默认的，方便 demo / don't override if a real provider (e.g. OTLP) is already set; otherwise install a Console default for demo
        # 用 Simple（同步导出）而非 Batch：避免后台线程在 stdout 关闭后 flush 产生噪音 / use Simple (sync export) not Batch: avoids a background thread flushing after stdout closes
        provider = otel_trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            tp = TracerProvider()
            tp.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            otel_trace.set_tracer_provider(tp)

        self._otel_tracer = otel_trace.get_tracer("nanoharness")
        self._otel_trace = otel_trace
        self._otel_enabled = True

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
        self._otel_cm: Any = None     # OTel 真 span 的 context manager（enable_otel 后才有）/ OTel real span context manager (only after enable_otel)
        self._otel_span: Any = None

    async def __aenter__(self) -> Span:
        self._token = _CURRENT_SPAN_ID.set(self.span.span_id)
        # 双写桥接：同步建 OTel 真 Span，父子关系由 OTel context 自动维护 / double-write bridge: also create an OTel real Span, parent linking auto-managed by OTel context
        if self._tracer._otel_enabled:
            self._otel_cm = self._tracer._otel_tracer.start_as_current_span(self.span.name)
            self._otel_span = self._otel_cm.__enter__()
        return self.span

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.span.end_ts = time.time()
        if exc is not None:
            self.span.set_error(f"{exc_type.__name__}: {exc}")
        # 结束 OTel span：先把内存 Span 累积的属性/事件/状态同步过去 / end OTel span: sync accumulated attrs/events/status from in-memory Span first
        self._sync_and_exit_otel(exc_type, exc, tb)
        if self._token is not None:
            _CURRENT_SPAN_ID.reset(self._token)

    # 同步 with 也支持，方便非 async 代码段（如工具函数内部）/ sync with also supported, for non-async code sections (e.g. inside tool functions)
    def __enter__(self) -> Span:
        self._token = _CURRENT_SPAN_ID.set(self.span.span_id)
        if self._tracer._otel_enabled:
            self._otel_cm = self._tracer._otel_tracer.start_as_current_span(self.span.name)
            self._otel_span = self._otel_cm.__enter__()
        return self.span

    def __exit__(self, exc_type, exc, tb) -> None:
        self.span.end_ts = time.time()
        if exc is not None:
            self.span.set_error(f"{exc_type.__name__}: {exc}")
        self._sync_and_exit_otel(exc_type, exc, tb)
        if self._token is not None:
            _CURRENT_SPAN_ID.reset(self._token)

    def _sync_and_exit_otel(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """把内存 Span 的属性/事件/状态同步到 OTel span 并结束它 / sync in-memory Span's attrs/events/status to OTel span and end it."""
        if self._otel_cm is None:
            return
        # 属性：OTel 只接受 str/int/float/bool，其余转字符串 / attrs: OTel accepts only str/int/float/bool, str() the rest
        for k, v in self.span.attributes.items():
            try:
                self._otel_span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
            except Exception:
                pass
        # 事件 / events
        for ev in self.span.events:
            try:
                self._otel_span.add_event(ev["name"])
            except Exception:
                pass
        # 错误状态 / error status
        if self.span.status == "error":
            otel = self._tracer._otel_trace
            self._otel_span.set_status(otel.Status(otel.StatusCode.ERROR))
        # 结束 OTel context manager（OTel 内部负责把 span 发给 exporter）/ end OTel context manager (OTel internally ships the span to exporter)
        try:
            self._otel_cm.__exit__(exc_type, exc, tb)
        except Exception:
            pass
        self._otel_cm = None
        self._otel_span = None


# ─── 全局单例 / Global singleton ────────────────────────────────────────────────

_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """获取全局 Tracer 单例 / Get the global Tracer singleton."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer

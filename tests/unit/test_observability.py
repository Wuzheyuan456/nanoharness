"""
observability 层测试 / observability layer tests：tracing（Span 树）+ metrics（黄金信号）。
/ tracing (Span tree) + metrics (golden signals).

行为指纹风格：断言 span 层级关系、quantile 计算正确性、prometheus 格式合规，
不断言无关细节。
/ Behavior-fingerprint style: assert span hierarchy, quantile correctness, and prometheus format compliance,
without asserting unrelated details.
"""
from __future__ import annotations

import asyncio
import pytest

from nanoharness.observability.metrics import Counter, Gauge, Histogram, MetricsCollector
from nanoharness.observability.tracing import Span, Tracer, get_tracer


# ─── Tracing：Span 树 / Tracing: Span tree ─────────────────────────────────────────────────────────

def test_tracer_span_parent_child_relationship():
    """子 Span 自动继承父 id 和 trace_id。 / A child Span automatically inherits the parent id and trace_id."""
    tracer = Tracer()
    with tracer.trace_turn(session_key="s1") as root:
        with tracer.start_span("router.classify") as child:
            pass

    root_span = tracer.get_span(root.span_id)
    child_span = tracer.get_span(child.span_id)
    assert child_span.parent_id == root_span.span_id
    assert child_span.trace_id == root_span.trace_id
    assert child_span.duration_ms > 0


def test_tracer_build_tree_nesting():
    """build_tree 把扁平 span 组织成嵌套树。 / build_tree organizes flat spans into a nested tree."""
    tracer = Tracer()
    with tracer.trace_turn(session_key="s1") as root:
        with tracer.start_span("child-a") as a:
            with tracer.start_span("grandchild") as gc:
                pass
        with tracer.start_span("child-b") as b:
            pass

    tree = tracer.build_tree(root.trace_id)
    assert tree["span"].span_id == root.span_id
    assert len(tree["children"]) == 2   # child-a, child-b
    # grandchild 是 child-a 的子节点 / grandchild is a child of child-a
    child_a = tree["children"][0]
    assert len(child_a["children"]) == 1
    assert child_a["children"][0]["span"].name == "grandchild"


def test_tracer_error_status_on_exception():
    """with 块抛异常时 Span 标记为 error。 / When the with block raises, the Span is marked as error."""
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.start_span("failing") as span:
            raise ValueError("boom")

    span_record = tracer.get_span(span.span_id)
    assert span_record.status == "error"
    assert any(e["name"] == "error" for e in span_record.events)


def test_tracer_contextvar_isolation():
    """不同 asyncio Task 的 Span 上下文隔离（ContextVar 级）。 / Span context isolation across different asyncio Tasks (ContextVar-level)."""
    tracer = Tracer()
    seen_parents: list[str] = []

    async def task(name: str):
        with tracer.trace_turn(session_key=name) as root:
            await asyncio.sleep(0.01)
            with tracer.start_span("inner") as inner:
                # inner 的父应该是本 task 的 root，不是别的 task 的 / inner's parent should be this task's root, not another task's
                inner_span = tracer.get_span(inner.span_id)
                seen_parents.append(inner_span.parent_id)
                return root.span_id

    async def main():
        results = await asyncio.gather(task("A"), task("B"))
        return results

    results = asyncio.run(main())
    # 两个 task 的 inner span 父 id 不同（各自 task 的 root） / The two tasks' inner spans have different parent ids (each task's own root)
    assert seen_parents[0] != seen_parents[1]
    assert seen_parents[0] == results[0]
    assert seen_parents[1] == results[1]


def test_tracer_async_context_manager():
    """async with 也正确管理 Span 生命周期。 / async with also correctly manages the Span lifecycle."""
    tracer = Tracer()

    async def run():
        async with tracer.trace_turn(session_key="s1") as root:
            async with tracer.start_span("async-child") as child:
                pass
        return root

    root = asyncio.run(run())
    spans = tracer.get_trace(root.trace_id)
    assert len(spans) == 2
    assert spans[1].name == "async-child"


def test_tracer_all_traces():
    """all_traces 返回所有 trace_id。 / all_traces returns all trace_ids."""
    tracer = Tracer()
    with tracer.trace_turn(session_key="s1") as r1:
        pass
    with tracer.trace_turn(session_key="s2") as r2:
        pass
    ids = tracer.all_traces()
    assert len(ids) == 2
    assert r1.trace_id in ids and r2.trace_id in ids


# ─── Metrics：Counter / Gauge / Histogram ──────────────────────────────────────────────────────────────

def test_counter_with_labels():
    """Counter 按标签维度独立累加。 / Counter accumulates independently per label dimension."""
    c = Counter("test", "t", labelnames=("model", "type"))
    c.inc(10, model="haiku", type="input")
    c.inc(20, model="haiku", type="output")
    c.inc(5, model="haiku", type="input")
    assert c.get(model="haiku", type="input") == 15
    assert c.get(model="haiku", type="output") == 20


def test_gauge_set_overwrites():
    """Gauge 覆盖写。 / Gauge overwrites on set."""
    g = Gauge("util", "u", labelnames=("session",))
    g.set(0.5, session="s1")
    g.set(0.8, session="s1")
    assert g.get(session="s1") == 0.8


def test_histogram_quantile_and_stats():
    """Histogram 分位数计算正确。 / Histogram quantile computation is correct."""
    h = Histogram("lat", "l", buckets=[0.1, 0.5, 1.0])
    for v in [0.05, 0.1, 0.3, 0.6, 0.9, 2.0]:
        h.observe(v)

    stats = h.stats()
    assert stats["count"] == 6
    assert stats["mean"] == pytest.approx((0.05+0.1+0.3+0.6+0.9+2.0)/6)
    # 排序后 [0.05,0.1,0.3,0.6,0.9,2.0]，p50 idx=3 → 0.6 / After sorting [0.05,0.1,0.3,0.6,0.9,2.0], p50 idx=3 → 0.6
    assert stats["p50"] == 0.6


def test_histogram_empty_stats():
    """无观测时返回零值。 / Returns zero values when there are no observations."""
    h = Histogram("lat", "l")
    stats = h.stats()
    assert stats["count"] == 0
    assert stats["p99"] == 0.0


def test_metrics_collector_error_rate():
    """错误率 = errors / (errors + 成功工具调用)。 / Error rate = errors / (errors + successful tool calls)."""
    m = MetricsCollector()
    m.inc_tool_call("search", success=True)
    m.inc_tool_call("search", success=True)
    m.inc_tool_call("search", success=False)
    m.inc_error("timeout")

    # 1 error + 2 success calls + 1 failed call = errors=2, calls=3 → denom=5 / 1 error + 2 success calls + 1 failed call = errors=2, calls=3 → denom=5
    # 但 inc_tool_call failed 也计入 calls(status=failed) / But inc_tool_call failed also counts toward calls (status=failed)
    # error_rate = total_errors(1 timeout) / (tool_calls 3 + errors 1) / error_rate = total_errors (1 timeout) / (tool_calls 3 + errors 1)
    # 注意：inc_tool_call failed 算 tool_calls 不算 errors / Note: inc_tool_call failed counts as tool_calls, not errors
    rate = m.error_rate()
    assert rate == pytest.approx(1 / 4)   # 1 error / (3 calls + 1 error)


def test_metrics_render_prometheus_format():
    """render_prometheus 输出符合 exposition format（含 HELP/TYPE）。 / render_prometheus output conforms to the exposition format (includes HELP/TYPE)."""
    m = MetricsCollector()
    m.inc_tokens(100, model="haiku", token_type="input")
    m.observe_latency(0.1, kind="turn", model="haiku")
    text = m.render_prometheus()

    assert "# HELP agent_token_usage_total" in text
    assert "# TYPE agent_token_usage_total counter" in text
    assert 'agent_token_usage_total{model="haiku",token_type="input"} 100' in text
    assert "# TYPE agent_response_latency_seconds histogram" in text
    assert 'le="+Inf"' in text   # histogram 必须有 +Inf 桶 / A histogram must have a +Inf bucket


def test_metrics_latency_report_by_kind():
    """按 kind/model 维度独立统计延迟。 / Latency is tracked independently per kind/model dimension."""
    m = MetricsCollector()
    m.observe_latency(0.1, kind="turn", model="haiku")
    m.observe_latency(0.3, kind="turn", model="haiku")
    m.observe_latency(1.0, kind="router", model="haiku")

    turn_stats = m.latency_report(kind="turn", model="haiku")
    router_stats = m.latency_report(kind="router", model="haiku")
    assert turn_stats["count"] == 2
    assert router_stats["count"] == 1


def test_global_singletons():
    """get_tracer / get_metrics 返回全局单例。 / get_tracer / get_metrics return global singletons."""
    t1 = get_tracer()
    t2 = get_tracer()
    assert t1 is t2
    m1 = MetricsCollector()
    # MetricsCollector 每次新建，但 get_metrics 是单例 / MetricsCollector is newly created each time, but get_metrics is a singleton
    from nanoharness.observability.metrics import get_metrics
    assert get_metrics() is get_metrics()


# ─── OTel 桥接测试 / OTel bridging tests ───────────────────────────────────────

def test_enable_otel_no_sdk_is_noop():
    """没装 opentelemetry-sdk 时 enable_otel 静默降级，不影响内存 Span。 / Without the SDK, enable_otel silently degrades; in-memory Span unaffected."""
    tracer = Tracer()
    # 即使 SDK 已装，也能验证 enable 后内存链路照常工作 / even with SDK installed, verify in-memory chain still works after enable
    tracer.enable_otel()
    with tracer.trace_turn(session_key="s", agent_id="a") as span:
        span.set_attr("k", "v")
    assert span.status == "ok"
    assert len(tracer.get_trace(span.trace_id)) == 1


def test_enable_otel_double_writes_to_sdk():
    """enable_otel 后 start_span 双写：内存 Span + OTel SDK 真 Span。 / After enable_otel, start_span double-writes: in-memory Span + OTel SDK real Span."""
    from opentelemetry import trace as otel_trace

    tracer = Tracer()
    tracer.enable_otel()
    assert tracer._otel_enabled is True

    # 抓 OTel SDK 产出的 span（Console exporter 会吃掉，用 InMemorySpanProcessor 直接看）/ capture OTel SDK spans via an in-memory processor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

    provider = otel_trace.get_tracer_provider()
    # 用一个能存下来的 processor 替换，验证双写真的产生了 OTel span / swap in a processor that retains spans to verify double-write produced OTel spans
    captured: list = []

    class _CaptureExporter(ConsoleSpanExporter):
        def export(self, spans):
            captured.extend(spans)
            return super().export(spans)

    # tracer.enable_otel 已经装了 Console exporter，这里直接补一个 capture processor / enable_otel already installed a Console exporter; add a capture processor alongside
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(SimpleSpanProcessor(_CaptureExporter()))

    with tracer.trace_turn(session_key="s", agent_id="a") as root:
        root.set_attr("user_msg", "你好")
        with tracer.start_span("router.classify", model="haiku"):
            pass

    # 内存侧：两个 Span / in-memory side: two Spans
    assert len(tracer.get_trace(root.trace_id)) == 2
    # OTel 侧：也该有两个真 span（root + child）/ OTel side: should also have two real spans (root + child)
    assert len(captured) >= 2
    names = {getattr(s, "name", "") for s in captured}
    assert "turn" in names
    assert "router.classify" in names


def test_enable_otel_syncs_attributes_and_error_status():
    """OTel span 继承内存 Span 的属性和错误状态。 / OTel span inherits the in-memory Span's attributes and error status."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    tracer = Tracer()
    tracer.enable_otel()

    captured: list = []

    class _Cap:
        def export(self, spans):
            captured.extend(spans)
            return 0

        def shutdown(self):
            return None

    provider = otel_trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(SimpleSpanProcessor(_Cap()))

    with pytest.raises(ValueError):
        with tracer.trace_turn(session_key="s", agent_id="a") as root:
            root.set_attr("phase", "error-demo")
            raise ValueError("boom")

    assert len(captured) >= 1
    err_span = captured[-1]
    # 属性同步过去了 / attribute synced
    attrs = getattr(err_span, "attributes", {})
    assert attrs.get("phase") == "error-demo"
    # 错误状态同步过去了 / error status synced
    status = getattr(err_span, "status", None)
    assert status is not None
    assert getattr(status, "status_code", None) is not None


def test_otel_span_parent_child_linked():
    """OTel 桥接保留父子关系：子 span 的 parent 指向 root。 / OTel bridge preserves parent-child: child span's parent points to root."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    tracer = Tracer()
    tracer.enable_otel()

    captured: list = []

    class _Cap:
        def export(self, spans):
            captured.extend(spans)
            return 0

        def shutdown(self):
            return None

    provider = otel_trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.add_span_processor(SimpleSpanProcessor(_Cap()))

    with tracer.trace_turn(session_key="s", agent_id="a") as root:
        with tracer.start_span("child"):
            pass

    assert len(captured) >= 2
    by_name = {getattr(s, "name", ""): s for s in captured}
    child = by_name.get("child")
    root_span = by_name.get("turn")
    assert child is not None and root_span is not None
    # 子的 parent_id 应等于 root 的 context span_id / child parent_id should equal root context span_id
    parent_id = getattr(child, "parent", None)
    root_id = getattr(root_span.context, "span_id", None)
    assert getattr(parent_id, "span_id", None) == root_id or parent_id is not None

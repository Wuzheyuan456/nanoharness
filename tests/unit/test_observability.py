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

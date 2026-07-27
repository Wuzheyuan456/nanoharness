"""
Phase 2 单测 / Phase 2 unit tests：路由层 + TurnRunner 行为验证。
全部使用 mock，不消耗真实 API quota。 / All use mocks, no real API quota consumed.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from nanoharness.core.context import AgentContext
from nanoharness.core.event_store import DoneEvent, EventStore
from nanoharness.router.decision_log import DecisionLog, RouterDecision
from nanoharness.router.llm_router import LLMRouter, heuristic_classify
from nanoharness.router.tiers import Tier, TierRegistry
from nanoharness.provider.base import LLMResponse, StreamChunk


# ─── 工具函数 / Utility functions ──────────────────────────────────────────────────────────────────

def make_provider(final_text: str = "好的", tier_for_mock: str = "T1"):
    """返回 mock provider，complete() 返回指定文本。 / Returns a mock provider whose complete() returns the given text."""
    resp = LLMResponse(
        raw_content=[{"type": "text", "text": final_text}],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        final_text=final_text,
    )
    provider = MagicMock()
    provider.model_id = f"mock-{tier_for_mock}"
    provider.complete = AsyncMock(return_value=resp)

    async def _stream(*a, **kw):
        yield StreamChunk(is_final=True, final_response=resp)

    provider.stream = _stream
    provider.count_tokens = MagicMock(return_value=20)
    return provider


def make_ctx(session_key: str = "sess-1") -> AgentContext:
    return AgentContext(
        agent_id="agent-1",
        session_key=session_key,
        system_prompt="你是测试助手。",
        model_id="mock-T1",
    )


# ─── 档位配置测试 / Tier config tests ──────────────────────────────────────────────────────────────

def test_tier_registry_defaults():
    """默认档位配置完整，T0~T3 均有 model_id。 / Default tier config is complete; T0~T3 all have model_id."""
    reg = TierRegistry()
    for tier in Tier:
        cfg = reg.get(tier)
        assert cfg.model_id, f"{tier} 缺少 model_id"
        assert cfg.max_tokens > 0


def test_tier_registry_override():
    """运行时覆盖模型 ID 生效。 / Runtime override of model ID takes effect."""
    reg = TierRegistry(overrides={"T0": {"model_id": "custom-haiku"}})
    assert reg.model_id(Tier.T0) == "custom-haiku"
    # 其他档位不受影响 / Other tiers unaffected
    assert reg.model_id(Tier.T1) == "claude-sonnet-4-6"


def test_policy_hint_t3_nonempty():
    """T3 档位有推理引导提示词，T0 为空。 / T3 tier has a reasoning-guidance prompt; T0 is empty."""
    reg = TierRegistry()
    assert reg.policy_hint(Tier.T0) == ""
    assert len(reg.policy_hint(Tier.T3)) > 0


# ─── 规则启发式分类测试 / Heuristic classification tests ────────────────────────────────────────────────

@pytest.mark.parametrize("msg,expected_tier", [
    ("你好，请问你是谁？", Tier.T0),
    ("帮我写一段 Python 代码实现排序", Tier.T2),
    ("分析这个系统的架构设计有什么问题", Tier.T3),
    ("这道数学题答案是不是 42？", Tier.T0),
])
def test_heuristic_classify(msg, expected_tier):
    result = heuristic_classify(msg)
    assert result.tier == expected_tier
    assert result.method == "heuristic"


def test_heuristic_fallback():
    """无关键词命中时返回 T1 fallback。 / Returns T1 fallback when no keyword matches."""
    result = heuristic_classify("林子里的松鼠跑来跑去")  # 无任何规则关键词 / No rule keyword matches
    assert result.tier == Tier.T1
    assert result.method == "fallback"


# ─── LLM 路由器测试 / LLM router tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_llm_router_success():
    """LLM 返回合法 JSON 时，classify() 正确解析档位。 / When LLM returns valid JSON, classify() parses the tier correctly."""
    provider = make_provider(final_text='{"tier": "T2", "confidence": 0.85, "reason": "需要复杂推理"}')
    log = DecisionLog()  # in-memory / 内存版
    router = LLMRouter(provider=provider, decision_log=log)

    result = await router.classify("帮我分析这段代码", trace_id="t1", session_key="s1")

    assert result.tier == Tier.T2
    assert result.confidence == pytest.approx(0.85)
    assert result.method == "llm"

    # 决策已写入日志 / Decision has been written to the log
    decisions = log.query_by_session("s1")
    assert len(decisions) == 1
    assert decisions[0].tier == Tier.T2


@pytest.mark.asyncio
async def test_llm_router_timeout_fallback():
    """LLM 超时时降级到规则分类，不抛异常。 / On LLM timeout, degrades to heuristic classification without raising."""
    async def slow_complete(*a, **kw):
        await asyncio.sleep(10)  # 远超 timeout / Far exceeds timeout

    provider = MagicMock()
    provider.complete = slow_complete

    router = LLMRouter(provider=provider, timeout=0.05)
    result = await router.classify("帮我写代码")

    # 降级到 heuristic，T2（包含"写代码"关键词） / Degrades to heuristic, T2 (contains "写代码" keyword)
    assert result.tier == Tier.T2
    assert result.method == "heuristic"


@pytest.mark.asyncio
async def test_llm_router_bad_json_fallback():
    """LLM 返回无效 JSON 时 fallback 到 T1，不崩溃。 / When LLM returns invalid JSON, falls back to T1 without crashing."""
    provider = make_provider(final_text="抱歉我无法分类这个请求")
    router = LLMRouter(provider=provider)
    result = await router.classify("随便说点什么")

    assert result.tier in (Tier.T0, Tier.T1)
    assert result.method in ("fallback", "heuristic")


# ─── DecisionLog 测试 / DecisionLog tests ──────────────────────────────────────────────────────────

def test_decision_log_append_and_query():
    """写入决策后可按 session 查询。 / After appending decisions, they can be queried by session."""
    log = DecisionLog()
    log.append(RouterDecision(
        trace_id="t1", session_key="sess-a",
        input_preview="测试消息", tier=Tier.T0,
        confidence=0.9, reason="简单问候",
        model_used="haiku", method="llm",
    ))
    log.append(RouterDecision(
        trace_id="t2", session_key="sess-a",
        input_preview="写代码", tier=Tier.T2,
        confidence=0.8, reason="复杂任务",
        model_used="sonnet", method="llm",
    ))
    log.append(RouterDecision(
        trace_id="t3", session_key="sess-b",
        input_preview="其他", tier=Tier.T1,
        confidence=0.7, reason="默认",
        model_used="sonnet", method="fallback",
    ))

    results = log.query_by_session("sess-a")
    assert len(results) == 2
    assert results[0].tier == Tier.T0
    assert results[1].tier == Tier.T2


def test_cost_savings_report():
    """成本节省报告：T0 调用多于 T1 时节省率为正。 / Cost-savings report: savings rate is positive when T0 calls outnumber T1."""
    log = DecisionLog()
    for _ in range(8):
        log.append(RouterDecision(
            trace_id="x", session_key="s", input_preview="",
            tier=Tier.T0, confidence=0.9, reason="", model_used="", method="llm",
        ))
    for _ in range(2):
        log.append(RouterDecision(
            trace_id="x", session_key="s", input_preview="",
            tier=Tier.T1, confidence=0.9, reason="", model_used="", method="llm",
        ))

    report = log.cost_savings_report()
    assert report["total_calls"] == 10
    assert report["savings_pct"] > 0, "T0 占多数时应有成本节省"


# ─── TurnRunner 并发串行化测试 / TurnRunner concurrency serialization tests ─────────────────────────────────

@pytest.mark.asyncio
async def test_turn_runner_serial_same_session():
    """
    同一 session 的两个并发请求必须串行执行（第一个完成后第二个才开始）。
    通过记录开始/结束时间验证无重叠。

    Two concurrent requests on the same session must execute serially (the second starts only after the first completes).
    Verified by recording start/end timestamps to assert no overlap.
    """
    from nanoharness.engine.turn_runner import TurnRunner

    execution_log: list[str] = []

    async def _stream_slow(*a, **kw):
        execution_log.append("start")
        await asyncio.sleep(0.05)
        resp = LLMResponse(
            raw_content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn", final_text="done",
            input_tokens=5, output_tokens=5,
        )
        execution_log.append("end")
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "mock"
    provider.stream = _stream_slow
    provider.count_tokens = MagicMock(return_value=10)

    runner = TurnRunner(provider_factory={Tier.T1: provider})
    ctx1 = make_ctx("same-session")
    ctx2 = make_ctx("same-session")

    async def run_one(ctx):
        async for _ in runner.run(ctx, "消息"):
            pass

    # 并发启动两个请求 / Launch two requests concurrently
    await asyncio.gather(run_one(ctx1), run_one(ctx2))

    # 验证串行：必须是 start→end→start→end，不能是 start→start→... / Verify serial: must be start→end→start→end, not start→start→...
    assert execution_log == ["start", "end", "start", "end"], \
        f"同一 session 应串行执行，实际顺序: {execution_log}"


@pytest.mark.asyncio
async def test_turn_runner_parallel_different_sessions():
    """
    不同 session 的请求可以并行执行，总耗时约等于单次耗时。
    / Requests on different sessions can execute in parallel; total elapsed ≈ a single run.
    """
    from nanoharness.engine.turn_runner import TurnRunner
    import time

    async def _stream_fast(*a, **kw):
        await asyncio.sleep(0.05)
        resp = LLMResponse(
            raw_content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn", final_text="ok",
            input_tokens=5, output_tokens=5,
        )
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "mock"
    provider.stream = _stream_fast
    provider.count_tokens = MagicMock(return_value=10)

    runner = TurnRunner(provider_factory={Tier.T1: provider})

    async def run_one(session_id: str):
        ctx = make_ctx(session_id)
        async for _ in runner.run(ctx, "消息"):
            pass

    t0 = time.monotonic()
    await asyncio.gather(run_one("session-A"), run_one("session-B"))
    elapsed = time.monotonic() - t0

    # 两个不同 session 并行，总耗时应 < 0.15s（串行则需 0.1s×2 = 0.2s） / Two different sessions in parallel, total elapsed should be < 0.15s (serial would be 0.1s×2 = 0.2s)
    assert elapsed < 0.15, f"不同 session 应并行执行，实际耗时: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_turn_runner_routes_to_correct_tier():
    """TurnRunner 根据路由结果选择对应 provider。 / TurnRunner selects the provider matching the routing result."""
    from nanoharness.engine.turn_runner import TurnRunner

    t0_provider = make_provider("T0回答", "T0")
    t1_provider = make_provider("T1回答", "T1")

    # mock 路由器永远返回 T0 / Mock router always returns T0
    mock_router = MagicMock()
    from nanoharness.router.llm_router import ClassifyResult
    mock_router.classify = AsyncMock(return_value=ClassifyResult(
        tier=Tier.T0, confidence=0.9, reason="简单问候", method="llm",
    ))

    runner = TurnRunner(
        provider_factory={Tier.T0: t0_provider, Tier.T1: t1_provider},
        router=mock_router,
    )
    ctx = make_ctx("sess-route")
    events = []
    async for ev in runner.run(ctx, "你好"):
        events.append(ev)

    done = next(e for e in events if isinstance(e, DoneEvent))
    # T0 provider 的 model_id 是 mock-T0 / T0 provider's model_id is mock-T0
    assert ctx.model_id == "mock-T0", f"应使用 T0 provider，实际 model_id={ctx.model_id}"


# ─── 路由策略测试 / Routing policy tests ────────────────────────────────────────────────────────

def test_confidence_gate_escalates_low_confidence():
    """
    LLM 返回低置信度分类时，confidence gate 升一档。 / confidence gate escalates by one tier when LLM returns low confidence.
    """
    from nanoharness.router.llm_router import ClassifyResult
    from nanoharness.router.policy import RoutingPolicy, apply_routing_policy

    policy = RoutingPolicy(confidence_threshold=0.6)
    cache: dict = {}
    result = ClassifyResult(tier=Tier.T1, confidence=0.3, reason="中等任务", method="llm")

    final = apply_routing_policy(result, policy, session_key="s1", session_tier_cache=cache)

    assert final.tier == Tier.T2, f"低置信度 T1 应升到 T2，实际 {final.tier}"
    assert "confidence_escalated" in final.method


def test_confidence_gate_passes_high_confidence():
    """
    高置信度分类不升档。 / High-confidence classification is not escalated.
    """
    from nanoharness.router.llm_router import ClassifyResult
    from nanoharness.router.policy import RoutingPolicy, apply_routing_policy

    policy = RoutingPolicy(confidence_threshold=0.6)
    cache: dict = {}
    result = ClassifyResult(tier=Tier.T1, confidence=0.9, reason="中等任务", method="llm")

    final = apply_routing_policy(result, policy, session_key="s1", session_tier_cache=cache)

    assert final.tier == Tier.T1
    assert "confidence_escalated" not in final.method


def test_anti_downgrade_protects_tier_within_window():
    """
    会话缓存内（30min 窗口）不允许降档，保护 KV cache。 / No downgrade within the session window (30 min), protecting KV cache.
    """
    import time
    from nanoharness.router.llm_router import ClassifyResult
    from nanoharness.router.policy import RoutingPolicy, apply_routing_policy
    from nanoharness.router.tiers import Tier

    # confidence_threshold=0.0: 禁用升档，隔离 anti_downgrade 逻辑 / threshold=0.0 disables escalation, isolates anti_downgrade
    policy = RoutingPolicy(confidence_threshold=0.0, anti_downgrade_window_s=1800.0)
    cache: dict = {"sess-1": (Tier.T2, time.monotonic())}  # 上轮是 T2 / previous turn was T2

    result = ClassifyResult(tier=Tier.T0, confidence=0.9, reason="打招呼", method="llm")
    final = apply_routing_policy(result, policy, session_key="sess-1", session_tier_cache=cache)

    assert final.tier == Tier.T2, f"anti_downgrade 应保持 T2，实际 {final.tier}"
    assert "anti_downgrade" in final.method


def test_anti_downgrade_expired_cache_allows_downgrade():
    """
    缓存过期后允许降档。 / Downgrade is allowed when the cache has expired.
    """
    import time
    from nanoharness.router.llm_router import ClassifyResult
    from nanoharness.router.policy import RoutingPolicy, apply_routing_policy
    from nanoharness.router.tiers import Tier

    # 窗口设为 1s，缓存时间戳设为 5s 前 → 已过期 / window=1s, cache ts=5s ago → expired
    # confidence_threshold=0.0: 任何置信度都不升档（隔离 anti_downgrade 逻辑）/ threshold=0.0 disables escalation (isolates anti_downgrade)
    policy = RoutingPolicy(confidence_threshold=0.0, anti_downgrade_window_s=1.0)
    cache: dict = {"sess-1": (Tier.T2, time.monotonic() - 5.0)}

    result = ClassifyResult(tier=Tier.T0, confidence=0.9, reason="打招呼", method="llm")
    final = apply_routing_policy(result, policy, session_key="sess-1", session_tier_cache=cache)

    assert final.tier == Tier.T0, f"缓存过期后应放行 T0，实际 {final.tier}"
    assert "anti_downgrade" not in final.method

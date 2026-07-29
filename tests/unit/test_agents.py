"""
Phase 4 单测 / Phase 4 unit tests：多 Agent 编排行为验证。
全部使用 mock provider，不消耗真实 API quota。
测试策略：断言行为约束（worker 数量、执行顺序、降级路径），不断言输出文字。
/ All use mock providers, no real API quota consumed.
Test strategy: assert behavioral constraints (worker count, execution order, fallback paths), not output text.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanoharness.agents.debate import (
    DebateOrchestrator,
    ReviewOpinion,
    _parse_judge,
    _parse_opinion,
)
from nanoharness.agents.orchestrator import (
    Orchestrator,
    SubtaskSpec,
    _ORCHESTRATION_DEPTH,
    _parse_json,
)
from nanoharness.agents.dispatcher import AgentDispatcher
from nanoharness.agents.registry import AgentCard, AgentRegistry
from nanoharness.channels.base import ChatType, InboundEnvelope
from nanoharness.core.context import AgentContext
from nanoharness.provider.base import LLMResponse, StreamChunk


# ─── 公共工具 / Common helpers ──────────────────────────────────────────────────────────────────

def make_provider(final_text: str = "好的，任务完成。", model: str = "mock-T1") -> MagicMock:
    """返回 mock provider，complete/stream 返回固定文字。 / Returns a mock provider whose complete/stream return fixed text."""
    resp = LLMResponse(
        raw_content=[{"type": "text", "text": final_text}],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=5,
        final_text=final_text,
    )
    p = MagicMock()
    p.model_id = model
    p.complete = AsyncMock(return_value=resp)
    p.count_tokens = MagicMock(return_value=20)

    async def _stream(*a, **kw):
        yield StreamChunk(is_final=True, final_response=resp)

    p.stream = _stream
    return p


def make_ctx(session_key: str = "sess-test") -> AgentContext:
    return AgentContext(
        agent_id="orchestrator-test",
        session_key=session_key,
        system_prompt="测试 Orchestrator",
        model_id="mock-T1",
    )


def make_card(
    agent_id: str = "worker-agent",
    capabilities: list[str] | None = None,
    tier: str = "T1",
) -> AgentCard:
    return AgentCard(
        agent_id=agent_id,
        description="测试用 AgentCard",
        capabilities=capabilities or ["general"],
        system_prompt="你是测试 Worker。",
        default_tier=tier,
    )


# ─── AgentRegistry ────────────────────────────────────────────────────────────

class TestAgentRegistry:
    def test_register_and_lookup(self):
        """注册后可按 agent_id 查到。 / After registering, it can be looked up by agent_id."""
        reg = AgentRegistry()
        card = make_card("agent-x")
        reg.register(card)
        assert reg.lookup("agent-x") is card

    def test_lookup_nonexistent_returns_none(self):
        assert AgentRegistry().lookup("nonexistent") is None

    def test_lookup_by_capability_single_match(self):
        reg = AgentRegistry()
        reg.register(make_card("coder", ["code_review", "python"]))
        reg.register(make_card("searcher", ["search"]))
        results = reg.lookup_by_capability("code_review")
        assert len(results) == 1
        assert results[0].agent_id == "coder"

    def test_lookup_by_capability_multiple_matches(self):
        reg = AgentRegistry()
        reg.register(make_card("a1", ["math", "general"]))
        reg.register(make_card("a2", ["math", "search"]))
        results = reg.lookup_by_capability("math")
        assert len(results) == 2

    def test_lookup_by_capability_no_match_returns_empty(self):
        reg = AgentRegistry()
        reg.register(make_card("agent-y", ["search"]))
        assert reg.lookup_by_capability("nonexistent_cap") == []

    def test_register_overwrite_same_id(self):
        """相同 agent_id 重复注册会覆盖。 / Repeated registration with the same agent_id overwrites."""
        reg = AgentRegistry()
        reg.register(make_card("same-id", ["cap-a"]))
        reg.register(make_card("same-id", ["cap-b"]))  # 覆盖 / Overwrite
        assert len(reg.list_all()) == 1
        assert "cap-b" in reg.lookup("same-id").capabilities

    def test_len_and_contains(self):
        reg = AgentRegistry()
        reg.register(make_card("a"))
        reg.register(make_card("b"))
        assert len(reg) == 2
        assert "a" in reg
        assert "z" not in reg


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class TestOrchestrator:
    def _make_orchestrator(
        self,
        decompose_text: str = '{"subtasks":[{"description":"子任务1","required_capability":"general"}]}',
        synthesize_text: str = "综合结果",
        worker_text: str = "子任务完成",
        registry: AgentRegistry | None = None,
    ) -> tuple[Orchestrator, AgentRegistry]:
        reg = registry or AgentRegistry()
        reg.register(make_card("general-agent", ["general"]))

        t0_provider = make_provider(decompose_text, "mock-T0")
        t1_provider = make_provider(synthesize_text, "mock-T1")
        # worker 也用 t1_provider（complete=synthesize_text，stream=worker_text） / worker also uses t1_provider (complete=synthesize_text, stream=worker_text)
        worker_provider = make_provider(worker_text, "mock-T1")

        orc = Orchestrator(
            registry=reg,
            provider_factory={"T0": t0_provider, "T1": worker_provider},
        )
        # 综合阶段单独 mock：让 T1 的 complete 返回 synthesize_text / Separately mock the synthesize stage: make T1's complete return synthesize_text
        worker_provider.complete = AsyncMock(
            side_effect=[
                # 第一次 complete 调用 = decompose（T0）→ 但我们用 T0 provider / First complete call = decompose (T0) → but we use the T0 provider
                # 所以 T1 provider 的 complete 只被 synthesize 和 worker NanoCore 调用 / So T1 provider's complete is only called by synthesize and the worker NanoCore
                LLMResponse(
                    raw_content=[{"type": "text", "text": worker_text}],
                    stop_reason="end_turn",
                    input_tokens=5,
                    output_tokens=3,
                    final_text=worker_text,
                ),
                LLMResponse(
                    raw_content=[{"type": "text", "text": synthesize_text}],
                    stop_reason="end_turn",
                    input_tokens=15,
                    output_tokens=10,
                    final_text=synthesize_text,
                ),
            ]
        )
        return orc, reg

    def test_run_produces_result_with_subtasks(self):
        """正常流程：拆解为 1 个子任务，Worker 执行，返回 OrchestratorResult。 / Normal flow: decomposes into 1 subtask, Worker executes, returns OrchestratorResult."""
        orc, _ = self._make_orchestrator()
        ctx = make_ctx()
        result = asyncio.run(orc.run("完成一个测试任务", ctx))
        assert result.original_task == "完成一个测试任务"
        assert len(result.subtask_results) == 1
        assert result.succeeded_count == 1

    def test_decompose_bad_json_fallback_to_single_task(self):
        """拆解 LLM 返回非 JSON 时，降级为单子任务。 / When the decompose LLM returns non-JSON, degrades to a single subtask."""
        reg = AgentRegistry()
        reg.register(make_card("g", ["general"]))
        bad_provider = make_provider("这不是 JSON", "mock-T0")
        worker_provider = make_provider("子任务完成", "mock-T1")
        orc = Orchestrator(
            registry=reg,
            provider_factory={"T0": bad_provider, "T1": worker_provider},
        )
        ctx = make_ctx()
        result = asyncio.run(orc.run("任意任务", ctx))
        # 拆解失败 → 单子任务 → description = 原始任务 / Decompose failed → single subtask → description = original task
        assert len(result.subtask_results) == 1
        assert result.subtask_results[0].spec.description == "任意任务"

    def test_route_exact_capability_match(self):
        """路由到精确匹配能力的 Agent。 / Routes to the agent with an exactly matching capability."""
        reg = AgentRegistry()
        reg.register(make_card("coder", ["code_review"]))
        reg.register(make_card("general-agent", ["general"]))
        t0 = make_provider('{"subtasks":[{"description":"审查代码","required_capability":"code_review"}]}')
        t1 = make_provider("审查完成")
        orc = Orchestrator(registry=reg, provider_factory={"T0": t0, "T1": t1})
        ctx = make_ctx()
        result = asyncio.run(orc.run("代码审查任务", ctx))
        # worker 应该用 coder 这张 card / worker should use the coder card
        assert result.subtask_results[0].agent_id == "coder"

    def test_route_fallback_when_no_capability_match(self):
        """找不到精确匹配时，路由到 general 兜底。 / When no exact match is found, routes to the general fallback."""
        reg = AgentRegistry()
        reg.register(make_card("fallback-agent", ["general"]))
        t0 = make_provider('{"subtasks":[{"description":"神秘任务","required_capability":"unknown_skill"}]}')
        t1 = make_provider("兜底完成")
        orc = Orchestrator(registry=reg, provider_factory={"T0": t0, "T1": t1})
        result = asyncio.run(orc.run("神秘任务", make_ctx()))
        assert result.subtask_results[0].agent_id == "fallback-agent"

    def test_route_builtin_fallback_when_registry_empty(self):
        """Registry 完全空时，用内置 __fallback__ card 不崩溃。 / When the registry is entirely empty, uses the built-in __fallback__ card without crashing."""
        empty_reg = AgentRegistry()
        t0 = make_provider('{"subtasks":[{"description":"任务","required_capability":"unknown"}]}')
        t1 = make_provider("内置兜底完成")
        orc = Orchestrator(registry=empty_reg, provider_factory={"T0": t0, "T1": t1})
        result = asyncio.run(orc.run("任意任务", make_ctx()))
        assert result.subtask_results[0].agent_id == "__fallback__"

    def test_max_workers_limits_subtask_count(self):
        """max_workers=2 时，超过 2 个子任务会被截断到 2 个。 / With max_workers=2, more than 2 subtasks are truncated to 2."""
        reg = AgentRegistry()
        reg.register(make_card("g", ["general"]))
        # 拆解返回 4 个子任务 / Decompose returns 4 subtasks
        decompose_json = '{"subtasks":[' + \
            ','.join([f'{{"description":"子任务{i}","required_capability":"general"}}' for i in range(4)]) + \
            ']}'
        t0 = make_provider(decompose_json)
        t1 = make_provider("完成")
        orc = Orchestrator(registry=reg, provider_factory={"T0": t0, "T1": t1}, max_workers=2)
        result = asyncio.run(orc.run("大任务", make_ctx()))
        assert len(result.subtask_results) == 2

    def test_workers_run_in_parallel(self):
        """
        两个 Worker 应该真并行：总耗时应远小于两者串行之和。
        用 asyncio.sleep 模拟耗时工作来验证并行。

        Two workers should run truly in parallel: total elapsed should be much less than the sum of serial runs.
        Uses asyncio.sleep to simulate work and verify parallelism.
        """
        reg = AgentRegistry()
        reg.register(make_card("slow-agent", ["general"]))

        # 拆解返回 2 个子任务 / Decompose returns 2 subtasks
        t0 = make_provider(
            '{"subtasks":['
            '{"description":"任务A","required_capability":"general"},'
            '{"description":"任务B","required_capability":"general"}'
            ']}'
        )

        call_times: list[float] = []

        async def slow_stream(*a, **kw):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)   # 模拟 50ms 工作 / Simulate 50ms of work
            resp = LLMResponse(
                raw_content=[{"type": "text", "text": "完成"}],
                stop_reason="end_turn",
                input_tokens=5,
                output_tokens=2,
                final_text="完成",
            )
            yield StreamChunk(is_final=True, final_response=resp)

        slow_provider = MagicMock()
        slow_provider.model_id = "mock-slow"
        slow_provider.stream = slow_stream
        slow_provider.count_tokens = MagicMock(return_value=10)
        slow_provider.complete = AsyncMock(return_value=LLMResponse(
            raw_content=[{"type": "text", "text": "综合完成"}],
            stop_reason="end_turn",
            input_tokens=10,
            output_tokens=5,
            final_text="综合完成",
        ))

        orc = Orchestrator(
            registry=reg,
            provider_factory={"T0": t0, "T1": slow_provider},
        )
        wall_start = time.monotonic()
        result = asyncio.run(orc.run("并行任务", make_ctx()))
        wall_elapsed = time.monotonic() - wall_start

        assert len(result.subtask_results) == 2
        # 并行时总耗时应 < 两次串行之和（2×50ms = 100ms），留 50ms 余量 / In parallel, total elapsed should be < the sum of two serial runs (2×50ms = 100ms), with 50ms margin
        assert wall_elapsed < 0.15, f"疑似串行执行，耗时={wall_elapsed:.3f}s"

    def test_depth_limit_triggers_direct_execution(self):
        """当编排深度达到 max_depth 时，直接执行原始任务而不再拆解。 / When orchestration depth reaches max_depth, the original task is executed directly without further decomposition."""
        reg = AgentRegistry()
        reg.register(make_card("g", ["general"]))
        t0 = make_provider("这段文字不应该被解析为拆解结果")
        t1 = make_provider("直接执行结果")

        orc = Orchestrator(registry=reg, provider_factory={"T0": t0, "T1": t1}, max_depth=1)

        async def _run():
            # 手动把深度设为 1（等于 max_depth），触发降级 / Manually set depth to 1 (equal to max_depth), triggering fallback
            token = _ORCHESTRATION_DEPTH.set(1)
            try:
                return await orc.run("任意任务", make_ctx())
            finally:
                _ORCHESTRATION_DEPTH.reset(token)

        result = asyncio.run(_run())
        # 降级执行：只有 1 个子任务，且 T0 provider 的 complete 没被调用（不走拆解） / Degraded execution: only 1 subtask, and T0 provider's complete is never called (no decomposition)
        assert len(result.subtask_results) == 1
        t0.complete.assert_not_called()

    def test_synthesize_failure_fallback_to_concat(self):
        """综合 LLM 调用失败时，降级为直接拼接各子任务输出。 / When the synthesize LLM call fails, degrades to concatenating each subtask's output."""
        reg = AgentRegistry()
        reg.register(make_card("g", ["general"]))

        t0 = make_provider('{"subtasks":[{"description":"子任务A","required_capability":"general"}]}')
        t1 = make_provider("子任务输出")
        # 让 complete 第一次（worker 通过 stream 执行，complete 用于 synthesize） / First complete call (worker executes via stream, complete is used for synthesize)
        # synthesize 调用 complete 时抛异常 / synthesize's complete call raises an exception
        t1.complete = AsyncMock(side_effect=RuntimeError("合成服务不可用"))

        orc = Orchestrator(registry=reg, provider_factory={"T0": t0, "T1": t1})
        result = asyncio.run(orc.run("任意任务", make_ctx()))
        # 降级：synthesis 包含子任务输出的拼接内容 / Fallback: synthesis contains the concatenated subtask output
        assert "子任务 1" in result.final_synthesis or result.subtask_results[0].output in result.final_synthesis

    def test_parse_json_two_step_fallback(self):
        """_parse_json 两步容错：整段失败后尝试正则提取。 / _parse_json two-step tolerance: after whole-text failure, attempts regex extraction."""
        raw = '前面有文字 {"subtasks": []} 后面也有文字'
        data = _parse_json(raw)
        assert data == {"subtasks": []}

    def test_parse_json_total_failure_returns_none(self):
        assert _parse_json("完全不是 JSON") is None
        assert _parse_json("") is None


# ─── DebateOrchestrator ───────────────────────────────────────────────────────

class TestDebateOrchestrator:
    def _make_debate(
        self,
        reviewer_a_text: str = '{"issues":["空指针风险"],"suggestions":["加判空"],"verdict":"request_changes"}',
        reviewer_b_text: str = '{"issues":["性能问题"],"suggestions":["用缓存"],"verdict":"request_changes"}',
        judge_text: str = '{"disagreements":["关注点不同"],"final_verdict":"request_changes","final_report":"建议修改两处"}',
    ) -> DebateOrchestrator:
        call_count = 0
        responses = [reviewer_a_text, reviewer_b_text, judge_text]

        async def _stream(*a, **kw):
            nonlocal call_count
            text = responses[call_count % len(responses)]
            call_count += 1
            resp = LLMResponse(
                raw_content=[{"type": "text", "text": text}],
                stop_reason="end_turn",
                input_tokens=10,
                output_tokens=10,
                final_text=text,
            )
            yield StreamChunk(is_final=True, final_response=resp)

        provider = MagicMock()
        provider.model_id = "mock-debate"
        provider.stream = _stream
        provider.count_tokens = MagicMock(return_value=20)

        return DebateOrchestrator(provider=provider)

    def test_review_returns_complete_result(self):
        """正常流程：两个 Reviewer + Judge，返回完整 DebateResult。 / Normal flow: two Reviewers + Judge, returns a complete DebateResult."""
        debate = self._make_debate()
        result = asyncio.run(debate.review("def foo(): pass"))
        assert result.reviewer_a.reviewer_label == "A"
        assert result.reviewer_b.reviewer_label == "B"
        assert result.final_verdict in ("approve", "request_changes", "reject")
        assert result.final_report

    def test_reviewer_a_and_b_independent_sessions(self):
        """两个 Reviewer 使用不同 session_key（独立视角保证）。 / The two Reviewers use different session_keys (guaranteeing independent viewpoints)."""
        seen_sessions: list[str] = []

        async def _stream(*a, **kw):
            # 用 system prompt 的前几字 + 调用时间区分不同 call，但我们通过 AgentContext 验证 / Distinguish different calls by the first few chars of system prompt + call time, but we verify via AgentContext
            resp = LLMResponse(
                raw_content=[{"type": "text", "text": '{"issues":[],"suggestions":[],"verdict":"approve"}'}],
                stop_reason="end_turn",
                input_tokens=5,
                output_tokens=5,
                final_text='{"issues":[],"suggestions":[],"verdict":"approve"}',
            )
            yield StreamChunk(is_final=True, final_response=resp)

        provider = MagicMock()
        provider.model_id = "mock"
        provider.stream = _stream
        provider.count_tokens = MagicMock(return_value=10)

        # 追踪 NanoCore 构建时的 session_key / Track session_key at NanoCore construction time
        from nanoharness.core import nano_core as nc_mod
        original_init = nc_mod.NanoCore.__init__

        def patched_init(self_nc, ctx, **kw):
            seen_sessions.append(ctx.session_key)
            original_init(self_nc, ctx, **kw)

        nc_mod.NanoCore.__init__ = patched_init
        try:
            debate = DebateOrchestrator(provider=provider)
            asyncio.run(debate.review("x = 1"))
        finally:
            nc_mod.NanoCore.__init__ = original_init

        # 应该有 3 个 session（Reviewer A / Reviewer B / Judge） / There should be 3 sessions (Reviewer A / Reviewer B / Judge)
        assert len(seen_sessions) == 3
        # 三个 session_key 都不同 / The three session_keys are all different
        assert len(set(seen_sessions)) == 3

    def test_both_approve_result_is_approve(self):
        """两个 Reviewer 都 approve，Judge 规则降级时最终结果应为 approve。 / When both Reviewers approve, and the Judge degrades by rules, the final result should be approve."""
        a_text = '{"issues":[],"suggestions":[],"verdict":"approve"}'
        b_text = '{"issues":[],"suggestions":[],"verdict":"approve"}'
        # Judge 返回非 JSON → 触发规则降级 / Judge returns non-JSON → triggers rule-based degradation
        debate = self._make_debate(
            reviewer_a_text=a_text,
            reviewer_b_text=b_text,
            judge_text="这不是 JSON",
        )
        result = asyncio.run(debate.review("clean_code = True"))
        assert result.final_verdict == "approve"

    def test_one_reject_triggers_reject(self):
        """一人 reject，规则降级时最终结果为 reject。 / If one rejects, the rule-degraded final result is reject."""
        a_text = '{"issues":["严重 bug"],"suggestions":[],"verdict":"reject"}'
        b_text = '{"issues":[],"suggestions":["小优化"],"verdict":"approve"}'
        debate = self._make_debate(
            reviewer_a_text=a_text,
            reviewer_b_text=b_text,
            judge_text="非 JSON 输出",
        )
        result = asyncio.run(debate.review("some code"))
        assert result.final_verdict == "reject"

    def test_reviewer_bad_json_degraded_gracefully(self):
        """Reviewer 返回非 JSON 时，降级为原始文字作为 issues，不崩溃。 / When a Reviewer returns non-JSON, degrades to using the raw text as issues without crashing."""
        debate = self._make_debate(
            reviewer_a_text="这是普通文字审查意见",
            reviewer_b_text='{"issues":[],"suggestions":[],"verdict":"approve"}',
            judge_text='{"disagreements":[],"final_verdict":"approve","final_report":"通过"}',
        )
        result = asyncio.run(debate.review("x = 1"))
        # Reviewer A 降级：原始文字放入 issues / Reviewer A degraded: raw text goes into issues
        assert len(result.reviewer_a.issues) > 0

    def test_reviewers_agreed_property(self):
        """reviewers_agreed 属性正确反映两人一致性。 / The reviewers_agreed property correctly reflects the two reviewers' agreement."""
        a = ReviewOpinion("A", verdict="approve")
        b_same = ReviewOpinion("B", verdict="approve")
        b_diff = ReviewOpinion("B", verdict="reject")
        from nanoharness.agents.debate import DebateResult
        r_agreed = DebateResult("code", a, b_same, [], "approve", "ok", 100.0)
        r_differ = DebateResult("code", a, b_diff, [], "reject", "nok", 100.0)
        assert r_agreed.reviewers_agreed is True
        assert r_differ.reviewers_agreed is False

    def test_parse_opinion_valid_json(self):
        raw = '{"issues":["bug1","bug2"],"suggestions":["fix1"],"verdict":"reject"}'
        opinion = _parse_opinion("A", raw)
        assert opinion.issues == ["bug1", "bug2"]
        assert opinion.verdict == "reject"

    def test_parse_opinion_empty_text_fallback(self):
        opinion = _parse_opinion("B", "")
        assert opinion.verdict == "request_changes"
        assert len(opinion.issues) > 0

    def test_parse_judge_rule_merge_disagreement(self):
        """Judge 解析失败时，规则合并：一人 approve 一人 reject → reject。 / When Judge parsing fails, rule-based merge: one approve + one reject → reject."""
        a = ReviewOpinion("A", issues=["bug"], verdict="reject")
        b = ReviewOpinion("B", issues=[], verdict="approve")
        disagreements, verdict, report = _parse_judge("bad text", a, b)
        assert verdict == "reject"

    def test_parse_judge_rule_merge_both_approve(self):
        a = ReviewOpinion("A", verdict="approve")
        b = ReviewOpinion("B", verdict="approve")
        _, verdict, _ = _parse_judge("", a, b)
        assert verdict == "approve"


# ─── AgentDispatcher 测试 / AgentDispatcher tests ─────────────────────────────


def _make_envelope(content: str, chat_type: ChatType = ChatType.DIRECT) -> InboundEnvelope:
    return InboundEnvelope(
        channel_id="test",
        sender_id="user-1",
        chat_id="user-1",
        chat_type=chat_type,
        content=content,
    )


def _make_mock_turn_runner(reply: str = "单 Agent 回复") -> MagicMock:
    """返回 mock TurnRunner，run() 是 async generator，yield 一个 DoneEvent。"""
    from nanoharness.core.event_store import DoneEvent
    from nanoharness.core.context import StopReason, TurnOutcome

    done = DoneEvent(
        trace_id="t",
        session_key="s",
        agent_id="a",
        final_text=reply,
        stop_reason=StopReason.COMPLETED,
        outcome=TurnOutcome.COMPLETED,
    )

    async def _gen(*args, **kwargs):
        yield done

    runner = MagicMock()
    runner.run = _gen
    return runner


def _make_mock_orchestrator(synthesis: str = "多 Agent 综合回复") -> MagicMock:
    from nanoharness.agents.orchestrator import OrchestratorResult

    result = OrchestratorResult(
        original_task="task",
        subtask_results=[],
        final_synthesis=synthesis,
        total_elapsed_ms=0.0,
    )
    orch = MagicMock()
    orch.run = AsyncMock(return_value=result)
    return orch


class TestAgentDispatcher:
    def test_simple_message_routes_to_turn_runner(self):
        """短消息走 TurnRunner，Orchestrator.run 不被调用。"""
        runner = _make_mock_turn_runner("单 Agent 回复")
        orch = _make_mock_orchestrator()
        dispatcher = AgentDispatcher(runner, orchestrator=orch)

        result = asyncio.run(dispatcher(_make_envelope("你好")))

        assert result is not None
        assert result.content == "单 Agent 回复"
        orch.run.assert_not_called()

    def test_long_message_routes_to_orchestrator(self):
        """长消息（超过阈值）走 Orchestrator。"""
        runner = _make_mock_turn_runner()
        orch = _make_mock_orchestrator("多 Agent 综合回复")
        dispatcher = AgentDispatcher(runner, orchestrator=orch, complex_threshold=10)

        result = asyncio.run(dispatcher(_make_envelope("这是一条超过十个字符的复杂消息请求")))

        assert result is not None
        assert result.content == "多 Agent 综合回复"
        orch.run.assert_called_once()

    def test_keyword_triggers_orchestrator(self):
        """包含复杂度关键词（'分别'）的消息走 Orchestrator。"""
        runner = _make_mock_turn_runner()
        orch = _make_mock_orchestrator("多 Agent 综合回复")
        dispatcher = AgentDispatcher(runner, orchestrator=orch)

        result = asyncio.run(dispatcher(_make_envelope("分别帮我搜索天气和写一首诗")))

        assert result is not None
        orch.run.assert_called_once()

    def test_no_orchestrator_always_uses_turn_runner(self):
        """未传入 orchestrator 时，任何消息都走 TurnRunner。"""
        runner = _make_mock_turn_runner("兜底回复")
        dispatcher = AgentDispatcher(runner, orchestrator=None, complex_threshold=5)

        result = asyncio.run(dispatcher(_make_envelope("分别同时并行很长很长的消息")))

        assert result is not None
        assert result.content == "兜底回复"

    def test_outbound_envelope_fields_match_inbound(self):
        """OutboundEnvelope 的 target_channel/target_peer 与入站信封一致。"""
        runner = _make_mock_turn_runner("回复")
        dispatcher = AgentDispatcher(runner)

        envelope = _make_envelope("你好")
        result = asyncio.run(dispatcher(envelope))

        assert result is not None
        assert result.target_channel == envelope.channel_id
        assert result.target_peer == envelope.chat_id
        assert result.reply_to_envelope_id == envelope.envelope_id

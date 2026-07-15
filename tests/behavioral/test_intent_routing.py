"""
意图路由行为测试。

行为测试的视角：不关心路由器输出了什么文字理由，
只关心：
  1. 给定输入消息，分类出的 tier 是否符合预期
  2. 被路由到的 Agent 执行后，行为指纹是否满足约束

全部使用 Mock Provider，不依赖真实 LLM。
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from nanoharness.provider.base import LLMResponse, StreamChunk
from nanoharness.router.llm_router import LLMRouter, heuristic_classify
from nanoharness.router.tiers import Tier

from tests.behavioral.fingerprint import (
    BehaviorConstraint,
    BehaviorFingerprint,
    run_and_fingerprint,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_classify_provider(tier: str, confidence: float = 0.9) -> MagicMock:
    """返回一个 mock provider，complete() 返回指定 tier 的 JSON。"""
    provider = MagicMock()
    provider.model_id = "test-classify-model"
    provider.complete = AsyncMock(return_value=MagicMock(
        final_text=f'{{"tier": "{tier}", "confidence": {confidence}, "reason": "测试"}}'
    ))
    return provider


def make_text_provider(*chunks: str) -> MagicMock:
    """返回一个流式文本 mock provider（无工具调用）。"""
    final_text = "".join(chunks)
    final_response = LLMResponse(
        raw_content=[{"type": "text", "text": final_text}],
        stop_reason="end_turn",
        final_text=final_text,
    )

    async def _stream(*args, **kwargs):
        for chunk in chunks:
            yield StreamChunk(delta_text=chunk)
        yield StreamChunk(is_final=True, final_response=final_response)

    provider = MagicMock()
    provider.model_id = "test-model"
    provider.stream = _stream
    provider.count_tokens = MagicMock(return_value=20)
    return provider


def make_tool_provider(tool_name: str, tool_use_id: str = "tc-1") -> MagicMock:
    """返回一个先请求工具调用、再返回最终文本的 mock provider。"""
    from nanoharness.provider.base import ToolCall

    tool_call = ToolCall(tool_use_id=tool_use_id, tool_name=tool_name, tool_input={})
    call_count = 0

    async def _stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raw = [{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": {}}]
            resp = LLMResponse(raw_content=raw, stop_reason="tool_use", tool_calls=[tool_call])
        else:
            resp = LLMResponse(
                raw_content=[{"type": "text", "text": "工具调用完成。"}],
                stop_reason="end_turn", final_text="工具调用完成。",
            )
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "test-model"
    provider.stream = _stream
    provider.count_tokens = MagicMock(return_value=20)
    return provider


# ─── LLMRouter 分类行为测试 ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_simple_greeting_routed_to_t0():
    """简单打招呼场景 → LLM 返回 T0 分类 → 路由结果是 T0。"""
    provider = make_classify_provider("T0", confidence=0.95)
    router = LLMRouter(provider=provider)

    result = await router.classify("你好，今天天气怎么样？")

    assert result.tier == Tier.T0
    assert result.confidence >= 0.9
    assert result.method == "llm"


@pytest.mark.asyncio
async def test_complex_refactoring_routed_to_t3():
    """复杂架构重构场景 → LLM 返回 T3 分类。"""
    provider = make_classify_provider("T3", confidence=0.88)
    router = LLMRouter(provider=provider)

    result = await router.classify("帮我分析这个微服务架构的安全漏洞并重构认证模块")

    assert result.tier == Tier.T3
    assert result.method == "llm"


@pytest.mark.asyncio
async def test_llm_timeout_degrades_to_heuristic():
    """LLM 调用超时 → 降级到规则启发式 → method == 'heuristic' 或 'fallback'。"""
    async def _slow_complete(*args, **kwargs):
        await asyncio.sleep(10)  # 明显超过 timeout
        return MagicMock(final_text='{"tier": "T1"}')

    provider = MagicMock()
    provider.model_id = "slow-model"
    provider.complete = _slow_complete

    router = LLMRouter(provider=provider, timeout=0.05)
    result = await router.classify("你好")

    # 超时后必须降级到非 llm 方法
    assert result.method in ("heuristic", "fallback")
    assert result.tier in (Tier.T0, Tier.T1, Tier.T2, Tier.T3)  # 有效值


@pytest.mark.asyncio
async def test_invalid_json_response_falls_back_to_t1():
    """LLM 返回非 JSON 垃圾文本 → 解析失败 → method == 'fallback' → tier == T1。"""
    provider = MagicMock()
    provider.model_id = "garbage-model"
    provider.complete = AsyncMock(return_value=MagicMock(
        final_text="对不起，我无法分类这个任务。"
    ))

    router = LLMRouter(provider=provider)
    result = await router.classify("任意输入")

    assert result.method == "fallback"
    assert result.tier == Tier.T1


def test_heuristic_classifier_greeting_matches_t0():
    """规则启发式：'你好' 关键词 → T0。"""
    result = heuristic_classify("你好啊，今天天气不错")
    assert result.tier == Tier.T0
    assert result.method == "heuristic"


def test_heuristic_classifier_refactor_matches_t3():
    """规则启发式：'重构' 关键词 → T3。"""
    result = heuristic_classify("请帮我重构这段服务代码")
    assert result.tier == Tier.T3
    assert result.method == "heuristic"


# ─── 指纹约束测试：路由到不同 tier 后的行为预期 ────────────────────────────────

@pytest.mark.asyncio
async def test_t0_routed_agent_produces_no_tool_calls():
    """
    T0 路由场景：简单问题不应触发工具调用。
    用 BehaviorConstraint 断言，不断言输出文字。
    """
    # T0 任务：只返回文本，不调用工具
    provider = make_text_provider("今天是晴天。")

    fp = await run_and_fingerprint("今天天气怎么样？", provider)

    # 行为约束：T0 任务不应有任何工具调用
    constraint = BehaviorConstraint(
        call_count_min=0,
        call_count_max=0,
        must_complete=True,
        error_allowed=False,
    )
    constraint.assert_satisfied(fp)
    assert fp.reached_done


@pytest.mark.asyncio
async def test_t1_routed_agent_with_tool_satisfies_constraint():
    """
    T1 路由场景：任务需要工具调用，约束要求 'search' 被调用。
    用 BehaviorConstraint 断言工具集合的超集关系。
    """
    from nanoharness.core.tool_executor import ToolRegistry, ToolDefinition

    async def search_fn(inputs: dict, ctx):
        return "搜索结果：今天天气晴"

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="search",
        description="搜索引擎",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        fn=search_fn,
    ))

    provider = make_tool_provider("search")

    fp = await run_and_fingerprint(
        "搜索今天天气",
        provider,
        tools=registry.as_fn_dict(),
        tool_definitions=registry.as_api_list(),
    )

    # 超集约束：实际调用集合 ⊇ {"search"}
    constraint = BehaviorConstraint(
        must_call_tools={"search"},
        must_execute_tools={"search"},
        call_count_min=1,
        call_count_max=2,   # 允许 ±1 浮动
        must_complete=True,
    )
    constraint.assert_satisfied(fp)

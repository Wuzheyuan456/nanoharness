"""
执行流深度控制行为测试 / Execution-flow control behavioral tests.

覆盖 Phase 8 的 6 个机制 / Covers the 6 Phase 8 mechanisms:
  A. StuckDetector — 重复签名（含振荡 A-B-A-B）/ repeated signature (incl. oscillation)
  B. StopReason + TurnOutcome — 终止原因可观测 / observable stop reason
  C. 两阶段优雅收尾 — max_iter 撞线先注入指令再做一次 / two-phase graceful finalization
  D. 工具返回契约 — ToolResult 序列化 / ToolResult contract serialization (单测在 test_core.py)
  E. 动态工具禁用 — 卡死后从 tool_definitions 隐藏 / dynamic disabling hides from tool_definitions
  F. per-tool 调用预算 — catch 同工具不同参数钻空子 / per-tool budget catches varying-args evasion

风格与 test_safety.py 一致：mock provider，不断言 LLM 输出文字，只断言事件类型 / 状态 / 计数 / 干预原因。
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from nanoharness.core.context import AgentContext
from nanoharness.core.event_store import (
    DoneEvent,
    ErrorEvent,
    EventStore,
    InterventionEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from nanoharness.core.nano_core import NanoCore
from nanoharness.core.tool_executor import ToolDefinition, ToolRegistry
from nanoharness.provider.base import LLMResponse, StreamChunk, ToolCall

# ── 辅助 / Helpers ──────────────────────────────────────────────────────────────

def make_ctx() -> AgentContext:
    return AgentContext(
        agent_id="ctrl-agent",
        session_key="ctrl-session",
        system_prompt="You are a test assistant.",
        model_id="test-model",
    )


def _tool_use_block(tc: ToolCall) -> dict[str, Any]:
    return {"type": "tool_use", "id": tc.tool_use_id, "name": tc.tool_name, "input": tc.tool_input}


def _text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


async def run_turn(
    provider: Any,
    tools: dict | None = None,
    tool_defs: list | None = None,
    *,
    max_iter: int = 20,
    max_tool_calls: int = 40,
    max_calls_per_tool: int = 5,
    stuck_threshold: int = 3,
) -> tuple[list, AgentContext]:
    """跑一轮 NanoCore，返回事件列表与 ctx / Run one NanoCore turn, return (events, ctx)."""
    ctx = make_ctx()
    store = EventStore()
    core = NanoCore(
        ctx=ctx, provider=provider,
        tools=tools or {}, tool_definitions=tool_defs or [],
        event_store=store,
        max_iter=max_iter, max_tool_calls=max_tool_calls,
        max_calls_per_tool=max_calls_per_tool, stuck_threshold=stuck_threshold,
    )
    events: list = []
    async for ev in core.run_turn("control test"):
        events.append(ev)
    return events, ctx


def _stop_reason(events: list) -> str:
    for ev in events:
        if isinstance(ev, DoneEvent):
            return ev.stop_reason
    return ""


def _outcome(events: list) -> str:
    for ev in events:
        if isinstance(ev, DoneEvent):
            return ev.outcome
    return ""


def _intervention_reasons(events: list) -> list[str]:
    return [ev.reason for ev in events if isinstance(ev, InterventionEvent)]


def _success_count(events: list, tool_name: str) -> int:
    return sum(
        1 for ev in events
        if isinstance(ev, ToolResultEvent) and ev.tool_name == tool_name and ev.success
    )


def _tool_call_count(events: list, tool_name: str) -> int:
    return sum(
        1 for ev in events
        if isinstance(ev, ToolCallEvent) and ev.tool_name == tool_name
    )


# ── 测试 / Tests ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stuck_repeated_tool_triggers_intervention_and_recovery():
    """A: 同工具同参数连续请求 3 次 → 第 3 次被跳过（不执行），发 stuck_loop 干预，恢复后模型收尾 / Same tool+args requested 3× → 3rd skipped (not executed), stuck_loop intervention fired, model finishes after recovery."""
    call_count = 0

    async def stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            tc = ToolCall(tool_use_id=f"su-{call_count}", tool_name="search", tool_input={"q": "same"})
            resp = LLMResponse(raw_content=[_tool_use_block(tc)], stop_reason="tool_use", tool_calls=[tc])
        else:
            resp = LLMResponse(raw_content=[_text_block("done")], stop_reason="end_turn", final_text="done")
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "t"
    provider.stream = stream
    provider.count_tokens = MagicMock(return_value=10)

    async def search_fn(inputs: dict, ctx):
        return "result"
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="search", description="search", input_schema={"type": "object"}, fn=search_fn,
    ))

    events, _ = await run_turn(provider, registry.as_fn_dict(), registry.as_api_list(), stuck_threshold=3)

    assert "stuck_loop" in _intervention_reasons(events)
    assert _stop_reason(events) == "completed"
    # LLM 请求 search 3 次 / LLM requested search 3 times
    assert _tool_call_count(events, "search") == 3
    # 但只成功执行了 2 次（第 3 次被跳过）/ but only 2 executed successfully (3rd skipped)
    assert _success_count(events, "search") == 2


@pytest.mark.asyncio
async def test_stuck_oscillation_a_b_a_b_detected():
    """A: 振荡 A-B-A-B-A（同输入重复签名）也能被 per-签名计数 catch；旧"只看连续相同"检测器抓不到这个 / Oscillation A-B-A-B-A (repeated signatures) is caught by per-signature counting; a consecutive-only detector would miss this."""
    # 同一工具始终用同一输入 → 签名会重复 / same tool always uses same input → signature repeats
    sequence = [("A", {"q": "x"}), ("B", {"q": "y"}), ("A", {"q": "x"}), ("B", {"q": "y"}), ("A", {"q": "x"})]
    call_count = 0

    async def stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= len(sequence):
            name, inp = sequence[call_count - 1]
            tc = ToolCall(tool_use_id=f"su-{call_count}", tool_name=name, tool_input=inp)
            resp = LLMResponse(raw_content=[_tool_use_block(tc)], stop_reason="tool_use", tool_calls=[tc])
        else:
            resp = LLMResponse(raw_content=[_text_block("done")], stop_reason="end_turn", final_text="done")
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "t"
    provider.stream = stream
    provider.count_tokens = MagicMock(return_value=10)

    async def a_fn(inp, ctx):
        return "a-result"

    async def b_fn(inp, ctx):
        return "b-result"

    tools = {"A": a_fn, "B": b_fn}
    defs = [
        {"name": "A", "description": "A", "input_schema": {"type": "object"}},
        {"name": "B", "description": "B", "input_schema": {"type": "object"}},
    ]

    events, _ = await run_turn(provider, tools, defs, stuck_threshold=3)

    # A 的签名在第 5 次累计到 3 → stuck；连续检测器只会看到 A-B-A-B-A 永远没有 3 连续相同 / A's signature reaches 3 on call 5 → stuck; a consecutive detector sees A-B-A-B-A and never 3-in-a-row identical
    assert "stuck_loop" in _intervention_reasons(events)
    assert _stop_reason(events) == "completed"


@pytest.mark.asyncio
async def test_per_tool_budget_catches_varying_args():
    """F: 同工具、不同参数调用到预算上限 → 触发 tool_call_budget 干预并禁用工具（绕过签名去重）/ Same tool, varying args hitting the per-tool budget → tool_call_budget intervention fires and disables the tool (evades signature dedup)."""
    call_count = 0

    async def stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 4:
            # 每次不同参数，签名去重抓不到 / different args each time — signature dedup won't catch
            tc = ToolCall(tool_use_id=f"su-{call_count}", tool_name="search", tool_input={"q": str(call_count)})
            resp = LLMResponse(raw_content=[_tool_use_block(tc)], stop_reason="tool_use", tool_calls=[tc])
        else:
            resp = LLMResponse(raw_content=[_text_block("done")], stop_reason="end_turn", final_text="done")
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "t"
    provider.stream = stream
    provider.count_tokens = MagicMock(return_value=10)

    async def search_fn(inputs: dict, ctx):
        return "result"
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="search", description="search", input_schema={"type": "object"}, fn=search_fn,
    ))

    events, _ = await run_turn(
        provider, registry.as_fn_dict(), registry.as_api_list(),
        max_calls_per_tool=3, stuck_threshold=99,  # 调高 stuck 阈值，隔离 F 的行为 / raise stuck threshold to isolate F
    )

    assert "tool_call_budget" in _intervention_reasons(events)
    assert "stuck_loop" not in _intervention_reasons(events)
    assert _stop_reason(events) == "completed"
    # 预算耗尽后第 4 次请求被跳过 / 4th request skipped after budget exhausted
    assert _success_count(events, "search") == 3


@pytest.mark.asyncio
async def test_two_phase_finalization_on_max_iter():
    """C: max_iter 撞线时注入 finalization 指令 + 剥离工具做一次，stop_reason=max_iterations / outcome=partial / On max_iter hit: inject finalization directive + tool-stripped retry; stop_reason=max_iterations / outcome=partial."""
    call_count = 0
    captured_tools: list = []

    async def stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_tools.append(kwargs.get("tools"))
        if call_count <= 2:
            tc = ToolCall(tool_use_id=f"su-{call_count}", tool_name="search", tool_input={"q": str(call_count)})
            resp = LLMResponse(raw_content=[_tool_use_block(tc)], stop_reason="tool_use", tool_calls=[tc])
        else:
            # 第 3 次调用（finalization，tools 应为 None）给最终答案 / 3rd call (finalization, tools should be None) gives final answer
            resp = LLMResponse(raw_content=[_text_block("final answer")], stop_reason="end_turn", final_text="final answer")
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "t"
    provider.stream = stream
    provider.count_tokens = MagicMock(return_value=10)

    async def search_fn(inputs: dict, ctx):
        return "ok"
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="search", description="search", input_schema={"type": "object"}, fn=search_fn,
    ))

    events, _ = await run_turn(
        provider, registry.as_fn_dict(), registry.as_api_list(),
        max_iter=2, stuck_threshold=99,
    )

    assert "finalization" in _intervention_reasons(events)
    assert _stop_reason(events) == "max_iterations"
    assert _outcome(events) == "partial"
    # finalization 那次调用 tools 被剥离为 None / the finalization call had tools stripped to None
    assert captured_tools[2] is None


@pytest.mark.asyncio
async def test_normal_done_has_completed_stop_reason():
    """B: 正常收尾 → stop_reason=completed / outcome=completed / Normal finish → stop_reason=completed / outcome=completed."""

    async def stream(*args, **kwargs):
        resp = LLMResponse(raw_content=[_text_block("hello")], stop_reason="end_turn", final_text="hello")
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "t"
    provider.stream = stream
    provider.count_tokens = MagicMock(return_value=10)

    events, _ = await run_turn(provider)

    assert _stop_reason(events) == "completed"
    assert _outcome(events) == "completed"


@pytest.mark.asyncio
async def test_dynamic_disabling_hides_tool_from_definitions():
    """E: 卡死后该工具从后续 provider 调用的 tool_definitions 中隐藏（另一个工具仍在）/ After stuck fires the disabled tool is hidden from subsequent provider tool_definitions (another tool remains)."""
    call_count = 0
    captured_tools: list = []

    async def stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        captured_tools.append(kwargs.get("tools"))
        if call_count <= 3:
            tc = ToolCall(tool_use_id=f"su-{call_count}", tool_name="search", tool_input={"q": "same"})
            resp = LLMResponse(raw_content=[_tool_use_block(tc)], stop_reason="tool_use", tool_calls=[tc])
        else:
            resp = LLMResponse(raw_content=[_text_block("done")], stop_reason="end_turn", final_text="done")
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "t"
    provider.stream = stream
    provider.count_tokens = MagicMock(return_value=10)

    async def search_fn(inp, ctx):
        return "r"

    async def calc_fn(inp, ctx):
        return "c"

    registry = ToolRegistry()
    registry.register(ToolDefinition(name="search", description="search", input_schema={"type": "object"}, fn=search_fn))
    registry.register(ToolDefinition(name="calc", description="calc", input_schema={"type": "object"}, fn=calc_fn))

    events, _ = await run_turn(provider, registry.as_fn_dict(), registry.as_api_list(), stuck_threshold=3)

    assert "stuck_loop" in _intervention_reasons(events)
    # 第 3 次调用后 search 被禁用 → 第 4 次调用收到的 tools 仍含 calc 但不含 search / after 3rd call search disabled → 4th call's tools still has calc but not search
    tools_after = captured_tools[3]
    assert tools_after is not None
    names = {d.get("name") for d in tools_after}
    assert "search" not in names
    assert "calc" in names


@pytest.mark.asyncio
async def test_error_turn_has_failed_outcome():
    """B: provider 抛不可恢复错误 → ErrorEvent，未达 DONE（等价 failed）/ Provider raises unrecoverable error → ErrorEvent, never reaches DONE (= failed)."""
    from nanoharness.provider.base import ProviderError, ProviderErrorType

    async def stream(*args, **kwargs):
        raise ProviderError(ProviderErrorType.AUTH_INVALID, "bad key", retryable=False)
        yield  # 让 stream 成为 async generator（此行不可达但需存在以构成生成器）/ make stream an async generator (unreachable but required to form a generator)

    provider = MagicMock()
    provider.model_id = "t"
    provider.stream = stream
    provider.count_tokens = MagicMock(return_value=10)

    events, _ = await run_turn(provider)

    assert any(isinstance(e, ErrorEvent) for e in events)
    assert not any(isinstance(e, DoneEvent) for e in events)

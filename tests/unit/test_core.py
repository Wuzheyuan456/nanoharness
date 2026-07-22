"""
核心单测 / Core unit tests：不依赖真实 LLM，mock provider 验证 NanoCore 状态机逻辑。
行为指纹测试 / Behavior-fingerprint tests：断言状态转换序列、事件类型、工具调用次数，不断言输出文本。
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nanoharness.core.context import AgentContext, AgentState
from nanoharness.core.event_store import (
    DoneEvent,
    EventStore,
    StateChangeEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from nanoharness.core.nano_core import NanoCore
from nanoharness.core.tool_executor import ToolDefinition, ToolRegistry
from nanoharness.provider.base import LLMResponse, StreamChunk, ToolCall

# ─── 测试固件 / Fixtures ─────────────────────────────────────────────────────────────────

def make_ctx(agent_id: str = "test-agent", session_key: str = "sess-1") -> AgentContext:
    return AgentContext(
        agent_id=agent_id,
        session_key=session_key,
        system_prompt="You are a test assistant.",
        model_id="claude-haiku-4-5-20251001",
    )


def make_stream(*chunks: str, tool_calls: list[ToolCall] | None = None):
    """
    返回一个 async mock provider / Returns an async mock provider，
    stream() 会产出文本分块，并可选地附带最终 tool_use 响应。
    """
    stop_reason = "tool_use" if tool_calls else "end_turn"
    final_text = "".join(chunks) if not tool_calls else ""
    raw_content = [{"type": "text", "text": final_text}] if final_text else []
    if tool_calls:
        for tc in tool_calls:
            raw_content.append({"type": "tool_use", "id": tc.tool_use_id, "name": tc.tool_name, "input": tc.tool_input})

    final_response = LLMResponse(
        raw_content=raw_content,
        stop_reason=stop_reason,
        input_tokens=10,
        output_tokens=len(chunks),
        tool_calls=tool_calls or [],
        final_text=final_text,
    )

    async def _stream(*args, **kwargs):
        for chunk in chunks:
            yield StreamChunk(delta_text=chunk)
        yield StreamChunk(is_final=True, final_response=final_response)

    provider = MagicMock()
    provider.model_id = "test-model"
    provider.stream = _stream
    provider.count_tokens = MagicMock(return_value=50)
    return provider


# ─── 测试 / Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_simple_text_response():
    """NanoCore 在无工具调用时产出 DoneEvent，状态机经过 IDLE→THINKING→DONE。 / NanoCore emits DoneEvent when no tool call; state machine goes IDLE→THINKING→DONE."""
    ctx = make_ctx()
    store = EventStore()
    provider = make_stream("Hello ", "world!")

    core = NanoCore(ctx=ctx, provider=provider, tools={}, tool_definitions=[], event_store=store)

    events = []
    async for ev in core.run_turn("Hi"):
        events.append(ev)

    # 最后一个事件必须是 DoneEvent / Last event must be DoneEvent
    assert isinstance(events[-1], DoneEvent)
    done: DoneEvent = events[-1]
    assert done.final_text == "Hello world!"
    assert done.total_tool_calls == 0

    # 状态转换序列 / State transition sequence
    state_changes = [e for e in events if isinstance(e, StateChangeEvent)]
    transitions = [(e.from_state, e.to_state) for e in state_changes]
    assert (AgentState.IDLE, AgentState.THINKING) in transitions
    assert (AgentState.THINKING, AgentState.DONE) in transitions

    # EventStore 有记录 / EventStore has records
    assert len(store) > 0


@pytest.mark.asyncio
async def test_tool_call_then_final_answer():
    """NanoCore 在工具调用后再次问 LLM，最终产出 DoneEvent。 / NanoCore re-queries LLM after a tool call, finally emits DoneEvent."""
    ctx = make_ctx()
    store = EventStore()

    # 第一次调用：返回工具请求 / First call: returns a tool request
    tool_call = ToolCall(tool_use_id="tc-1", tool_name="add", tool_input={"a": 1, "b": 2})
    first_provider = make_stream(tool_calls=[tool_call])

    # 第二次调用：返回最终答案 / Second call: returns the final answer
    second_provider = make_stream("The answer is 3.")

    call_count = 0

    async def smart_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            async for chunk in first_provider.stream():
                yield chunk
        else:
            async for chunk in second_provider.stream():
                yield chunk

    provider = MagicMock()
    provider.model_id = "test-model"
    provider.stream = smart_stream
    provider.count_tokens = MagicMock(return_value=50)

    async def add_fn(inputs: dict, ctx):
        return str(inputs["a"] + inputs["b"])

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="add", description="Add two numbers",
        input_schema={"type": "object", "properties": {"a": {}, "b": {}}},
        fn=add_fn,
    ))

    core = NanoCore(
        ctx=ctx, provider=provider,
        tools=registry.as_fn_dict(),
        tool_definitions=registry.as_api_list(),
        event_store=store,
    )

    events = []
    async for ev in core.run_turn("What is 1+2?"):
        events.append(ev)

    assert isinstance(events[-1], DoneEvent)
    assert events[-1].total_tool_calls == 1

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "add"

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert tool_results[0].success is True


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_result():
    """调用未注册的工具时，ToolResultEvent.success == False，但 NanoCore 继续运行。 / Calling an unregistered tool yields ToolResultEvent.success == False, but NanoCore keeps running."""
    ctx = make_ctx()
    tool_call = ToolCall(tool_use_id="tc-99", tool_name="nonexistent", tool_input={})

    call_count = 0

    async def smart_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raw = [{"type": "tool_use", "id": "tc-99", "name": "nonexistent", "input": {}}]
            resp = LLMResponse(raw_content=raw, stop_reason="tool_use", tool_calls=[tool_call])
            yield StreamChunk(is_final=True, final_response=resp)
        else:
            resp = LLMResponse(raw_content=[{"type": "text", "text": "I couldn't do that."}],
                               stop_reason="end_turn", final_text="I couldn't do that.")
            yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "test"
    provider.stream = smart_stream
    provider.count_tokens = MagicMock(return_value=10)

    core = NanoCore(ctx=ctx, provider=provider, tools={}, tool_definitions=[])

    events = []
    async for ev in core.run_turn("Do something"):
        events.append(ev)

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert tool_results[0].success is False


@pytest.mark.asyncio
async def test_compaction_turn_boundary_protection():
    """retreat_to_turn_boundary 不会切断 tool_use / tool_result 配对。 / retreat_to_turn_boundary does not split a tool_use / tool_result pair."""
    from nanoharness.core.compaction import find_turn_boundary_cut, retreat_to_turn_boundary
    from nanoharness.core.context import Message

    messages = [
        Message(role="user", content="q1", token_count=10),
        Message(role="assistant", content=[{"type": "tool_use", "id": "x", "name": "f", "input": {}}], token_count=10),
        Message(role="user", content=[{"type": "tool_result", "tool_use_id": "x", "content": "ok"}], token_count=10),
        Message(role="assistant", content="Final answer.", token_count=10),
    ]

    # 预算刚好会切在 tool_use/tool_result 配对边界 / Budget that would cut right at the tool_use/tool_result pair boundary
    cut = find_turn_boundary_cut(messages, keep_budget_tokens=20)
    safe_cut = retreat_to_turn_boundary(messages, cut)

    # 安全切点不能落在 tool_use 与其 tool_result 之间 / Safe cut must not land between a tool_use and its tool_result
    if safe_cut < len(messages):
        assert not messages[safe_cut].is_tool_result(), "Cut must not start with orphaned tool_result"
    if safe_cut > 0:
        assert not messages[safe_cut - 1].is_tool_use(), "Cut must not follow tool_use without tool_result"


# ── 工具返回契约 / Tool Result Contract ────────────────────────────────────────

def test_tool_result_contract_wraps_bare_string():
    """裸 str 返回值被自动包装为 SUCCESS 契约（向后兼容）/ A bare str return is auto-wrapped as a SUCCESS contract (backward compatible)."""
    from nanoharness.core.tool_executor import ToolResultStatus

    tr = NanoCore._coerce_tool_result("hello world")
    assert tr.status == ToolResultStatus.SUCCESS
    serialized = NanoCore._serialize_tool_result(tr)
    assert serialized == "[status: success] hello world"


def test_tool_result_contract_structured_with_hint():
    """结构化 ToolResult 带 next_action_hint 时，序列化含强指引行 / A structured ToolResult with next_action_hint serializes to include the strong-guidance line."""
    from nanoharness.core.tool_executor import ToolResult, ToolResultStatus

    tr = ToolResult(
        status=ToolResultStatus.SUCCESS,
        content="file written",
        next_action_hint="now run the tests to verify",
    )
    serialized = NanoCore._serialize_tool_result(tr)
    assert serialized.startswith("[status: success] file written")
    assert "[next_action: now run the tests to verify]" in serialized


@pytest.mark.asyncio
async def test_tool_result_failure_sets_success_false():
    """工具返回 ToolResult(FAILURE) 时，ToolResultEvent.success == False（不计入 tools_executed）/ A tool returning ToolResult(FAILURE) yields ToolResultEvent.success == False (not counted in tools_executed)."""
    from nanoharness.core.tool_executor import (
        ToolDefinition,
        ToolRegistry,
        ToolResult,
        ToolResultStatus,
    )

    ctx = make_ctx()
    tool_call = ToolCall(tool_use_id="tc-fail", tool_name="flaky", tool_input={})
    call_count = 0

    async def smart_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raw = [{"type": "tool_use", "id": "tc-fail", "name": "flaky", "input": {}}]
            resp = LLMResponse(raw_content=raw, stop_reason="tool_use", tool_calls=[tool_call])
        else:
            resp = LLMResponse(
                raw_content=[{"type": "text", "text": "giving up."}],
                stop_reason="end_turn", final_text="giving up.",
            )
        yield StreamChunk(is_final=True, final_response=resp)

    provider = MagicMock()
    provider.model_id = "test"
    provider.stream = smart_stream
    provider.count_tokens = MagicMock(return_value=10)

    async def flaky_fn(inputs: dict, ctx):
        return ToolResult(status=ToolResultStatus.FAILURE, content="disk full", error_code="no_space")

    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="flaky", description="always fails",
        input_schema={"type": "object", "properties": {}}, fn=flaky_fn,
    ))

    core = NanoCore(
        ctx=ctx, provider=provider,
        tools=registry.as_fn_dict(),
        tool_definitions=registry.as_api_list(),
        event_store=EventStore(),
    )

    tool_results = []
    async for ev in core.run_turn("try it"):
        if isinstance(ev, ToolResultEvent):
            tool_results.append(ev)

    assert tool_results[0].success is False
    # 序列化内容含 status / error_code 强指引 / serialized content carries status / error_code guidance
    assert "failure" in tool_results[0].output_preview
    assert "no_space" in tool_results[0].output_preview

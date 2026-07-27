"""
Hook 单测 / Hook unit tests：具体 Hook 实现的行为验证。
全部使用 mock，不触发真实 LLM 调用。 / All use mocks; no real LLM calls triggered.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from nanoharness.engine.hooks.defaults import InputSanitizationHook, TurnMetricsHook
from nanoharness.engine.hooks.types import TurnHookContext, TurnHookResult

# ─── 工具函数 / Utility functions ────────────────────────────────────────────

def make_hook_ctx(user_message: str = "hello", model_id: str = "mock") -> TurnHookContext:
    return TurnHookContext(
        session_key="sess-1",
        agent_id="agent-1",
        turn_id="turn-1",
        trace_id="trace-1",
        user_message=user_message,
        model_id=model_id,
    )


def make_hook_result(elapsed_ms: float = 200.0, input_tokens: int = 10, output_tokens: int = 20) -> TurnHookResult:
    return TurnHookResult(
        final_text="ok",
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        elapsed_ms=elapsed_ms,
    )


# ─── InputSanitizationHook 测试 / InputSanitizationHook tests ────────────────

@pytest.mark.asyncio
async def test_input_sanitization_hook_logs_on_oversized_input(caplog):
    """
    超长消息触发 WARNING 日志。 / Oversized message triggers a WARNING log.
    """
    hook = InputSanitizationHook(max_chars=10)
    ctx = make_hook_ctx(user_message="A" * 11)  # 超过上限 / exceeds limit

    with caplog.at_level(logging.WARNING):
        await hook.before_turn(ctx)

    assert any("输入过长" in r.message or "oversized" in r.message for r in caplog.records), \
        "应记录超长 warning / should log an oversized warning"


@pytest.mark.asyncio
async def test_input_sanitization_hook_silent_on_normal_input(caplog):
    """
    正常长度输入不触发任何日志。 / Normal-length input does not trigger any log.
    """
    hook = InputSanitizationHook(max_chars=100)
    ctx = make_hook_ctx(user_message="short message")

    with caplog.at_level(logging.WARNING):
        await hook.before_turn(ctx)

    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_records, "正常输入不应触发 warning / normal input must not trigger warning"


# ─── TurnMetricsHook 测试 / TurnMetricsHook tests ────────────────────────────

@pytest.mark.asyncio
async def test_turn_metrics_hook_records_latency_after_turn():
    """
    after_turn：延迟和 token 写入 metrics。 / after_turn: latency and tokens written to metrics.
    """
    mock_metrics = MagicMock()
    hook = TurnMetricsHook(metrics=mock_metrics)
    ctx = make_hook_ctx(model_id="claude-haiku")
    result = make_hook_result(elapsed_ms=500.0, input_tokens=100, output_tokens=50)

    await hook.after_turn(ctx, result)

    mock_metrics.observe_latency.assert_called_once_with(0.5, kind="turn", model="claude-haiku")
    mock_metrics.inc_tokens.assert_any_call(100, token_type="input")
    mock_metrics.inc_tokens.assert_any_call(50, token_type="output")


@pytest.mark.asyncio
async def test_turn_metrics_hook_records_error_on_error():
    """
    on_error：异常类型写入 metrics。 / on_error: exception type written to metrics.
    """
    mock_metrics = MagicMock()
    hook = TurnMetricsHook(metrics=mock_metrics)
    ctx = make_hook_ctx()

    await hook.on_error(ctx, ValueError("something went wrong"))

    mock_metrics.inc_error.assert_called_once_with(error_type="ValueError")

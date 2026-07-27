"""
Provider 单测 / Provider unit tests：ProviderSelector 重试 + failover 行为验证。
全部使用 mock，不消耗真实 API quota。 / All use mocks; no real API quota consumed.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nanoharness.provider.base import LLMResponse, ProviderError, ProviderErrorType, StreamChunk
from nanoharness.provider.selector import ProviderSelector, RetryConfig, _backoff_delay

# ─── 工具函数 / Utility functions ────────────────────────────────────────────

def _make_ok_provider(model_id: str = "mock-primary") -> MagicMock:
    """返回正常响应的 mock provider。 / Returns a mock provider that always succeeds."""
    resp = LLMResponse(
        raw_content=[{"type": "text", "text": "ok"}],
        stop_reason="end_turn", final_text="ok",
        input_tokens=5, output_tokens=5,
    )
    p = MagicMock()
    p.model_id = model_id
    p.complete = AsyncMock(return_value=resp)
    p.count_tokens = MagicMock(return_value=10)

    async def _stream(*a, **kw):
        yield StreamChunk(is_final=True, final_response=resp)

    p.stream = _stream
    return p


def _make_error_provider(error_type: ProviderErrorType, model_id: str = "mock-primary") -> MagicMock:
    """返回永远抛 ProviderError 的 mock provider。 / Returns a mock provider that always raises ProviderError."""
    exc = ProviderError(error_type, f"mock {error_type}", retryable=error_type in (
        ProviderErrorType.RATE_LIMITED, ProviderErrorType.SERVER_ERROR, ProviderErrorType.TIMEOUT,
    ))
    p = MagicMock()
    p.model_id = model_id
    p.complete = AsyncMock(side_effect=exc)
    p.count_tokens = MagicMock(return_value=10)

    async def _stream(*a, **kw):
        raise exc
        yield  # 让 Python 把它解析成 async generator

    p.stream = _stream
    return p


# ─── 测试 / Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_selector_retries_on_rate_limit_then_succeeds():
    """
    RATE_LIMITED 重试后成功：前 N-1 次抛 RATE_LIMITED，第 N 次成功。
    / Retries on RATE_LIMITED, succeeds on the N-th attempt.
    """
    call_count = 0
    resp = LLMResponse(
        raw_content=[{"type": "text", "text": "ok"}],
        stop_reason="end_turn", final_text="ok",
        input_tokens=5, output_tokens=5,
    )

    async def _flaky_complete(*a, **kw):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderError(ProviderErrorType.RATE_LIMITED, "429", retryable=True)
        return resp

    primary = MagicMock()
    primary.model_id = "mock"
    primary.complete = _flaky_complete

    cfg = RetryConfig(max_attempts=3, base_delay_s=0.0)  # delay=0 让测试不阻塞 / delay=0 so test doesn't block
    sel = ProviderSelector(primary=primary, retry=cfg)
    result = await sel.complete(system="s", messages=[])

    assert result.final_text == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_selector_failover_to_backup_when_primary_exhausted():
    """
    主 provider 重试全部耗尽后切换到 fallback provider。 / After primary exhausts all retries, fall over to backup.
    """
    primary = _make_error_provider(ProviderErrorType.RATE_LIMITED, "mock-primary")
    backup = _make_ok_provider("mock-backup")

    cfg = RetryConfig(max_attempts=2, base_delay_s=0.0)
    sel = ProviderSelector(primary=primary, fallbacks=[backup], retry=cfg)
    result = await sel.complete(system="s", messages=[])

    assert result.final_text == "ok"
    assert backup.complete.called, "fallback provider 应该被调用 / fallback provider should have been called"


@pytest.mark.asyncio
async def test_selector_no_retry_on_auth_error():
    """
    AUTH_INVALID 不重试，直接 re-raise。 / AUTH_INVALID is immediately re-raised without retrying.
    """
    call_count = 0

    async def _auth_fail(*a, **kw):
        nonlocal call_count
        call_count += 1
        raise ProviderError(ProviderErrorType.AUTH_INVALID, "401", retryable=False)

    primary = MagicMock()
    primary.model_id = "mock"
    primary.complete = _auth_fail

    sel = ProviderSelector(primary=primary, retry=RetryConfig(max_attempts=3, base_delay_s=0.0))
    with pytest.raises(ProviderError) as exc_info:
        await sel.complete(system="s", messages=[])

    assert exc_info.value.error_type == ProviderErrorType.AUTH_INVALID
    assert call_count == 1, "AUTH_INVALID 不应重试 / AUTH_INVALID must not retry"


@pytest.mark.asyncio
async def test_selector_context_too_long_passthrough():
    """
    CONTEXT_TOO_LONG 立即 re-raise，不重试，不 failover。 / CONTEXT_TOO_LONG is immediately re-raised; no retry, no failover.

    CONTEXT_TOO_LONG 是压缩信号，必须透传给 NanoCore._call_provider 处理。 /
    CONTEXT_TOO_LONG is a compaction signal; it must reach NanoCore._call_provider unchanged.
    """
    call_count = 0

    async def _ctx_overflow(*a, **kw):
        nonlocal call_count
        call_count += 1
        raise ProviderError(ProviderErrorType.CONTEXT_TOO_LONG, "context_length", retryable=False)

    primary = MagicMock()
    primary.model_id = "mock"
    primary.complete = _ctx_overflow
    backup = _make_ok_provider("mock-backup")

    sel = ProviderSelector(primary=primary, fallbacks=[backup], retry=RetryConfig(max_attempts=3, base_delay_s=0.0))
    with pytest.raises(ProviderError) as exc_info:
        await sel.complete(system="s", messages=[])

    assert exc_info.value.error_type == ProviderErrorType.CONTEXT_TOO_LONG
    assert call_count == 1
    assert not backup.complete.called, "CONTEXT_TOO_LONG 不应 failover / CONTEXT_TOO_LONG must not failover"


def test_backoff_delay_is_exponential():
    """
    指数退避：attempt 越大 delay 越长，且不超过 max_delay。 / Exponential backoff: delay grows with attempt and is capped at max_delay.
    """
    cfg = RetryConfig(base_delay_s=1.0, max_delay_s=10.0, jitter_factor=0.0)  # jitter=0 使测试确定性 / jitter=0 for determinism

    delays = [_backoff_delay(i, cfg) for i in range(4)]

    assert delays[0] == pytest.approx(1.0)   # base
    assert delays[1] == pytest.approx(2.0)   # base×2
    assert delays[2] == pytest.approx(4.0)   # base×4
    assert delays[3] == pytest.approx(8.0)   # base×8

    # max_delay 上限 / max_delay cap
    cfg_small = RetryConfig(base_delay_s=1.0, max_delay_s=3.0, jitter_factor=0.0)
    assert _backoff_delay(5, cfg_small) == pytest.approx(3.0)

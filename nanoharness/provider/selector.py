from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator

from nanoharness.core.context import Message
from nanoharness.provider.base import (
    LLMProvider,
    LLMResponse,
    ProviderError,
    ProviderErrorType,
    StreamChunk,
)

log = logging.getLogger(__name__)


# ─── 重试配置 / Retry config ──────────────────────────────────────────────────

@dataclass
class RetryConfig:
    """
    指数退避重试参数。 / Exponential backoff retry parameters.

    面试话术 / Interview talking point:
    "base×2^attempt 给出指数增长，加 0~25% 随机抖动（jitter）防止
    多个 client 同时被限速后在同一时刻集体重试（thundering herd）。
    max_delay 防止延迟无限增长，3 次重试对 RATE_LIMITED 已经够用。"
    """
    max_attempts: int = 3           # 单 provider 最多重试次数（含首次） / max attempts per provider (including first)
    base_delay_s: float = 1.0       # 首次重试等待基准（秒） / base wait before first retry (seconds)
    max_delay_s: float = 30.0       # 退避上限（秒） / backoff ceiling (seconds)
    jitter_factor: float = 0.25     # 抖动比例：delay *= (1 + jitter_factor * rand) / jitter ratio


def _backoff_delay(attempt: int, cfg: RetryConfig) -> float:
    """
    计算第 attempt 次重试前的等待时间（含 jitter）。 / Compute wait time before the (attempt)-th retry (with jitter).

    delay = min(base × 2^attempt, max) × (1 + jitter_factor × rand[0,1))
    attempt=0 → base; attempt=1 → 2×base; attempt=2 → 4×base; ...
    """
    base = min(cfg.base_delay_s * (2 ** attempt), cfg.max_delay_s)
    jitter = base * cfg.jitter_factor * random.random()
    return base + jitter


# ─── 不重试的错误类型 / Non-retryable error types ────────────────────────────────────────────────────

# AUTH_INVALID：密钥问题，重试无意义 / AUTH_INVALID: key issue, retrying is pointless
# CONTEXT_TOO_LONG：触发压缩信号，需在 _call_provider 层处理，不能在这里吞掉 / CONTEXT_TOO_LONG: compaction signal, must not be swallowed here
_NO_RETRY_TYPES = frozenset({ProviderErrorType.AUTH_INVALID, ProviderErrorType.CONTEXT_TOO_LONG})


# ─── ProviderSelector ────────────────────────────────────────────────────────

class ProviderSelector:
    """
    在单一 provider 上做指数退避重试，可选地在多个 provider 间做 failover。 / Applies exponential-backoff retries to a single provider and optionally fails over across multiple providers.

    实现 LLMProvider Protocol，对上层透明：TurnRunner 无需感知重试逻辑。 / Implements LLMProvider Protocol — transparent to TurnRunner; retry logic lives here.

    重试策略（每个 provider 独立）： / Retry strategy (per-provider):
      RATE_LIMITED / SERVER_ERROR / TIMEOUT → 最多 max_attempts 次，指数退避 + jitter
      AUTH_INVALID                          → 立即 re-raise（密钥无效，重试无效）
      CONTEXT_TOO_LONG                      → 立即 re-raise（压缩信号，透传给 NanoCore）
      重试耗尽 → 换下一个 fallback provider / retry exhausted → try next fallback provider

    面试话术 / Interview talking point:
    "ProviderError.retryable 标记在 provider/base.py 早就定义了，但 Phase 1/2 都没接通。
    ProviderSelector 是把这个死代码修活的地方——它实现 LLMProvider Protocol，
    上层代码（TurnRunner）感知不到它的存在，原始 provider 和 ProviderSelector 完全互换。"
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: list[LLMProvider] | None = None,
        retry: RetryConfig | None = None,
    ) -> None:
        self._primary = primary
        self._fallbacks = fallbacks or []
        self._retry = retry or RetryConfig()

    @property
    def model_id(self) -> str:
        # 暴露主 provider 的 model_id，保持协议兼容 / Expose primary's model_id for protocol compatibility
        return self._primary.model_id

    def count_tokens(self, messages: list[Message], system: str = "") -> int:
        return self._primary.count_tokens(messages, system)

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        last_exc: ProviderError | None = None
        for provider in [self._primary] + self._fallbacks:
            for attempt in range(self._retry.max_attempts):
                try:
                    return await provider.complete(system, messages, tools=tools, max_tokens=max_tokens)
                except ProviderError as exc:
                    last_exc = exc
                    if exc.error_type in _NO_RETRY_TYPES:
                        raise
                    if attempt < self._retry.max_attempts - 1:
                        delay = _backoff_delay(attempt, self._retry)
                        log.warning(
                            "provider %s 第 %d 次重试（%s），等待 %.1fs / provider %s attempt %d (%s), waiting %.1fs",
                            provider.model_id, attempt + 1, exc.error_type, delay,
                            provider.model_id, attempt + 1, exc.error_type, delay,
                        )
                        await asyncio.sleep(delay)
            if self._fallbacks and provider is not self._fallbacks[-1]:
                log.warning(
                    "主 provider %s 重试耗尽，切换 fallback / primary provider %s exhausted, switching to fallback",
                    provider.model_id, provider.model_id,
                )
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        """
        流式调用带重试。 / Streaming call with retry.

        注意：流式响应一旦开始产出 chunk 就不能回滚，所以重试只在首个 chunk 之前的
        连接阶段触发——provider 抛出 ProviderError 而非在 yield 途中断流时重试。 /
        Note: once chunks start streaming they cannot be rolled back — retry fires only before
        the first chunk, on connect-phase ProviderErrors, not mid-stream breaks.
        """
        last_exc: ProviderError | None = None
        for provider in [self._primary] + self._fallbacks:
            for attempt in range(self._retry.max_attempts):
                try:
                    async for chunk in provider.stream(system, messages, tools=tools, max_tokens=max_tokens):
                        yield chunk
                    return  # 成功完成 / completed successfully
                except ProviderError as exc:
                    last_exc = exc
                    if exc.error_type in _NO_RETRY_TYPES:
                        raise
                    if attempt < self._retry.max_attempts - 1:
                        delay = _backoff_delay(attempt, self._retry)
                        log.warning(
                            "provider %s stream 第 %d 次重试（%s），等待 %.1fs / provider %s stream attempt %d (%s), waiting %.1fs",
                            provider.model_id, attempt + 1, exc.error_type, delay,
                            provider.model_id, attempt + 1, exc.error_type, delay,
                        )
                        await asyncio.sleep(delay)
            if self._fallbacks and provider is not self._fallbacks[-1]:
                log.warning(
                    "provider %s 流式重试耗尽，切换 fallback / provider %s stream exhausted, switching to fallback",
                    provider.model_id, provider.model_id,
                )
        assert last_exc is not None
        raise last_exc

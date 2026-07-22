from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

# ─── 工具注册表 / Tool Registry ────────────────────────────────────────────────

ToolCallable = Callable[[dict[str, Any], Any], Any]  # （输入，ToolContext）-> Any / (input, ToolContext) -> Any


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: ToolCallable

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """
    简单的 名称 → ToolDefinition 映射 / Simple name → ToolDefinition map.
    被 NanoCore 用来构建 tool_definitions 并路由调用 / Used by NanoCore to build tool_definitions and route calls.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_fn(self, name: str) -> ToolCallable | None:
        t = self._tools.get(name)
        return t.fn if t else None

    def as_fn_dict(self) -> dict[str, ToolCallable]:
        return {name: t.fn for name, t in self._tools.items()}

    def as_api_list(self) -> list[dict[str, Any]]:
        return [t.to_api_dict() for t in self._tools.values()]


# ─── 重试配置 / Retry Config ─────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 0.5     # 秒 / seconds
    max_delay: float = 30.0
    jitter: float = 0.25        # 休眠时间的 ±jitter 比例 / ±jitter fraction of sleep time


# ─── 工具返回契约 / Tool Result Contract ─────────────────────────────────────


class ToolResultStatus:
    """工具返回的状态枚举（字符串常量，避免 StrEnum 依赖）/ Tool return status as string constants."""
    SUCCESS = "success"   # 成功 / succeeded
    FAILURE = "failure"   # 失败（可重试或换方法）/ failed (retry or change approach)
    PENDING = "pending"   # 未定（异步未完成）/ pending (async incomplete)


@dataclass
class ToolResult:
    """
    工具的可判定返回契约 / Decidable return contract for tools.

    死循环常因工具返回模糊（null / 空串 / 大段 stack trace）使 LLM 无法判断下一步 / Dead loops often stem from vague tool returns (null / empty / stack trace) leaving the LLM unable to decide next step.
    要求工具明确返回 status 与可选的 next_action_hint：即使 LLM 长上下文失忆，最新 Observation 里的强指引也能把它拉回正轨 / Require explicit status and optional next_action_hint: even if the LLM forgets long context, the strong hint in the latest observation pulls it back on track.
    老工具返回裸 str 会被自动包装，向后兼容 / Legacy tools returning a bare str are auto-wrapped, backward compatible.
    """
    status: str = ToolResultStatus.SUCCESS
    content: str = ""
    error_code: str = ""
    next_action_hint: str = ""


# ─── 卡死检测器 / Stuck Detector ─────────────────────────────────────────────

@dataclass
class StuckDecision:
    """
    StuckDetector.observe 的返回值 / Return value of StuckDetector.observe.
    触发时携带工具名与该签名的累计计数 / When stuck, carries the tool name and the signature's cumulative count.
    不抛异常——调用方据此决定是注入恢复消息还是硬停 / Does NOT raise — the caller decides whether to inject a recovery message or hard-stop.
    """
    stuck: bool = False
    tool_name: str = ""
    signature: str = ""
    count: int = 0


class StuckDetector:
    """
    整轮 per-签名计数卡死检测 / Per-signature stuck detector over a whole turn.

    记录每个 (tool_name, input_hash) 签名在本轮被请求的次数，≥ threshold 即判定卡死 / Tracks how many times each (tool_name, input_hash) signature was requested this turn; fires at ≥ threshold.

    相比"只看连续相同"的旧实现，per-签名计数还能 catch 振荡（A-B-A-B-A-B：A、B 各自累计到阈值）/ Compared to a "consecutive-identical-only" impl, per-signature counting also catches oscillation (A-B-A-B-A-B: both A and B reach the threshold).

    面试话术 / Interview talking point:
    "A common failure mode is the model looping on the same tool — either the exact
    same call or an A/B/A/B oscillation. I fingerprint calls by tool+input hash and
    count per-signature across the whole turn, firing a recovery injection (not a
    hard abort) once a signature's count crosses the threshold."
    """

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._counts: dict[str, int] = {}

    def observe(self, tool_name: str, tool_input: dict[str, Any]) -> StuckDecision:
        """
        记录一次工具调用请求并返回是否判定卡死 / Record a tool-call request and return whether it's judged stuck.
        触发后不再累加（避免重复注入）/ After firing, stop incrementing that signature (avoid double injection).
        """
        sig = self._fingerprint(tool_name, tool_input)
        self._counts[sig] = self._counts.get(sig, 0) + 1
        count = self._counts[sig]
        if count >= self._threshold:
            return StuckDecision(stuck=True, tool_name=tool_name, signature=sig, count=count)
        return StuckDecision(stuck=False, tool_name=tool_name, signature=sig, count=count)

    @staticmethod
    def _fingerprint(tool_name: str, tool_input: dict[str, Any]) -> str:
        payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ─── 工具执行器 / Tool Executor ────────────────────────────────────────────────

class ToolExecutor:
    """
    执行工具调用，特性如下 / Executes tool calls with:
      - 单次调用超时（asyncio.wait_for）/ per-call timeout (asyncio.wait_for)
      - 瞬时错误上的指数退避加抖动 / exponential backoff with jitter on transient errors
      - 卡死检测（同一签名累计 N 次 → 返回失败而非 raise）/ stuck detection (same signature N times → returns failure instead of raising)

    设计为 turn 间无状态 — 调用方每轮新建实例并传入同一个 ToolRegistry / Designed to be stateless between turns — callers create a fresh instance per turn and pass the same ToolRegistry.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = 30.0,
        retry: RetryConfig | None = None,
        stuck_threshold: int = 3,
    ) -> None:
        self._registry = registry
        self._timeout = timeout
        self._retry = retry or RetryConfig()
        self._stuck_detector = StuckDetector(stuck_threshold)

    async def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_context: Any,
    ) -> tuple[bool, str]:
        """
        返回 (success, output_str) / Returns (success, output_str).
        若同一签名累计过多则返回 (False, "stuck_loop_detected") 而非抛错 / Returns (False, "stuck_loop_detected") when a signature's count crosses the threshold (no exception).
        """
        fn = self._registry.get_fn(tool_name)
        if fn is None:
            return False, f"Unknown tool: '{tool_name}'"

        decision = self._stuck_detector.observe(tool_name, tool_input)
        if decision.stuck:
            return False, f"stuck_loop_detected: tool '{tool_name}' signature repeated {decision.count} times"

        attempt = 0
        last_exc: Exception | None = None

        while attempt < self._retry.max_attempts:
            try:
                raw = await asyncio.wait_for(
                    self._run_fn(fn, tool_input, tool_context),
                    timeout=self._timeout,
                )
                output = str(raw) if not isinstance(raw, str) else raw
                return True, output

            except asyncio.TimeoutError:
                return False, f"Tool '{tool_name}' timed out after {self._timeout}s."

            except Exception as exc:
                last_exc = exc
                if not self._is_transient(exc):
                    log.warning("tool %s non-transient error: %s", tool_name, exc)
                    return False, f"Tool error: {exc}"

                attempt += 1
                sleep = min(
                    self._retry.base_delay * (2 ** (attempt - 1)),
                    self._retry.max_delay,
                )
                jitter = sleep * self._retry.jitter * (2 * random.random() - 1)
                sleep = max(0.0, sleep + jitter)
                log.debug("tool %s attempt %d failed, retry in %.1fs", tool_name, attempt, sleep)
                await asyncio.sleep(sleep)

        return False, f"Tool '{tool_name}' failed after {self._retry.max_attempts} attempts: {last_exc}"

    @staticmethod
    async def _run_fn(fn: ToolCallable, tool_input: dict[str, Any], ctx: Any) -> Any:
        result = fn(tool_input, ctx)
        if asyncio.iscoroutine(result):
            return await result
        return result

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ("timeout", "connection", "temporary", "unavailable"))

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

# ─── Tool Registry ────────────────────────────────────────────────────────────

ToolCallable = Callable[[dict[str, Any], Any], Any]  # (input, ToolContext) -> Any


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
    Simple name → ToolDefinition map.
    Used by NanoCore to build tool_definitions and route calls.
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


# ─── Retry Config ─────────────────────────────────────────────────────────────

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 0.5     # seconds
    max_delay: float = 30.0
    jitter: float = 0.25        # ±jitter fraction of sleep time


# ─── Dead-loop Detector ───────────────────────────────────────────────────────

class DeadLoopDetector:
    """
    Detects repeated identical tool calls within a turn.

    If the same (tool_name, input_hash) pair appears ≥ threshold times
    consecutively, raise to force the model to stop looping.

    Interview talking point:
    "A common failure mode is the model calling the same tool in a loop when
    it doesn't understand the result. I fingerprint calls by tool+input hash
    and abort after N consecutive repeats."
    """

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold
        self._recent: deque[str] = deque(maxlen=threshold)

    def check(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        key = self._fingerprint(tool_name, tool_input)
        self._recent.append(key)
        if len(self._recent) == self._threshold and len(set(self._recent)) == 1:
            raise DeadLoopError(
                f"Tool '{tool_name}' called {self._threshold} times in a row "
                f"with identical inputs. Aborting to prevent infinite loop."
            )

    @staticmethod
    def _fingerprint(tool_name: str, tool_input: dict[str, Any]) -> str:
        payload = json.dumps({"tool": tool_name, "input": tool_input}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class DeadLoopError(RuntimeError):
    pass


# ─── Tool Executor ────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Executes tool calls with:
      - per-call timeout (asyncio.wait_for)
      - exponential backoff with jitter on transient errors
      - dead-loop detection (same call repeated N times → abort)

    Designed to be stateless between turns — callers create a fresh instance
    per turn and pass the same ToolRegistry.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float = 30.0,
        retry: RetryConfig | None = None,
        dead_loop_threshold: int = 3,
    ) -> None:
        self._registry = registry
        self._timeout = timeout
        self._retry = retry or RetryConfig()
        self._loop_detector = DeadLoopDetector(dead_loop_threshold)

    async def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_context: Any,
    ) -> tuple[bool, str]:
        """
        Returns (success, output_str).
        Raises DeadLoopError if the same call repeats too many times.
        """
        fn = self._registry.get_fn(tool_name)
        if fn is None:
            return False, f"Unknown tool: '{tool_name}'"

        self._loop_detector.check(tool_name, tool_input)

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

            except DeadLoopError:
                raise

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

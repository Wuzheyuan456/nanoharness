from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from nanoharness.core.context import AgentContext, AgentState, Message, TurnContext
from nanoharness.core.event_store import (
    AgentEvent,
    CompactionEvent,
    DoneEvent,
    ErrorEvent,
    StateChangeEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from nanoharness.provider.base import LLMProvider, ProviderError, ProviderErrorType

log = logging.getLogger(__name__)


class NanoCore:
    """
    手写的 ReAct 状态机 / Hand-written ReAct state machine.

    单一职责：驱动 THINKING → TOOL_CALLING 循环，直到模型产出最终答案或超过 max_iter / max_tool_calls / Single responsibility: drive the THINKING → TOOL_CALLING loop until the model either produces a final answer or exceeds max_iter / max_tool_calls.

    所有副作用（记忆、钩子、路由）都位于本类之外，在构造时注入。NanoCore 绝不从 channels、memory 或 agents 层导入，只从 core.* 和 provider.* 导入 / All side-effects (memory, hooks, routing) live OUTSIDE this class and are injected at construction time. NanoCore never imports from channels, memory, or agents layers — only from core.* and provider.*.

    用法 / Usage:
        async for event in nano.run_turn("what's the weather?"):
            print(event)
    """

    MAX_ITER = 20         # ReAct 迭代次数的硬上限 / hard ceiling on ReAct iterations
    MAX_TOOL_CALLS = 40   # 工具调用总数的独立上限 / separate ceiling on total tool invocations
    TOOL_TIMEOUT = 30.0   # 单次工具调用被中止前的秒数 / seconds before a single tool call is aborted

    def __init__(
        self,
        ctx: AgentContext,
        provider: LLMProvider,
        tools: dict[str, Any],        # 名称 → async 可调用对象 / name → async callable
        tool_definitions: list[dict[str, Any]],  # Anthropic tool_definitions 格式 / Anthropic tool_definitions format
        compaction: Any | None = None, # CompactionEngine（注入，避免循环导入） / CompactionEngine (injected, avoids circular import)
        event_store: Any | None = None,
        max_iter: int = MAX_ITER,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        self._ctx = ctx
        self._provider = provider
        self._tools = tools
        self._tool_defs = tool_definitions
        self._compaction = compaction
        self._event_store = event_store
        self._max_iter = max_iter
        self._max_tool_calls = max_tool_calls

    # ── 公共入口 / Public entry point ────────────────────────────────────────

    async def run_turn(self, user_message: str) -> AsyncIterator[AgentEvent]:
        """
        异步生成器，按发生顺序产出 AgentEvents / Async generator.  Yields AgentEvents as they occur.
        调用方驱动生成器；流式文本以 TextDeltaEvent 到达 / Caller drives the generator; streaming text arrives as TextDeltaEvent.
        最终事件总是 DoneEvent 或 ErrorEvent / Final event is always DoneEvent or ErrorEvent.
        """
        turn = TurnContext(
            session_key=self._ctx.session_key,
            agent_id=self._ctx.agent_id,
        )

        self._ctx.append_message(Message(role="user", content=user_message))

        try:
            async for event in self._react_loop(turn):
                yield event
        except Exception as exc:
            error_event = ErrorEvent(
                trace_id=turn.trace_id,
                session_key=turn.session_key,
                agent_id=turn.agent_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                recoverable=False,
            )
            self._emit(error_event)
            yield error_event

    # ── ReAct 循环 / ReAct loop ───────────────────────────────────────────────

    async def _react_loop(self, turn: TurnContext) -> AsyncIterator[AgentEvent]:
        while turn.iterations < self._max_iter:
            # ── 思考 / THINKING ──────────────────────────────────────────────
            yield self._transition(turn, AgentState.THINKING)

            # 预检：若正在耗尽上下文窗口则压缩 / Preflight: compact if we're burning through the context window
            if self._should_compact(turn):
                async for ev in self._do_compact(turn):
                    yield ev

            final_text = ""
            async for chunk_event in self._call_provider(turn):
                yield chunk_event
                if isinstance(chunk_event, TextDeltaEvent):
                    final_text += chunk_event.delta

            # _call_provider 把响应存在 turn 上，这里取回 / _call_provider stores the response on turn; retrieve it
            response = turn._last_response  # type: ignore[attr-defined]
            if response is None:
                break

            # ── 工具调用 / TOOL CALLING ───────────────────────────────────────
            if response.wants_tool_call:
                if turn.tool_call_count >= self._max_tool_calls:
                    log.warning("max_tool_calls hit, forcing final answer")
                    break

                yield self._transition(turn, AgentState.TOOL_CALLING)
                self._ctx.append_message(response.to_assistant_message())

                tool_result_blocks: list[dict[str, Any]] = []
                for tc in response.tool_calls:
                    async for ev in self._execute_tool(turn, tc):
                        yield ev
                        if isinstance(ev, ToolResultEvent):
                            tool_result_blocks.append(
                                self._make_tool_result_block(tc.tool_use_id, ev)
                            )

                self._ctx.append_message(Message(
                    role="user",
                    content=tool_result_blocks,
                ))
                turn.iterations += 1
                continue  # 下一轮迭代 → 带着工具结果进入 THINKING / next iteration → THINKING with tool results in history

            # ── 完成 — 模型产出最终答案 / DONE — model produced final answer ────
            self._ctx.append_message(response.to_assistant_message())
            self._ctx.total_input_tokens += response.input_tokens
            self._ctx.total_output_tokens += response.output_tokens

            done = DoneEvent(
                trace_id=turn.trace_id,
                session_key=turn.session_key,
                agent_id=turn.agent_id,
                final_text=response.final_text,
                total_input_tokens=self._ctx.total_input_tokens,
                total_output_tokens=self._ctx.total_output_tokens,
                total_tool_calls=self._ctx.total_tool_calls,
                elapsed_ms=turn.elapsed_ms(),
            )
            self._emit(done)
            yield self._transition(turn, AgentState.DONE)
            yield done
            return

        # 跳出循环 — max_iter 超限 / Fell out of loop — max_iter exceeded
        yield self._force_done(turn, "max_iterations_exceeded")

    # ── Provider 调用（流式） / Provider call (streaming) ─────────────────────

    async def _call_provider(self, turn: TurnContext) -> AsyncIterator[AgentEvent]:
        turn._last_response = None  # type: ignore[attr-defined]
        try:
            async for chunk in self._provider.stream(
                system=self._ctx.system_prompt,
                messages=self._ctx.history,
                tools=self._tool_defs or None,
            ):
                if chunk.delta_text:
                    ev = TextDeltaEvent(
                        trace_id=turn.trace_id,
                        session_key=turn.session_key,
                        agent_id=turn.agent_id,
                        delta=chunk.delta_text,
                    )
                    self._emit(ev)
                    yield ev

                if chunk.is_final and chunk.final_response:
                    turn._last_response = chunk.final_response  # type: ignore[attr-defined]
                    self._ctx.total_input_tokens += chunk.final_response.input_tokens
                    self._ctx.total_output_tokens += chunk.final_response.output_tokens

        except ProviderError as exc:
            if exc.error_type == ProviderErrorType.CONTEXT_TOO_LONG and not turn.has_compacted:
                # 紧急压缩 — 调用中上下文窗口超限 / Emergency compaction — context window exceeded mid-call
                async for ev in self._do_compact(turn):
                    yield ev
                # 压缩后重试一次 / Retry once after compaction
                async for ev in self._call_provider(turn):
                    yield ev
            else:
                raise

    # ── 工具执行 / Tool execution ────────────────────────────────────────────

    async def _execute_tool(self, turn: TurnContext, tc: Any) -> AsyncIterator[AgentEvent]:
        import time

        call_ev = ToolCallEvent(
            trace_id=turn.trace_id,
            session_key=turn.session_key,
            agent_id=turn.agent_id,
            tool_name=tc.tool_name,
            tool_use_id=tc.tool_use_id,
            input_summary=str(tc.tool_input)[:200],
        )
        self._emit(call_ev)
        yield call_ev
        turn.tool_call_count += 1
        self._ctx.total_tool_calls += 1

        fn = self._tools.get(tc.tool_name)
        t0 = time.monotonic()
        success = True
        output = ""

        if fn is None:
            success = False
            output = f"Tool '{tc.tool_name}' not found."
        else:
            try:
                raw = await asyncio.wait_for(
                    fn(tc.tool_input, turn.to_tool_context()),
                    timeout=self.TOOL_TIMEOUT,
                )
                output = str(raw) if not isinstance(raw, str) else raw
            except asyncio.TimeoutError:
                success = False
                output = f"Tool '{tc.tool_name}' timed out after {self.TOOL_TIMEOUT}s."
            except Exception as exc:
                success = False
                output = f"Tool error: {exc}"
                log.warning("tool %s raised: %s", tc.tool_name, exc)

        latency = (time.monotonic() - t0) * 1000
        result_ev = ToolResultEvent(
            trace_id=turn.trace_id,
            session_key=turn.session_key,
            agent_id=turn.agent_id,
            tool_name=tc.tool_name,
            tool_use_id=tc.tool_use_id,
            success=success,
            latency_ms=latency,
            output_preview=output[:200],
        )
        # 把完整输出暂存在事件上，供 _make_tool_result_block 读取 / stash full output on event so _make_tool_result_block can read it
        result_ev._full_output = output  # type: ignore[attr-defined]
        self._emit(result_ev)
        yield result_ev

    # ── 压缩 / Compaction ────────────────────────────────────────────────────

    def _should_compact(self, turn: TurnContext) -> bool:
        if turn.has_compacted or self._compaction is None:
            return False
        # 在模型上下文窗口的 80% 处触发（估算值） / Trigger at 80% of the model's context window (estimated)
        approx = self._ctx.approximate_token_count()
        limit = getattr(self._compaction, "context_window_limit", 180_000)
        return approx > limit * 0.80

    async def _do_compact(self, turn: TurnContext) -> AsyncIterator[AgentEvent]:
        if self._compaction is None or turn.has_compacted:
            return
        turn.has_compacted = True
        tokens_before = self._ctx.approximate_token_count()

        await self._compaction.compact(self._ctx)

        tokens_after = self._ctx.approximate_token_count()
        ev = CompactionEvent(
            trace_id=turn.trace_id,
            session_key=turn.session_key,
            agent_id=turn.agent_id,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        self._emit(ev)
        yield ev

    # ── 辅助方法 / Helpers ───────────────────────────────────────────────────

    def _transition(self, turn: TurnContext, new_state: AgentState) -> StateChangeEvent:
        ev = StateChangeEvent(
            trace_id=turn.trace_id,
            session_key=turn.session_key,
            agent_id=turn.agent_id,
            from_state=str(turn.state),
            to_state=str(new_state),
        )
        turn.state = new_state
        self._emit(ev)
        return ev

    def _force_done(self, turn: TurnContext, reason: str) -> DoneEvent:
        ev = DoneEvent(
            trace_id=turn.trace_id,
            session_key=turn.session_key,
            agent_id=turn.agent_id,
            final_text=f"[Stopped: {reason}]",
            elapsed_ms=turn.elapsed_ms(),
        )
        self._emit(ev)
        return ev

    def _emit(self, event: AgentEvent) -> None:
        if self._event_store is not None:
            self._event_store.append(event)

    @staticmethod
    def _make_tool_result_block(tool_use_id: str, ev: ToolResultEvent) -> dict[str, Any]:
        full = getattr(ev, "_full_output", ev.output_preview)
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": full,
            "is_error": not ev.success,
        }

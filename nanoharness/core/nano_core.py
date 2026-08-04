from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from nanoharness.core.context import (
    AgentContext,
    AgentState,
    Message,
    StopReason,
    TurnContext,
    TurnOutcome,
    classify_outcome,
)
from nanoharness.core.event_store import (
    AgentEvent,
    CompactionEvent,
    DoneEvent,
    ErrorEvent,
    InterventionEvent,
    StateChangeEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from nanoharness.core.tool_executor import StuckDetector, ToolResult, ToolResultStatus
from nanoharness.provider.base import LLMProvider, ProviderError, ProviderErrorType

log = logging.getLogger(__name__)


# ── 注入给模型的操作指令（非注释，英文以匹配 LLM 指令惯例）/ Directives injected into the model (not comments; English per LLM-instruction convention) ──
# 两阶段收尾：预算撞线时注入，要求模型不再调工具直接给最终答案 / two-phase finalization: injected when a budget is hit, asks the model to answer without tools
_FINALIZATION_DIRECTIVE = (
    "The configured iteration/tool-call limit has been reached. "
    "Do not call any more tools. Give the best concise final answer from the work done so far."
)
# 卡死恢复：重复签名或单工具预算耗尽时注入，要求换方法 / stuck recovery: injected on repeated signature or per-tool budget exhaustion, asks the model to change approach
_STUCK_RECOVERY_DIRECTIVE = (
    "Repeated or budget-exhausted tool calls were detected and the tool has been disabled for the rest of this turn. "
    "Do not retry the same approach. Either switch to a different tool/approach or give a final answer based on current information."
)


class NanoCore:
    """
    手写的 ReAct 状态机 / Hand-written ReAct state machine.

    单一职责：驱动 THINKING → TOOL_CALLING 循环，直到模型产出最终答案或撞预算（max_iter / max_tool_calls / per-tool 预算 / 卡死检测）/ Single responsibility: drive the THINKING → TOOL_CALLING loop until the model produces a final answer or hits a budget (max_iter / max_tool_calls / per-tool budget / stuck detection).

    所有副作用（记忆、钩子、路由）都位于本类之外，在构造时注入。NanoCore 绝不从 channels、memory 或 agents 层导入，只从 core.* 和 provider.* 导入 / All side-effects (memory, hooks, routing) live OUTSIDE this class and are injected at construction time. NanoCore never imports from channels, memory, or agents layers — only from core.* and provider.*.

    用法 / Usage:
        async for event in nano.run_turn("what's the weather?"):
            print(event)
    """

    MAX_ITER = 20         # ReAct 迭代次数的硬上限 / hard ceiling on ReAct iterations
    MAX_TOOL_CALLS = 40   # 工具调用总数的独立上限 / separate ceiling on total tool invocations
    TOOL_TIMEOUT = 30.0   # 单次工具调用被中止前的秒数 / seconds before a single tool call is aborted
    STUCK_THRESHOLD = 3        # 同签名累计 N 次即判卡死 / signature count to judge stuck
    MAX_CALLS_PER_TOOL = 5     # 单工具调用预算（catch 不同参数钻空子）/ per-tool call budget (catches varying-args evasion)

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
        max_calls_per_tool: int = MAX_CALLS_PER_TOOL,
        stuck_threshold: int = STUCK_THRESHOLD,
    ) -> None:
        self._ctx = ctx
        self._provider = provider
        self._tools = tools
        self._tool_defs = tool_definitions
        self._compaction = compaction
        self._event_store = event_store
        self._max_iter = max_iter
        self._max_tool_calls = max_tool_calls
        self._max_calls_per_tool = max_calls_per_tool
        self._stuck = StuckDetector(stuck_threshold)

    # ── 公共属性 / Public properties ─────────────────────────────────────────

    @property
    def ctx(self) -> AgentContext:
        return self._ctx

    def swap_tools(
        self,
        tools: dict[str, Any],
        tool_definitions: list[dict[str, Any]],
    ) -> None:
        """
        热换工具集（下一个 run_turn 生效）/ Hot-swap the tool set (effective on next run_turn call).
        asyncio 单线程，turn 间调用无竞态 / asyncio single-thread; safe to call between turns.
        """
        self._tools = tools
        self._tool_defs = tool_definitions

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
        while True:
            # ── 预算检查（含两阶段优雅收尾）/ budget checks (with two-phase graceful finalization) ──
            budget_reason = self._budget_exhausted(turn)
            if budget_reason is not None:
                async for ev in self._handle_budget_exhausted(turn, budget_reason):
                    yield ev
                return  # _handle_budget_exhausted 总是以 DoneEvent 结束并返回 / _handle_budget_exhausted always ends with a DoneEvent and returns

            # ── 思考 / THINKING ──────────────────────────────────────────────
            yield self._transition(turn, AgentState.THINKING)

            # 预检：若正在耗尽上下文窗口则压缩 / Preflight: compact if we're burning through the context window
            if self._should_compact(turn):
                async for ev in self._do_compact(turn):
                    yield ev

            final_text = ""
            async for chunk_event in self._call_provider(turn, self._visible_tool_defs(turn)):
                yield chunk_event
                if isinstance(chunk_event, TextDeltaEvent):
                    final_text += chunk_event.delta

            # _call_provider 把响应存在 turn 上，这里取回 / _call_provider stores the response on turn; retrieve it
            response = turn._last_response  # type: ignore[attr-defined]
            if response is None:
                # provider 没产出最终响应 → 按错误终止 / provider produced no final response → terminate as error
                yield self._force_done(turn, StopReason.ERROR)
                return

            # ── 工具调用 / TOOL CALLING ───────────────────────────────────────
            if response.wants_tool_call:
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
                stop_reason=StopReason.COMPLETED.value,
                outcome=TurnOutcome.COMPLETED.value,
            )
            self._emit(done)
            yield self._transition(turn, AgentState.DONE)
            yield done
            return

    # ── Provider 调用（流式） / Provider call (streaming) ─────────────────────

    async def _call_provider(self, turn: TurnContext, tools: list[dict[str, Any]] | None) -> AsyncIterator[AgentEvent]:
        turn._last_response = None  # type: ignore[attr-defined]
        try:
            async for chunk in self._provider.stream(
                system=self._ctx.system_prompt,
                messages=self._ctx.history,
                tools=tools or None,
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
                async for ev in self._call_provider(turn, tools):
                    yield ev
            else:
                raise

    # ── 工具执行 / Tool execution ────────────────────────────────────────────

    async def _execute_tool(self, turn: TurnContext, tc: Any) -> AsyncIterator[AgentEvent]:
        """
        执行单次工具调用，整合卡死检测 / 工具禁用 / per-tool 预算 / 返回契约 / Execute a single tool call, integrating stuck-detection / tool-disabling / per-tool budget / return contract.

        LLM 请求过的工具一律发 ToolCallEvent（计入 tools_called）；但卡死 / 禁用 / 预算路径跳过执行，发 success=False 的 ToolResultEvent（不计入 tools_executed）/ Tools the LLM requested always emit a ToolCallEvent (counted in tools_called); but the stuck/disabled/budget paths skip execution and emit a success=False ToolResultEvent (NOT counted in tools_executed).
        """
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

        # ── 动态禁用兜底 / disabled-tool guard ──
        if tc.tool_name in turn.denied_tools:
            tr = ToolResult(
                status=ToolResultStatus.FAILURE,
                content=f"Tool '{tc.tool_name}' has been disabled for the rest of this turn. Change approach or give a final answer.",
                error_code="tool_disabled",
            )
            async for ev in self._emit_tool_result(turn, tc, tr, latency_ms=0.0, intervention_reason="tool_disabled"):
                yield ev
            return

        # ── 卡死检测（per-签名累计）/ stuck detection (per-signature count) ──
        decision = self._stuck.observe(tc.tool_name, tc.tool_input)
        if decision.stuck:
            turn.denied_tools.add(tc.tool_name)
            self._ctx.append_message(Message(role="user", content=_STUCK_RECOVERY_DIRECTIVE))
            tr = ToolResult(
                status=ToolResultStatus.FAILURE,
                content=f"Tool '{tc.tool_name}' skipped — identical signature repeated {decision.count} times. Change approach or give a final answer.",
                error_code="stuck_loop",
            )
            async for ev in self._emit_tool_result(turn, tc, tr, latency_ms=0.0, intervention_reason="stuck_loop"):
                yield ev
            return

        # ── per-tool 调用预算（catch 同工具不同参数的钻空子）/ per-tool call budget (catches same-tool varying-args evasion) ──
        turn.tool_call_counts[tc.tool_name] = turn.tool_call_counts.get(tc.tool_name, 0) + 1
        if turn.tool_call_counts[tc.tool_name] > self._max_calls_per_tool:
            turn.denied_tools.add(tc.tool_name)
            self._ctx.append_message(Message(role="user", content=_STUCK_RECOVERY_DIRECTIVE))
            tr = ToolResult(
                status=ToolResultStatus.FAILURE,
                content=f"Tool '{tc.tool_name}' call budget ({self._max_calls_per_tool}) exhausted. Change approach or give a final answer.",
                error_code="tool_call_budget",
            )
            async for ev in self._emit_tool_result(turn, tc, tr, latency_ms=0.0, intervention_reason="tool_call_budget"):
                yield ev
            return

        # ── 正常执行 / normal execution ──
        fn = self._tools.get(tc.tool_name)
        t0 = time.monotonic()
        if fn is None:
            tr = ToolResult(
                status=ToolResultStatus.FAILURE,
                content=f"Tool '{tc.tool_name}' not found.",
                error_code="unknown_tool",
            )
        else:
            try:
                raw = await asyncio.wait_for(
                    fn(tc.tool_input, turn.to_tool_context()),
                    timeout=self.TOOL_TIMEOUT,
                )
                tr = self._coerce_tool_result(raw)
            except asyncio.TimeoutError:
                tr = ToolResult(
                    status=ToolResultStatus.FAILURE,
                    content=f"Tool '{tc.tool_name}' timed out after {self.TOOL_TIMEOUT}s.",
                    error_code="timeout",
                )
            except Exception as exc:
                tr = ToolResult(
                    status=ToolResultStatus.FAILURE,
                    content=f"Tool error: {exc}",
                    error_code="tool_error",
                )
                log.warning("tool %s raised: %s", tc.tool_name, exc)

        latency = (time.monotonic() - t0) * 1000
        async for ev in self._emit_tool_result(turn, tc, tr, latency_ms=latency):
            yield ev

    async def _emit_tool_result(
        self, turn: TurnContext, tc: Any, tr: ToolResult, latency_ms: float, intervention_reason: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """把 ToolResult 序列化、发 InterventionEvent（可选）+ ToolResultEvent / Serialize a ToolResult, emit InterventionEvent (optional) + ToolResultEvent."""
        if intervention_reason is not None:
            iv = InterventionEvent(
                trace_id=turn.trace_id,
                session_key=turn.session_key,
                agent_id=turn.agent_id,
                reason=intervention_reason,
                tool_name=tc.tool_name,
            )
            self._emit(iv)
            yield iv

        content = self._serialize_tool_result(tr)
        result_ev = ToolResultEvent(
            trace_id=turn.trace_id,
            session_key=turn.session_key,
            agent_id=turn.agent_id,
            tool_name=tc.tool_name,
            tool_use_id=tc.tool_use_id,
            success=(tr.status == ToolResultStatus.SUCCESS),
            latency_ms=latency_ms,
            output_preview=content[:200],
        )
        # 把完整输出暂存在事件上，供 _make_tool_result_block 读取 / stash full output on event so _make_tool_result_block can read it
        result_ev._full_output = content  # type: ignore[attr-defined]
        self._emit(result_ev)
        yield result_ev

    @staticmethod
    def _coerce_tool_result(raw: Any) -> ToolResult:
        """把工具任意返回值归一化为 ToolResult（裸 str 自动包装，向后兼容）/ Normalize any tool return into a ToolResult (bare str auto-wrapped, backward compatible)."""
        if isinstance(raw, ToolResult):
            return raw
        return ToolResult(status=ToolResultStatus.SUCCESS, content=str(raw))

    @staticmethod
    def _serialize_tool_result(tr: ToolResult) -> str:
        """把 ToolResult 序列化为带 status / next_action 的强指引字符串 / Serialize a ToolResult into a strong-guidance string with status / next_action."""
        s = f"[status: {tr.status}] {tr.content}"
        if tr.next_action_hint:
            s += f"\n[next_action: {tr.next_action_hint}]"
        if tr.error_code and tr.status != ToolResultStatus.SUCCESS:
            s += f"\n[error_code: {tr.error_code}]"
        return s

    # ── 预算与两阶段收尾 / Budget & two-phase finalization ────────────────────

    def _budget_exhausted(self, turn: TurnContext) -> StopReason | None:
        """返回首个已耗尽的预算对应的 StopReason，否则 None / Return the StopReason for the first exhausted budget, else None."""
        if turn.iterations >= self._max_iter:
            return StopReason.MAX_ITERATIONS
        if turn.tool_call_count >= self._max_tool_calls:
            return StopReason.MAX_TOOL_CALLS
        return None

    async def _handle_budget_exhausted(self, turn: TurnContext, reason: StopReason) -> AsyncIterator[AgentEvent]:
        """
        两阶段优雅收尾：首次撞预算时注入"别调工具直接答"指令并剥离工具再做一次；二次或仍调工具则硬停 / Two-phase graceful finalization: on first budget hit inject an "answer without tools" directive and retry tool-stripped; on second hit or further tool requests, hard-stop.
        """
        if not turn.finalization_attempted:
            turn.finalization_attempted = True
            self._ctx.append_message(Message(role="user", content=_FINALIZATION_DIRECTIVE))
            iv = InterventionEvent(
                trace_id=turn.trace_id,
                session_key=turn.session_key,
                agent_id=turn.agent_id,
                reason="finalization",
            )
            self._emit(iv)
            yield iv

            # 剥离工具做一次最终回答尝试 / tool-stripped final-answer attempt
            yield self._transition(turn, AgentState.THINKING)
            async for chunk_event in self._call_provider(turn, tools=None):
                yield chunk_event
            response = turn._last_response  # type: ignore[attr-defined]
            if response is not None and not response.wants_tool_call:
                # 模型给了最终答案 → 以撞预算原因收尾（outcome=partial）/ model gave a final answer → end with the budget reason (outcome=partial)
                self._ctx.append_message(response.to_assistant_message())
                self._ctx.total_input_tokens += response.input_tokens
                self._ctx.total_output_tokens += response.output_tokens
                yield self._force_done(turn, reason, final_text=response.final_text)
                return
            # 模型仍要调工具 → 硬停 / model still wants tools → hard-stop
        yield self._force_done(turn, reason)

    def _visible_tool_defs(self, turn: TurnContext) -> list[dict[str, Any]]:
        """过滤掉已动态禁用的工具定义，模型看不到也就不会再调 / Filter out dynamically-disabled tool definitions so the model can't re-request them."""
        if not turn.denied_tools:
            return self._tool_defs
        return [d for d in self._tool_defs if d.get("name") not in turn.denied_tools]

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

    def _force_done(self, turn: TurnContext, stop_reason: StopReason, final_text: str = "") -> DoneEvent:
        """
        预算/卡死/错误等非正常终止时构造 DoneEvent（带 stop_reason + outcome）/ Build a DoneEvent for non-normal termination (budget/stuck/error) carrying stop_reason + outcome.
        """
        ev = DoneEvent(
            trace_id=turn.trace_id,
            session_key=turn.session_key,
            agent_id=turn.agent_id,
            final_text=final_text or f"[Stopped: {stop_reason.value}]",
            elapsed_ms=turn.elapsed_ms(),
            stop_reason=stop_reason.value,
            outcome=classify_outcome(stop_reason).value,
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

"""
行为指纹测试框架 / Behavioral fingerprint test framework.

核心理念：不断言 LLM 输出的文字内容（不确定），断言行为约束 / Core idea: don't assert LLM output text (non-deterministic), assert behavioral constraints:
  - 调用了哪些工具（超集关系：实际调用集合 ⊇ must_call_tools）
  - 禁止某些工具被成功执行（安全边界：must_not_execute_tools ∩ tools_executed = ∅）
  - 工具调用次数范围（容忍 ±1 浮动：call_count_min ≤ count ≤ call_count_max）
  - 最终状态是否符合预期（must_complete → final_state == "DONE"）

区分"请求调用"与"成功执行"：
  - tools_called: LLM 请求调用的工具集合（无论工具是否存在）
  - tools_executed: 实际执行成功的工具集合（ToolResultEvent.success=True）
  安全测试的核心断言：forbidden_tool ∉ tools_executed
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nanoharness.core.context import AgentContext
from nanoharness.core.event_store import (
    DoneEvent, ErrorEvent, EventStore,
    StateChangeEvent, ToolCallEvent, ToolResultEvent,
)
from nanoharness.core.nano_core import NanoCore
from nanoharness.provider.base import LLMProvider


# ─── 行为指纹 / Behavior Fingerprint ────────────────────────────────────────

@dataclass
class BehaviorFingerprint:
    """一次 Agent 运行的行为快照，记录可观测的行为事实 / A behavioral snapshot of one Agent run, recording observable behavior facts."""
    tools_called: set[str] = field(default_factory=set)      # LLM 请求调用的工具（含失败） / tools requested by LLM (incl. failed)
    tools_executed: set[str] = field(default_factory=set)    # 成功执行的工具 / successfully executed tools
    tool_call_count: int = 0                                  # 总调用请求次数 / total call request count
    states_visited: list[str] = field(default_factory=list)  # 状态转换序列 / state transition sequence
    final_state: str = ""                                     # 最终状态（"DONE" / 状态名） / final state ("DONE" / state name)
    error_occurred: bool = False                              # 是否出现 ErrorEvent / whether ErrorEvent occurred
    event_kinds: list[str] = field(default_factory=list)     # 事件类型序列（调试用） / event kind sequence (for debugging)

    @property
    def reached_done(self) -> bool:
        return self.final_state == "DONE"


# ─── 行为约束 / Behavior Constraint ─────────────────────────────────────────

@dataclass
class BehaviorConstraint:
    """
    声明式行为约束规格 / Declarative behavior constraint spec.

    超集关系（must_call/execute_tools）：允许 Agent 调用更多工具，
    只要约束指定的工具都被调用/执行了就算满足 / Superset relation: the Agent may call more tools,
    as long as all tools specified by the constraint are called/executed.
    禁止执行（must_not_execute_tools）：安全边界，指定工具不得成功执行 / Forbidden execution: safety boundary, specified tools must not execute successfully.
    次数范围（call_count_min/max）：对 ±1 浮动的工具调用次数进行宽松约束 / Count range: loose constraint on tool call count tolerating ±1 fluctuation.
    """
    must_call_tools: set[str] = field(default_factory=set)         # 必须请求调用（超集） / must be requested (superset)
    must_execute_tools: set[str] = field(default_factory=set)      # 必须成功执行（超集） / must be executed successfully (superset)
    must_not_execute_tools: set[str] = field(default_factory=set)  # 禁止成功执行（安全边界） / must not execute successfully (safety boundary)
    call_count_min: int = 0                                         # 最少请求次数（含） / minimum request count (inclusive)
    call_count_max: int = 100                                       # 最多请求次数（含） / maximum request count (inclusive)
    must_complete: bool = True                                      # 必须到达 DONE 状态 / must reach DONE state
    error_allowed: bool = True                                      # 是否允许 ErrorEvent / whether ErrorEvent is allowed

    def assert_satisfied(self, fp: BehaviorFingerprint) -> None:
        """断言指纹满足所有约束，违反时抛出带可读说明的 AssertionError / Assert the fingerprint satisfies all constraints, raising a readable AssertionError on violation."""
        missing_calls = self.must_call_tools - fp.tools_called
        assert not missing_calls, (
            f"必须调用的工具未被请求: {missing_calls}（实际调用: {fp.tools_called}）"
        )

        missing_exec = self.must_execute_tools - fp.tools_executed
        assert not missing_exec, (
            f"必须成功执行的工具未执行: {missing_exec}（实际执行: {fp.tools_executed}）"
        )

        # 安全边界：禁止工具与已执行工具的交集必须为空 / Safety boundary: intersection of forbidden and executed tools must be empty
        forbidden_exec = self.must_not_execute_tools & fp.tools_executed
        assert not forbidden_exec, (
            f"禁止执行的工具被成功调用: {forbidden_exec}（安全边界被突破）"
        )

        assert self.call_count_min <= fp.tool_call_count <= self.call_count_max, (
            f"工具调用次数 {fp.tool_call_count} 不在范围 "
            f"[{self.call_count_min}, {self.call_count_max}] 内"
        )

        if self.must_complete:
            assert fp.reached_done, (
                f"Agent 未到达 DONE 状态（实际最终状态: {fp.final_state!r}）"
            )

        if not self.error_allowed:
            assert not fp.error_occurred, "约束不允许 ErrorEvent，但检测到了错误"


# ─── 辅助运行函数 / Helper Run Function ─────────────────────────────────────

async def run_and_fingerprint(
    prompt: str,
    provider: LLMProvider,
    tools: dict | None = None,
    tool_definitions: list | None = None,
    ctx: AgentContext | None = None,
) -> BehaviorFingerprint:
    """
    运行一次 NanoCore turn，采集并返回行为指纹 / Run one NanoCore turn, collect and return the behavior fingerprint.

    不捕获 run_turn 抛出的异常——框架级异常意味着测试环境有问题，
    应该让它向上传播而不是被吞掉变成空指纹 / Do not catch exceptions from run_turn — a framework-level exception means the test environment is broken;
    let it propagate instead of being swallowed into an empty fingerprint.
    """
    if ctx is None:
        ctx = AgentContext(
            agent_id="fp-agent",
            session_key="fp-session",
            system_prompt="你是测试助手。",
            model_id=getattr(provider, "model_id", "test-model"),
        )

    store = EventStore()
    core = NanoCore(
        ctx=ctx,
        provider=provider,
        tools=tools or {},
        tool_definitions=tool_definitions or [],
        event_store=store,
    )

    fp = BehaviorFingerprint()

    async for event in core.run_turn(prompt):
        fp.event_kinds.append(event.kind)

        if isinstance(event, ToolCallEvent):
            fp.tools_called.add(event.tool_name)
            fp.tool_call_count += 1

        elif isinstance(event, ToolResultEvent):
            if event.success:
                fp.tools_executed.add(event.tool_name)

        elif isinstance(event, StateChangeEvent):
            fp.states_visited.append(event.to_state)
            fp.final_state = event.to_state

        elif isinstance(event, ErrorEvent):
            fp.error_occurred = True

        elif isinstance(event, DoneEvent):
            fp.final_state = "DONE"

    return fp

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

# ─── 状态机 / State Machine ────────────────────────────────────────────────────

class AgentState(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    DONE = "done"
    ERROR = "error"


# ─── 消息类型 / Message Types ────────────────────────────────────────────────────

# Anthropic 多块内容，用于工具调用 / 结果 / Anthropic multi-block content for tool use / result
ContentBlock = dict[str, Any]

# 对话历史中的一条消息（Anthropic 格式） / A message in the conversation history (Anthropic format)
@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]
    token_count: int = 0  # 已知时由 provider.count_tokens() 填充 / filled by provider.count_tokens() when known

    def is_tool_use(self) -> bool:
        if isinstance(self.content, list):
            return any(b.get("type") == "tool_use" for b in self.content)
        return False

    def is_tool_result(self) -> bool:
        if isinstance(self.content, list):
            return any(b.get("type") == "tool_result" for b in self.content)
        return False

    def to_api_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


# ─── 工具上下文（frozen = 并发安全） / Tool Context (frozen = concurrent-safe) ──

@dataclass(frozen=True)
class ToolContext:
    """单次工具调用的上下文。不可变，避免并发子 Agent 相互覆盖 / Per-tool-call context. Immutable so concurrent subagents can't clobber each other."""
    session_key: str
    agent_id: str
    trace_id: str
    workspace_dir: str = "/tmp/nanoharness"
    budget_limit: int = 10       # 每轮最大工具调用数 / max tool calls per turn
    sandbox_enabled: bool = False

    def with_budget(self, new_limit: int) -> "ToolContext":
        return replace(self, budget_limit=new_limit)


# ─── Turn 级上下文 / Turn-level Context ────────────────────────────────────────

@dataclass
class TurnContext:
    """单轮 ReAct 循环的可变状态 / Mutable state for a single turn through the ReAct loop."""
    session_key: str
    agent_id: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: AgentState = AgentState.IDLE
    iterations: int = 0
    tool_call_count: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # 本轮已执行过压缩则置 True（防止重复压缩） / set to True once compaction has run this turn (prevent double-compact)
    has_compacted: bool = False

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def to_tool_context(self, workspace_dir: str = "/tmp/nanoharness") -> ToolContext:
        return ToolContext(
            session_key=self.session_key,
            agent_id=self.agent_id,
            trace_id=self.trace_id,
            workspace_dir=workspace_dir,
        )


# ─── Agent 级上下文 / Agent-level Context ───────────────────────────────────────

@dataclass
class AgentContext:
    """
    贯穿整个 Agent session 生命周期的持久化上下文 / Persistent context that lives for the lifetime of one agent session.
    构造时传入 NanoCore；NanoCore 不持有它的所有权 / Passed to NanoCore on construction; NanoCore does NOT own it.
    """
    agent_id: str
    session_key: str
    system_prompt: str
    model_id: str

    # 可变的对话历史（Anthropic 格式的 Message 列表） / mutable conversation history (Anthropic-format Message list)
    history: list[Message] = field(default_factory=list)

    # 由 Harness 各层注入的元数据（记忆召回、技能等） / metadata injected by Harness layers (memory recall, skills, etc.)
    extra_context: dict[str, Any] = field(default_factory=dict)

    # 累计统计 / accumulated stats
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0

    def history_as_api_list(self) -> list[dict[str, Any]]:
        return [m.to_api_dict() for m in self.history]

    def append_message(self, msg: Message) -> None:
        self.history.append(msg)

    def approximate_token_count(self) -> int:
        """粗略启发式：累加已存的 token_count，否则按字符数 /4 估算 / Rough heuristic: sum stored token_count or fall back to char/4."""
        total = len(self.system_prompt) // 4
        for m in self.history:
            if m.token_count > 0:
                total += m.token_count
            elif isinstance(m.content, str):
                total += len(m.content) // 4
            else:
                import json
                total += len(json.dumps(m.content)) // 4
        return total

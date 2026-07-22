from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

# ─── 基础事件 / Base Event ───────────────────────────────────────────────────────

@dataclass
class AgentEvent:
    # 非默认字段必须在前；子类用判别字符串覆盖 `kind` / Non-default fields must come first; subclasses override `kind` with a discriminator string
    trace_id: str
    session_key: str
    agent_id: str
    kind: str = ""
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)


# ─── 具体事件类型 / Concrete Event Types ─────────────────────────────────────────────

@dataclass
class StateChangeEvent(AgentEvent):
    kind: str = "state_change"
    from_state: str = ""
    to_state: str = ""


@dataclass
class ToolCallEvent(AgentEvent):
    kind: str = "tool_call"
    tool_name: str = ""
    tool_use_id: str = ""
    input_summary: str = ""    # 输入的截断表示，非完整 payload / truncated repr of input, not full payload


@dataclass
class ToolResultEvent(AgentEvent):
    kind: str = "tool_result"
    tool_name: str = ""
    tool_use_id: str = ""
    success: bool = True
    latency_ms: float = 0.0
    output_preview: str = ""   # 结果的前 200 个字符 / first 200 chars of result


@dataclass
class CompactionEvent(AgentEvent):
    kind: str = "compaction"
    tokens_before: int = 0
    tokens_after: int = 0
    messages_removed: int = 0


@dataclass
class DoneEvent(AgentEvent):
    kind: str = "done"
    final_text: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    elapsed_ms: float = 0.0


@dataclass
class ErrorEvent(AgentEvent):
    kind: str = "error"
    error_type: str = ""
    error_message: str = ""
    recoverable: bool = False


@dataclass
class TextDeltaEvent(AgentEvent):
    """流式 token 增量 — 消费者可重组或增量展示 / Streaming token delta — consumers can reassemble or display incrementally."""
    kind: str = "text_delta"
    delta: str = ""


# ─── 事件存储 / Event Store ──────────────────────────────────────────────────────

class EventStore:
    """
    内存中只追加的事件日志，带 trace_id 索引 / In-memory append-only event log with trace_id indexing.

    生产扩展点：在 flush() 中把 _log 换成 SQLite 写入 / Production extension point: swap _log for SQLite writes in flush().
    当前范围：单进程，支持 Gradio 回放仪表板 / Current scope: single-process, supports Gradio replay dashboard.
    """

    def __init__(self) -> None:
        self._log: list[AgentEvent] = []
        self._by_trace: dict[str, list[AgentEvent]] = {}

    def append(self, event: AgentEvent) -> None:
        self._log.append(event)
        self._by_trace.setdefault(event.trace_id, []).append(event)

    def get_trace(self, trace_id: str) -> list[AgentEvent]:
        return list(self._by_trace.get(trace_id, []))

    def get_session(self, session_key: str) -> list[AgentEvent]:
        return [e for e in self._log if e.session_key == session_key]

    def last_n(self, n: int) -> list[AgentEvent]:
        return list(self._log[-n:])

    def clear(self) -> None:
        self._log.clear()
        self._by_trace.clear()

    def __len__(self) -> int:
        return len(self._log)

from __future__ import annotations

import json
import logging
from typing import Any

from nanoharness.core.context import Message
from nanoharness.engine.hooks.types import CompactionHookContext, CompactionHookResult
from nanoharness.memory.store import MemoryEntry, MemoryStore, MemoryType

log = logging.getLogger(__name__)

# 注入 context 时的 importance（工具结果高于普通对话） / importance when injecting into context (tool results rank higher than ordinary chat)
_IMPORTANCE_BY_TYPE: dict[MemoryType, float] = {
    MemoryType.TOOL_RESULT: 0.85,
    MemoryType.FACT:        0.70,
    MemoryType.PREFERENCE:  0.75,
    MemoryType.SUMMARY:     0.60,
}


class MemoryCompactionHook:
    """
    在上下文压缩触发前，把即将被丢弃的消息里的重要信息写入 L3 长期记忆。 / Before compaction triggers, persist important info from soon-to-be-dropped messages into L3 long-term memory.

    这解决了"Compaction Failure Mode B"： / Solves "Compaction Failure Mode B":
    压缩只看 token 数，不看信息价值，可能把关键工具结果当普通对话丢掉。 / Compaction looks only at token count, not info value, and may drop key tool results as ordinary chat.
    Hook 提前拦截，把高价值内容持久化，压缩后信息不丢。 / The hook intercepts early, persists high-value content, so info survives compaction.

    面试话术 / Interview talking point:
    "压缩的最大风险不是 API 格式被破坏（已用 turn-boundary 保护解决），
    而是关键工具结果被当作普通文字压缩掉了——比如搜索发现的重要事实。
    CompactionHook 在压缩前扫描待删消息，把 tool_result 和高分消息写入
    长期记忆，这样下次会话还能通过 FTS5 检索到。"
    """

    def __init__(
        self,
        store: MemoryStore,
        agent_id: str,
        session_key: str,
        provider: Any | None = None,   # 可选：用 LLM 提炼，无则直接存原文 / optional: use LLM to distill; if absent, store raw text
    ) -> None:
        self._store = store
        self._agent_id = agent_id
        self._session_key = session_key
        self._provider = provider

    async def before_compact(self, ctx: CompactionHookContext) -> None:
        """压缩触发前：从 AgentContext.history 中抽取重要内容写入 L3。 / Before compaction: extract important content from AgentContext.history into L3."""
        # 注意：此时 AgentContext 已在 NanoCore 里，通过 extra 传进来 / Note: AgentContext lives in NanoCore here, passed via extra
        # 实际使用时由 MemoryManager 组装并调用 / In practice assembled and invoked by MemoryManager
        pass  # 真正的逻辑在 extract_from_messages() / real logic lives in extract_from_messages()

    async def after_compact(
        self, ctx: CompactionHookContext, result: CompactionHookResult
    ) -> None:
        pass

    def extract_from_messages(self, messages: list[Message]) -> int:
        """
        扫描消息列表，把值得保留的内容写入 L3。 / Scan the message list and persist worth-keeping content into L3.
        返回写入的条数。 / Returns the number of entries written.

        保留规则： / Retention rules:
        - tool_result 消息 → MemoryType.TOOL_RESULT / - tool_result messages → MemoryType.TOOL_RESULT
        - 超过 200 字符的 user 消息（可能含重要指令）→ MemoryType.FACT / - user messages over 200 chars (may contain key instructions) → MemoryType.FACT
        """
        saved = 0
        for msg in messages:
            entry = self._classify_message(msg)
            if entry:
                self._store.upsert(entry)
                saved += 1
        if saved:
            log.info("压缩前保护：写入 %d 条长期记忆 (session=%s)", saved, self._session_key)
        return saved

    def _classify_message(self, msg: Message) -> MemoryEntry | None:
        if msg.is_tool_result():
            content = self._extract_tool_result_text(msg)
            if not content or len(content) < 20:
                return None
            return MemoryEntry(
                content=content[:800],   # 防止超长工具结果撑爆存储 / guard against oversized tool results blowing up storage
                agent_id=self._agent_id,
                session_key=self._session_key,
                memory_type=MemoryType.TOOL_RESULT,
                importance=_IMPORTANCE_BY_TYPE[MemoryType.TOOL_RESULT],
            )

        if msg.role == "user" and isinstance(msg.content, str) and len(msg.content) > 200:
            return MemoryEntry(
                content=msg.content[:800],
                agent_id=self._agent_id,
                session_key=self._session_key,
                memory_type=MemoryType.FACT,
                importance=_IMPORTANCE_BY_TYPE[MemoryType.FACT],
            )

        return None

    @staticmethod
    def _extract_tool_result_text(msg: Message) -> str:
        """从 tool_result 消息块里提取文字内容。 / Extract text content from a tool_result message block."""
        if isinstance(msg.content, str):
            return msg.content
        parts: list[str] = []
        for block in msg.content:
            if isinstance(block, dict):
                c = block.get("content", "")
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
        return "\n".join(parts)

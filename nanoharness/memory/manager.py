from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoharness.core.context import AgentContext
from nanoharness.memory.compaction_hooks import MemoryCompactionHook
from nanoharness.memory.consolidation import SessionConsolidator
from nanoharness.memory.retrieval import MemoryRetriever, RetrievalConfig
from nanoharness.memory.store import MemoryEntry, MemoryStore, MemoryType

log = logging.getLogger(__name__)

# key 名称：召回结果写入 ctx.extra_context 的字段 / key name: field where recall results are written into ctx.extra_context
MEMORY_CONTEXT_KEY = "memory_context"


@dataclass
class MemoryConfig:
    db_path: str | Path = ":memory:"          # 生产建议: ~/.nanoharness/memory.db / production suggestion
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    # prefetch 触发阈值：history 超过多少条才查记忆（避免每轮都查） / prefetch trigger threshold: min history length to query memories (avoids per-turn queries)
    prefetch_min_turns: int = 2
    # 是否在压缩前自动保护重要消息 / whether to auto-protect important messages before compaction
    enable_compaction_protection: bool = True


class MemoryManager:
    """
    记忆系统对外的统一入口。调用方只需要知道这一个类。 / Unified external entry of the memory system; callers only need this one class.

    三个核心方法： / Three core methods:
    - prefetch(ctx, query) : turn 开始前召回相关记忆，注入 ctx.extra_context / - prefetch(ctx, query): recall related memories before turn, inject into ctx.extra_context
    - on_compact(ctx)      : 压缩前保护重要消息写入 L3 / - on_compact(ctx): protect important messages into L3 before compaction
    - flush(ctx)           : session 结束后触发巩固（异步后台） / - flush(ctx): trigger consolidation after session ends (async background)

    存储分层： / Storage tiers:
    - L1 短期记忆 = ctx.history（已有，不在这里管理） / - L1 short-term = ctx.history (already exists, not managed here)
    - L2 会话摘要 = SQLite sessions 表 / - L2 session summaries = SQLite sessions table
    - L3 长期事实 = SQLite memories 表 + FTS5 索引 / - L3 long-term facts = SQLite memories table + FTS5 index

    面试话术 / Interview talking point:
    "MemoryManager 是记忆系统的门面（Facade 模式）。
    TurnRunner 只调这一个类，不直接操作 Store / Retriever / Hook，
    这样底层换成向量库只需改 MemoryManager 内部，上层零修改。"
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        provider: Any | None = None,   # LLMProvider，用于巩固和压缩保护 / LLMProvider, for consolidation and compaction protection
    ) -> None:
        self._cfg = config or MemoryConfig()
        self._provider = provider
        self._store = MemoryStore(self._cfg.db_path)
        self._retriever = MemoryRetriever(self._store, self._cfg.retrieval)
        self._consolidator = (
            SessionConsolidator(self._store, provider) if provider else None
        )
        self._session_start_times: dict[str, float] = {}

    # ── 主流程接口 / Main flow interface ─────────────────────────────────────────────────────────────

    async def prefetch(self, ctx: AgentContext, query: str) -> str:
        """
        turn 开始前调用，查询并返回记忆 context 字符串。 / Called before turn starts; queries and returns the memory context string.
        同时写入 ctx.extra_context[MEMORY_CONTEXT_KEY]。 / Also writes to ctx.extra_context[MEMORY_CONTEXT_KEY].

        调用时机：TurnRunner._execute() 路由决策之后、NanoCore 启动之前。 / Timing: after routing decision in TurnRunner._execute(), before NanoCore starts.
        """
        # session 首次 turn，记录开始时间 / first turn of session, record start time
        if ctx.session_key not in self._session_start_times:
            self._session_start_times[ctx.session_key] = time.time()

        # history 太短时不召回（减少无意义 DB 查询） / skip recall when history is too short (reduce pointless DB queries)
        if len(ctx.history) < self._cfg.prefetch_min_turns:
            return ""

        memories, summaries = self._retriever.retrieve_with_sessions(query, ctx.agent_id)
        context_text = self._retriever.format_for_context(memories, summaries)

        if context_text:
            ctx.extra_context[MEMORY_CONTEXT_KEY] = context_text
            log.debug("记忆召回: session=%s 记忆=%d条 摘要=%d条",
                      ctx.session_key, len(memories), len(summaries))

        return context_text

    async def on_compact(self, ctx: AgentContext) -> int:
        """
        上下文压缩前调用，把 history 中高价值内容写入 L3。 / Called before context compaction; writes high-value content from history into L3.
        返回写入条数。 / Returns the number of entries written.
        """
        if not self._cfg.enable_compaction_protection:
            return 0
        hook = MemoryCompactionHook(
            store=self._store,
            agent_id=ctx.agent_id,
            session_key=ctx.session_key,
            provider=self._provider,
        )
        return hook.extract_from_messages(ctx.history)

    async def capture_turn(
        self,
        ctx: AgentContext,
        user_message: str,
        assistant_text: str,
    ) -> None:
        """
        每轮 ReAct 结束后调用，把 (用户消息, 助手回复) 原文写入 L3 作为情节记忆。
        无 LLM 提炼，只是原始捕获。flush() 的 LLM 巩固从 EPISODE 条目中提炼摘要和事实。

        设计对照 / Design reference:
        - opensquilla TurnFinalizerStage.run() → TurnCaptureService（per-turn raw capture）
        - DeerFlow MemoryMiddleware.after_agent()（per-turn capture + debounce LLM）
        NanoHarness 选择 opensquilla 路线：raw capture 不用 LLM，LLM 提炼留给 flush()。

        / Called after each ReAct turn; writes (user_message, assistant_text) as episodic memory into L3.
        No LLM extraction — raw capture only. flush()'s LLM consolidation distills from EPISODE entries.
        """
        if not assistant_text:
            return
        # 截断防止单条记忆过长 / Truncate to prevent oversized entries
        content = f"[用户] {user_message[:500]}\n[助手] {assistant_text[:1000]}"
        self._store.upsert(MemoryEntry(
            content=content,
            agent_id=ctx.agent_id,
            session_key=ctx.session_key,
            memory_type=MemoryType.EPISODE,
            importance=0.5,
        ))
        log.debug("情节记忆写入: session=%s / episode written: session=%s", ctx.session_key, ctx.session_key)

    async def flush(self, ctx: AgentContext) -> None:
        """
        session 结束时调用。触发异步巩固（后台跑，不阻塞响应）。 / Called when session ends; triggers async consolidation (background, non-blocking).
        如果没有 provider 则跳过 LLM 提炼，只记录时间戳。 / Without a provider, skips LLM distillation and only records a timestamp.
        """
        started_at = self._session_start_times.pop(ctx.session_key, time.time())
        if self._consolidator:
            await self._consolidator.consolidate(ctx, started_at)
        else:
            log.debug("无 provider，跳过 session 巩固: session=%s", ctx.session_key)

    # ── 直接写入接口（供工具/外部调用） / Direct write interface (for tools/external callers) ───────────────────────────────────────

    def remember(
        self,
        agent_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.7,
        session_key: str = "",
    ) -> MemoryEntry:
        """
        显式写入一条长期记忆。 / Explicitly write one long-term memory entry.
        可由工具函数调用（如 remember_user_preference 工具）。 / Can be called by tool functions (e.g. remember_user_preference tool).
        """
        entry = MemoryEntry(
            content=content,
            agent_id=agent_id,
            session_key=session_key,
            memory_type=memory_type,
            importance=importance,
        )
        self._store.upsert(entry)
        return entry

    def search(self, query: str, agent_id: str) -> list[str]:
        """简单检索接口，返回内容字符串列表，供工具函数使用。 / Simple retrieval interface, returns content strings, for tool functions."""
        results = self._retriever.retrieve(query, agent_id)
        return [r.entry.content for r in results]

    # ── 统计 / Stats ──────────────────────────────────────────────────────────────────

    def stats(self, agent_id: str) -> dict:
        return {
            "memory_count": self._store.count(agent_id),
            "recent_sessions": len(self._store.get_recent_sessions(agent_id)),
            "db_path": str(self._cfg.db_path),
        }

    def close(self) -> None:
        self._store.close()

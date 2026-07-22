from __future__ import annotations

import math
import time
from dataclasses import dataclass

from nanoharness.memory.store import MemoryEntry, MemoryStore, MemoryType


# ─── 检索配置 / Retrieval config ──────────────────────────────────────────────────────────────────

@dataclass
class RetrievalConfig:
    top_k: int = 5                  # 最终注入 context 的条数 / final count injected into context
    fts_candidate_limit: int = 20   # FTS5 宽召回数量（top_k 的 4 倍） / FTS5 wide-recall count (4× top_k)
    time_decay_days: float = 30.0   # 时间衰减半衰期（天） / time-decay half-life (days)
    importance_weight: float = 0.4  # importance 分在最终得分中的权重 / weight of importance in final score
    recency_weight: float = 0.3     # 时间新鲜度权重 / time-recency weight
    bm25_weight: float = 0.3        # BM25 相关性权重 / BM25 relevance weight
    # 低于此分数的记忆不注入（防止噪声） / memories below this score are not injected (noise guard)
    min_score_threshold: float = 0.1
    # 最近 N 个 session 摘要总是注入（不参与排序） / always inject the latest N session summaries (not ranked)
    always_inject_recent_sessions: int = 2


@dataclass
class RetrievedMemory:
    entry: MemoryEntry
    score: float       # 综合得分 [0, 1] / composite score [0, 1]
    reason: str        # 为什么被召回（debug 用） / why it was recalled (for debug)


# ─── 检索引擎 / Retrieval engine ──────────────────────────────────────────────────────────────────

class MemoryRetriever:
    """
    基于 FTS5 BM25 + 时间衰减 + 重要度的混合检索。 / Hybrid retrieval based on FTS5 BM25 + time decay + importance.

    检索流程： / Retrieval flow:
    1. FTS5 宽召回（取 top_k×4 候选） / 1. FTS5 wide recall (top_k×4 candidates)
    2. 对每个候选计算综合得分： / 2. Compute composite score for each candidate:
       score = bm25_weight × bm25归一化 + recency_weight × 时间衰减 + importance_weight × importance
    3. 过滤低分，取 Top-K / 3. Filter low scores, take Top-K
    4. 最近 N 个 session 摘要额外注入（不竞争名额） / 4. Inject the latest N session summaries additionally (no quota competition)

    面试话术 / Interview talking point:
    "没有向量检索，但三路加权排序已经能解决主要问题：
    BM25 保证关键词相关性，时间衰减保证近期信息优先，
    importance 权重保证高价值信息（工具结果/用户偏好）不被普通对话淹没。"
    """

    def __init__(self, store: MemoryStore, config: RetrievalConfig | None = None) -> None:
        self._store = store
        self._cfg = config or RetrievalConfig()

    def retrieve(self, query: str, agent_id: str) -> list[RetrievedMemory]:
        """主入口，返回按综合得分排序的记忆列表。 / Main entry, returns memories sorted by composite score."""
        # 1. FTS5 宽召回 / wide recall
        candidates = self._store.fts_search(
            query, agent_id, limit=self._cfg.fts_candidate_limit
        )

        if not candidates:
            return []

        # 2. BM25 分归一化到 [0, 1] / normalize BM25 to [0, 1]
        max_bm25 = max(score for _, score in candidates) or 1.0
        now = time.time()

        # 3. 计算综合得分 / compute composite score
        scored: list[RetrievedMemory] = []
        for entry, bm25_score in candidates:
            bm25_norm = bm25_score / max_bm25
            recency   = self._time_decay(entry.accessed_at, now)
            score = (
                self._cfg.bm25_weight       * bm25_norm +
                self._cfg.recency_weight    * recency   +
                self._cfg.importance_weight * entry.importance
            )
            if score >= self._cfg.min_score_threshold:
                scored.append(RetrievedMemory(
                    entry=entry,
                    score=round(score, 4),
                    reason=f"bm25={bm25_norm:.2f} recency={recency:.2f} imp={entry.importance:.2f}",
                ))

        # 4. 排序取 Top-K，顺手 touch 被召回的记忆 / sort, take Top-K, and touch recalled memories
        top = sorted(scored, key=lambda r: r.score, reverse=True)[: self._cfg.top_k]
        for r in top:
            self._store.touch(r.entry.id)

        return top

    def retrieve_with_sessions(
        self, query: str, agent_id: str
    ) -> tuple[list[RetrievedMemory], list[str]]:
        """
        返回 (记忆列表, 会话摘要列表)。 / Returns (memory list, session summary list).
        摘要单独返回，由 MemoryManager 决定如何拼接到 context。 / Summaries returned separately; MemoryManager decides how to splice into context.
        """
        memories = self.retrieve(query, agent_id)
        sessions = self._store.get_recent_sessions(
            agent_id, limit=self._cfg.always_inject_recent_sessions
        )
        session_summaries = [s.summary for s in sessions]
        return memories, session_summaries

    def format_for_context(
        self,
        memories: list[RetrievedMemory],
        session_summaries: list[str],
    ) -> str:
        """
        把检索结果拼成注入 system prompt 的文字块。 / Assemble retrieval results into a text block injected into the system prompt.
        格式化为自然语言，方便 LLM 直接理解。 / Formatted as natural language for direct LLM comprehension.
        """
        parts: list[str] = []

        if session_summaries:
            parts.append("【近期会话摘要】")
            for i, s in enumerate(session_summaries, 1):
                parts.append(f"{i}. {s}")

        if memories:
            parts.append("【相关记忆】")
            for r in memories:
                tag = r.entry.memory_type.value
                parts.append(f"[{tag}] {r.entry.content}")

        return "\n".join(parts) if parts else ""

    # ── 时间衰减函数 / Time decay function ──────────────────────────────────────────────────────────

    def _time_decay(self, accessed_at: float, now: float) -> float:
        """
        指数衰减：score = e^(-λ×天数) / Exponential decay: score = e^(-λ×days)
        半衰期 = time_decay_days，即经过半衰期后得分衰减到 0.5。 / Half-life = time_decay_days; after one half-life the score decays to 0.5.

        λ = ln(2) / 半衰期 / λ = ln(2) / half-life
        """
        days_ago = (now - accessed_at) / 86400.0
        lam = math.log(2) / max(self._cfg.time_decay_days, 1.0)
        return math.exp(-lam * days_ago)

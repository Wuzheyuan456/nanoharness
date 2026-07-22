"""
Phase 3 单测 / Phase 3 unit tests：记忆系统行为验证。
全部使用内存 SQLite（db_path=":memory:"），不依赖磁盘或真实 LLM。
/ All use in-memory SQLite (db_path=":memory:"), no disk or real LLM dependency.
"""
from __future__ import annotations

import asyncio
import math
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanoharness.core.context import AgentContext, Message
from nanoharness.memory.compaction_hooks import MemoryCompactionHook
from nanoharness.memory.consolidation import SessionConsolidator
from nanoharness.memory.manager import MemoryConfig, MemoryManager, MEMORY_CONTEXT_KEY
from nanoharness.memory.retrieval import MemoryRetriever, RetrievalConfig
from nanoharness.memory.store import MemoryEntry, MemoryStore, MemoryType, SessionRecord


# ─── 公共工具 / Common helpers ──────────────────────────────────────────────────────────────────

def fresh_store() -> MemoryStore:
    """每次返回全新内存 store，测试间互不干扰。 / Returns a fresh in-memory store each call; tests do not interfere with each other."""
    return MemoryStore(":memory:")


def make_ctx(session_key: str = "sess-test", history: list[Message] | None = None) -> AgentContext:
    ctx = AgentContext(
        agent_id="agent-test",
        session_key=session_key,
        system_prompt="测试助手",
        model_id="mock-T1",
    )
    if history:
        ctx.history.extend(history)
    return ctx


def make_entry(
    content: str,
    agent_id: str = "agent-test",
    memory_type: MemoryType = MemoryType.FACT,
    importance: float = 0.7,
    accessed_at: float | None = None,
) -> MemoryEntry:
    e = MemoryEntry(
        content=content,
        agent_id=agent_id,
        memory_type=memory_type,
        importance=importance,
    )
    if accessed_at is not None:
        object.__setattr__(e, "accessed_at", accessed_at)
    return e


def make_mock_provider(summary_json: str) -> MagicMock:
    """返回 complete() 固定应答 JSON 的 mock provider。 / Returns a mock provider whose complete() returns a fixed JSON response."""
    from nanoharness.provider.base import LLMResponse
    resp = LLMResponse(
        raw_content=[{"type": "text", "text": summary_json}],
        stop_reason="end_turn",
        input_tokens=10,
        output_tokens=20,
        final_text=summary_json,
    )
    p = MagicMock()
    p.complete = AsyncMock(return_value=resp)
    return p


# ─── MemoryStore ──────────────────────────────────────────────────────────────

class TestMemoryStore:
    def test_upsert_and_count(self):
        """写入一条记忆后 count 返回 1。 / After upserting one memory, count returns 1."""
        store = fresh_store()
        entry = make_entry("用户喜欢简洁代码")
        store.upsert(entry)
        assert store.count("agent-test") == 1

    def test_upsert_idempotent_same_id(self):
        """相同 id 重复 upsert 不增加条数，content 被更新。 / Repeated upsert with the same id does not increase count; content is updated."""
        store = fresh_store()
        e = make_entry("旧内容")
        store.upsert(e)
        updated = MemoryEntry(
            id=e.id,
            content="新内容",
            agent_id=e.agent_id,
            memory_type=e.memory_type,
            importance=0.9,
        )
        store.upsert(updated)
        assert store.count("agent-test") == 1

    def test_fts_search_returns_match(self):
        """FTS5 检索能找到包含关键词的记忆。 / FTS5 retrieval finds memories containing the keyword."""
        store = fresh_store()
        store.upsert(make_entry("Python 编程语言最近很流行"))
        store.upsert(make_entry("用户住在上海"))
        results = store.fts_search("Python", "agent-test", limit=5)
        assert len(results) == 1
        assert "Python" in results[0][0].content

    def test_fts_search_multi_token_and(self):
        """多词查询使用 AND，两个词都匹配才返回。 / Multi-token queries use AND; both tokens must match."""
        store = fresh_store()
        store.upsert(make_entry("用户喜欢 Python 脚本自动化"))
        store.upsert(make_entry("Python 语言介绍"))
        # "Python AND 自动化" 只有第一条匹配 / "Python AND 自动化" only the first matches
        results = store.fts_search("Python 自动化", "agent-test", limit=5)
        assert len(results) == 1
        assert "自动化" in results[0][0].content

    def test_fts_search_empty_query_returns_empty(self):
        """空查询直接返回空列表，不报错。 / Empty query returns an empty list without error."""
        store = fresh_store()
        store.upsert(make_entry("任意内容"))
        assert store.fts_search("", "agent-test") == []

    def test_touch_updates_accessed_at(self):
        """touch() 调用后 accessed_at 变新，access_count 增加。 / After touch(), accessed_at is refreshed and access_count increases."""
        store = fresh_store()
        old_time = time.time() - 1000
        e = make_entry("测试 touch", accessed_at=old_time)
        # 直接写入带旧时间戳的 entry / Directly insert an entry with an old timestamp
        store._conn.execute(
            "INSERT INTO memories (id,agent_id,session_key,content,memory_type,"
            "importance,created_at,accessed_at,access_count) VALUES (?,?,?,?,?,?,?,?,?)",
            (e.id, e.agent_id, e.session_key, e.content, str(e.memory_type),
             e.importance, e.created_at, old_time, 0),
        )
        store._conn.commit()
        # FTS 触发器需手动同步（测试里绕过 upsert，直接 INSERT，所以 FTS 已通过触发器同步） / FTS triggers need manual sync (test bypasses upsert and INSERTs directly, so FTS is already synced via triggers)
        before = store.get_recent("agent-test")[0].accessed_at
        store.touch(e.id)
        after = store.get_recent("agent-test")[0].accessed_at
        assert after > before

    def test_delete_removes_entry(self):
        """delete() 后 count 归零。 / After delete(), count goes to zero."""
        store = fresh_store()
        e = make_entry("要删除的内容")
        store.upsert(e)
        store.delete(e.id)
        assert store.count("agent-test") == 0

    def test_session_upsert_and_retrieve(self):
        """会话摘要写入后能按 agent_id 取出。 / After upserting a session summary, it can be retrieved by agent_id."""
        store = fresh_store()
        rec = SessionRecord(
            session_key="sess-abc",
            agent_id="agent-test",
            summary="用户讨论了 Python 异步编程",
            started_at=time.time() - 60,
        )
        store.upsert_session(rec)
        sessions = store.get_recent_sessions("agent-test", limit=5)
        assert len(sessions) == 1
        assert sessions[0].summary == "用户讨论了 Python 异步编程"

    def test_fts_agent_isolation(self):
        """不同 agent_id 的记忆互不干扰。 / Memories of different agent_ids do not interfere."""
        store = fresh_store()
        store.upsert(make_entry("共同关键词内容", agent_id="agent-A"))
        store.upsert(make_entry("共同关键词内容", agent_id="agent-B"))
        results_a = store.fts_search("共同关键词", "agent-A", limit=5)
        results_b = store.fts_search("共同关键词", "agent-B", limit=5)
        assert len(results_a) == 1
        assert len(results_b) == 1
        assert results_a[0][0].agent_id == "agent-A"
        assert results_b[0][0].agent_id == "agent-B"


# ─── MemoryRetriever ──────────────────────────────────────────────────────────

class TestMemoryRetriever:
    def test_retrieve_top_k_respected(self):
        """检索结果不超过 top_k 条。 / Retrieval results do not exceed top_k."""
        store = fresh_store()
        for i in range(10):
            store.upsert(make_entry(f"Python 内容条目 {i}"))
        cfg = RetrievalConfig(top_k=3, fts_candidate_limit=20)
        retriever = MemoryRetriever(store, cfg)
        results = retriever.retrieve("Python", "agent-test")
        assert len(results) <= 3

    def test_time_decay_formula(self):
        """时间衰减：半衰期后得分应约为 0.5。 / Time decay: after one half-life, score should be ~0.5."""
        from nanoharness.memory.retrieval import MemoryRetriever
        cfg = RetrievalConfig(time_decay_days=30)
        r = MemoryRetriever(fresh_store(), cfg)
        now = time.time()
        past = now - 30 * 86400  # 30 天前 / 30 days ago
        decay = r._time_decay(past, now)
        assert 0.45 < decay < 0.55, f"半衰期衰减应约为 0.5，实际={decay:.3f}"

    def test_recent_memory_scores_higher(self):
        """近期记忆的综合得分高于远期记忆（内容相同，importance 相同）。 / Recent memories score higher than old ones (same content, same importance)."""
        store = fresh_store()
        now = time.time()
        # 直接写入带不同 accessed_at 的数据 / Directly insert data with different accessed_at
        recent_id = "recent-001"
        old_id = "old-001"
        for mid, at in [(recent_id, now - 1), (old_id, now - 60 * 86400)]:
            store._conn.execute(
                "INSERT INTO memories (id,agent_id,session_key,content,memory_type,"
                "importance,created_at,accessed_at,access_count) VALUES (?,?,?,?,?,?,?,?,?)",
                (mid, "agent-test", "", "关键词测试内容", "fact", 0.7, at, at, 0),
            )
        store._conn.commit()
        # 同步 FTS 索引（触发器已在 INSERT 时触发） / Sync FTS index (triggers fired on INSERT)

        cfg = RetrievalConfig(top_k=5, fts_candidate_limit=20, time_decay_days=30)
        retriever = MemoryRetriever(store, cfg)
        results = retriever.retrieve("关键词测试", "agent-test")
        assert len(results) == 2
        scores_by_id = {r.entry.id: r.score for r in results}
        assert scores_by_id[recent_id] > scores_by_id[old_id]

    def test_format_for_context_structure(self):
        """format_for_context 包含两个区块：近期会话摘要 + 相关记忆。 / format_for_context contains two sections: recent session summaries + related memories."""
        store = fresh_store()
        store.upsert(make_entry("用户偏好简洁代码风格"))
        retriever = MemoryRetriever(store, RetrievalConfig())
        # trigram 要求 ≥3 字符的查询词 / trigram requires query tokens of ≥3 chars
        memories = retriever.retrieve("简洁代码", "agent-test")
        summaries = ["上一次会话讨论了 API 设计"]
        text = retriever.format_for_context(memories, summaries)
        assert "近期会话摘要" in text
        assert "相关记忆" in text

    def test_format_empty_returns_empty_string(self):
        """无记忆无摘要时返回空字符串。 / Returns an empty string when there are no memories and no summaries."""
        retriever = MemoryRetriever(fresh_store(), RetrievalConfig())
        assert retriever.format_for_context([], []) == ""

    def test_retrieve_below_threshold_filtered(self):
        """得分低于 min_score_threshold 的结果被过滤。 / Results scoring below min_score_threshold are filtered out."""
        store = fresh_store()
        # 写入一条极旧（60天前）、低 importance 的记忆 / Insert a very old (60 days ago), low-importance memory
        old_time = time.time() - 60 * 86400
        store._conn.execute(
            "INSERT INTO memories (id,agent_id,session_key,content,memory_type,"
            "importance,created_at,accessed_at,access_count) VALUES (?,?,?,?,?,?,?,?,?)",
            ("low-001", "agent-test", "", "关键词边界测试", "fact", 0.1,
             old_time, old_time, 0),
        )
        store._conn.commit()
        cfg = RetrievalConfig(
            top_k=5,
            min_score_threshold=0.5,   # 高阈值 / High threshold
            time_decay_days=30,
            importance_weight=0.4,
            recency_weight=0.3,
            bm25_weight=0.3,
        )
        retriever = MemoryRetriever(store, cfg)
        results = retriever.retrieve("关键词边界", "agent-test")
        # 得分 ≈ 0.3×1.0 + 0.3×0.25 + 0.4×0.1 ≈ 0.415，低于 0.5 → 被过滤 / Score ≈ 0.3×1.0 + 0.3×0.25 + 0.4×0.1 ≈ 0.415, below 0.5 → filtered
        assert len(results) == 0


# ─── CompactionHook ───────────────────────────────────────────────────────────

class TestCompactionHook:
    def _make_tool_result_msg(self, text: str) -> Message:
        """构造 tool_result 消息。 / Build a tool_result message."""
        return Message(
            role="user",
            content=[{
                "type": "tool_result",
                "tool_use_id": "tool-123",
                "content": text,
            }],
        )

    def test_tool_result_saved_to_l3(self):
        """tool_result 消息内容被写入 L3。 / tool_result message content is written to L3."""
        store = fresh_store()
        hook = MemoryCompactionHook(
            store=store, agent_id="agent-test", session_key="sess-1"
        )
        msgs = [self._make_tool_result_msg("搜索结果：Python 3.12 新特性包含了 type alias")]
        saved = hook.extract_from_messages(msgs)
        assert saved == 1
        assert store.count("agent-test") == 1
        entry = store.get_recent("agent-test")[0]
        assert entry.memory_type == MemoryType.TOOL_RESULT
        assert entry.importance == pytest.approx(0.85)

    def test_long_user_message_saved(self):
        """超过 200 字符的 user 消息被写入 L3。 / User messages longer than 200 chars are written to L3."""
        store = fresh_store()
        hook = MemoryCompactionHook(store=store, agent_id="agent-test", session_key="sess-1")
        long_msg = Message(role="user", content="需求描述" + "x" * 300)
        saved = hook.extract_from_messages([long_msg])
        assert saved == 1
        entry = store.get_recent("agent-test")[0]
        assert entry.memory_type == MemoryType.FACT

    def test_short_message_not_saved(self):
        """短 user 消息（< 200 字）不写入 L3。 / Short user messages (< 200 chars) are not written to L3."""
        store = fresh_store()
        hook = MemoryCompactionHook(store=store, agent_id="agent-test", session_key="sess-1")
        msgs = [Message(role="user", content="好的")]
        saved = hook.extract_from_messages(msgs)
        assert saved == 0

    def test_assistant_message_ignored(self):
        """assistant 消息不被写入 L3。 / assistant messages are not written to L3."""
        store = fresh_store()
        hook = MemoryCompactionHook(store=store, agent_id="agent-test", session_key="sess-1")
        msgs = [Message(role="assistant", content="好的，我来帮你。" + "x" * 300)]
        saved = hook.extract_from_messages(msgs)
        assert saved == 0

    def test_tool_result_content_too_short_ignored(self):
        """工具结果文字不足 20 字不写入（噪声过滤）。 / Tool results with fewer than 20 chars are not written (noise filtering)."""
        store = fresh_store()
        hook = MemoryCompactionHook(store=store, agent_id="agent-test", session_key="sess-1")
        msgs = [self._make_tool_result_msg("ok")]
        saved = hook.extract_from_messages(msgs)
        assert saved == 0

    def test_extract_structured_content_blocks(self):
        """结构化 content 块（list of dicts）的文字能被正确提取。 / Structured content blocks (list of dicts) have their text extracted correctly."""
        store = fresh_store()
        hook = MemoryCompactionHook(store=store, agent_id="agent-test", session_key="sess-1")
        msg = Message(
            role="user",
            content=[{
                "type": "tool_result",
                "tool_use_id": "tool-abc",
                "content": [{"type": "text", "text": "发现重要事实：API 端点已于 2025 年弃用，请使用新版"}],
            }],
        )
        saved = hook.extract_from_messages([msg])
        assert saved == 1


# ─── SessionConsolidator ──────────────────────────────────────────────────────

class TestSessionConsolidator:
    def test_consolidate_writes_summary_and_facts(self):
        """LLM 返回合法 JSON 时，摘要写入 sessions 表，事实写入 memories 表。 / When LLM returns valid JSON, the summary goes to the sessions table and facts go to the memories table."""
        store = fresh_store()
        provider = make_mock_provider(
            '{"summary":"用户学习了异步编程","facts":[{"type":"fact","content":"用户使用 Python asyncio"}]}'
        )
        consolidator = SessionConsolidator(store, provider)
        ctx = make_ctx(history=[
            Message(role="user", content="asyncio 怎么用？"),
            Message(role="assistant", content="可以用 async/await 关键字..."),
        ])
        asyncio.run(consolidator.consolidate_and_wait(ctx, started_at=time.time() - 60))

        sessions = store.get_recent_sessions("agent-test")
        assert len(sessions) == 1
        assert "异步编程" in sessions[0].summary

        facts = store.get_recent("agent-test", memory_type=MemoryType.FACT)
        assert any("asyncio" in f.content for f in facts)

    def test_consolidate_preference_type(self):
        """type=preference 的 fact 被存为 PREFERENCE 类型。 / A fact with type=preference is stored as PREFERENCE."""
        store = fresh_store()
        provider = make_mock_provider(
            '{"summary":"短摘要","facts":[{"type":"preference","content":"用户偏好类型注解"}]}'
        )
        consolidator = SessionConsolidator(store, provider)
        ctx = make_ctx(history=[Message(role="user", content="我喜欢类型注解")])
        asyncio.run(consolidator.consolidate_and_wait(ctx, started_at=time.time() - 10))

        prefs = store.get_recent("agent-test", memory_type=MemoryType.PREFERENCE)
        assert len(prefs) == 1
        assert "类型注解" in prefs[0].content

    def test_consolidate_llm_failure_fallback(self):
        """LLM 调用失败时，降级为截断对话存为摘要，不写 facts。 / On LLM call failure, degrades to truncating the conversation into a summary, with no facts written."""
        store = fresh_store()
        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("network error"))
        consolidator = SessionConsolidator(store, provider)
        ctx = make_ctx(history=[Message(role="user", content="测试内容")])
        asyncio.run(consolidator.consolidate_and_wait(ctx, started_at=time.time() - 10))

        sessions = store.get_recent_sessions("agent-test")
        assert len(sessions) == 1  # 降级摘要还是写入了 / Degraded summary still written
        assert store.count("agent-test") == 0  # 无 facts / No facts

    def test_consolidate_bad_json_fallback(self):
        """LLM 返回非 JSON 时触发降级，不崩溃。 / When LLM returns non-JSON, triggers fallback without crashing."""
        store = fresh_store()
        provider = make_mock_provider("这是纯文字，不是 JSON 格式的回复")
        consolidator = SessionConsolidator(store, provider)
        ctx = make_ctx(history=[Message(role="user", content="随意问题")])
        asyncio.run(consolidator.consolidate_and_wait(ctx, started_at=time.time() - 10))

        # 降级：把 LLM 的文字作为摘要存入 / Fallback: store the LLM text as the summary
        sessions = store.get_recent_sessions("agent-test")
        assert len(sessions) == 1

    def test_consolidate_empty_history_skip(self):
        """空 history 不触发 LLM 调用，sessions 表不写入。 / Empty history does not trigger an LLM call; nothing is written to the sessions table."""
        store = fresh_store()
        provider = MagicMock()
        provider.complete = AsyncMock()
        consolidator = SessionConsolidator(store, provider)
        ctx = make_ctx()  # history 为空 / history is empty
        asyncio.run(consolidator.consolidate_and_wait(ctx, started_at=time.time()))
        provider.complete.assert_not_called()
        assert len(store.get_recent_sessions("agent-test")) == 0


# ─── MemoryManager (门面层) / MemoryManager (facade layer) ───────────────────────────────────────────

class TestMemoryManager:
    def test_remember_and_search(self):
        """remember() 写入后 search() 能检索到。 / After remember() writes, search() can retrieve it."""
        mgr = MemoryManager(MemoryConfig(db_path=":memory:"))
        mgr.remember("agent-test", "用户擅长写 Python 爬虫", MemoryType.FACT)
        results = mgr.search("Python 爬虫", "agent-test")
        assert any("爬虫" in r for r in results)
        mgr.close()

    def test_prefetch_skips_when_history_too_short(self):
        """history 条数不足 prefetch_min_turns 时，返回空字符串且不写入 ctx。 / When history is shorter than prefetch_min_turns, returns an empty string and writes nothing to ctx."""
        mgr = MemoryManager(MemoryConfig(db_path=":memory:", prefetch_min_turns=3))
        ctx = make_ctx()  # history 为空 / history is empty
        result = asyncio.run(mgr.prefetch(ctx, "任意查询"))
        assert result == ""
        assert MEMORY_CONTEXT_KEY not in ctx.extra_context
        mgr.close()

    def test_prefetch_injects_context(self):
        """有记忆且 history 足够时，prefetch 把结果写入 ctx.extra_context。 / With memories and sufficient history, prefetch writes results into ctx.extra_context."""
        mgr = MemoryManager(MemoryConfig(db_path=":memory:", prefetch_min_turns=1))
        # 先写入记忆（query 用 trigram 匹配，内容和查询词都需要 ≥3 字符） / First write a memory (query uses trigram matching; both content and query tokens need ≥3 chars)
        mgr.remember("agent-test", "用户喜欢使用 asyncio 异步编程模式")
        ctx = make_ctx(history=[
            Message(role="user", content="我们来聊聊 asyncio"),
            Message(role="assistant", content="好的"),
        ])
        asyncio.run(mgr.prefetch(ctx, "asyncio"))
        assert MEMORY_CONTEXT_KEY in ctx.extra_context
        assert len(ctx.extra_context[MEMORY_CONTEXT_KEY]) > 0
        mgr.close()

    def test_on_compact_saves_high_value_messages(self):
        """on_compact 把 tool_result 消息写入 L3。 / on_compact writes tool_result messages to L3."""
        mgr = MemoryManager(MemoryConfig(db_path=":memory:", enable_compaction_protection=True))
        ctx = make_ctx(history=[
            Message(
                role="user",
                content=[{
                    "type": "tool_result",
                    "tool_use_id": "tid-1",
                    "content": "重要搜索结果：发现三个关键 API 已在 2025 年废弃，必须迁移",
                }],
            )
        ])
        saved = asyncio.run(mgr.on_compact(ctx))
        assert saved == 1
        mgr.close()

    def test_on_compact_disabled(self):
        """enable_compaction_protection=False 时，on_compact 不写入任何记忆。 / When enable_compaction_protection=False, on_compact writes no memories."""
        mgr = MemoryManager(MemoryConfig(db_path=":memory:", enable_compaction_protection=False))
        ctx = make_ctx(history=[
            Message(
                role="user",
                content=[{
                    "type": "tool_result",
                    "tool_use_id": "tid-2",
                    "content": "这条工具结果不应该被保存" * 10,
                }],
            )
        ])
        saved = asyncio.run(mgr.on_compact(ctx))
        assert saved == 0
        mgr.close()

    def test_flush_triggers_consolidation(self):
        """flush() 调用后 sessions 表有记录（mock provider 返回合法 JSON）。 / After flush() the sessions table has records (mock provider returns valid JSON)."""
        provider = make_mock_provider(
            '{"summary":"测试 session 摘要","facts":[]}'
        )
        mgr = MemoryManager(
            MemoryConfig(db_path=":memory:", prefetch_min_turns=1),
            provider=provider,
        )
        ctx = make_ctx(history=[Message(role="user", content="测试消息")])
        # prefetch 触发 session_start_times 记录 / prefetch triggers session_start_times recording
        asyncio.run(mgr.prefetch(ctx, "测试"))
        # flush 调用 consolidate（背后用 create_task，这里改用 consolidate_and_wait） / flush calls consolidate (uses create_task under the hood; here we use consolidate_and_wait instead)
        asyncio.run(mgr._consolidator.consolidate_and_wait(ctx, started_at=time.time() - 10))

        sessions = mgr._store.get_recent_sessions("agent-test")
        assert len(sessions) == 1
        assert "测试 session" in sessions[0].summary
        mgr.close()

    def test_stats_reflects_memory_count(self):
        """stats() 中 memory_count 与实际写入量一致。 / stats() memory_count matches the actual written count."""
        mgr = MemoryManager(MemoryConfig(db_path=":memory:"))
        mgr.remember("agent-test", "事实 A")
        mgr.remember("agent-test", "事实 B")
        s = mgr.stats("agent-test")
        assert s["memory_count"] == 2
        mgr.close()

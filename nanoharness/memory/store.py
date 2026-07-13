from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Iterator


# ─── 记忆类型 ──────────────────────────────────────────────────────────────────

class MemoryType(StrEnum):
    FACT       = "fact"        # 客观事实（"用户叫 Alex"）
    PREFERENCE = "preference"  # 用户偏好（"喜欢简洁代码风格"）
    SUMMARY    = "summary"     # 会话摘要
    TOOL_RESULT = "tool_result"  # 重要工具结果（"搜索发现 xxx API 已废弃"）


# ─── 数据模型 ──────────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    content: str
    agent_id: str
    memory_type: MemoryType
    importance: float = 0.5          # 0.0 ~ 1.0，写入时由调用方评估
    session_key: str = ""            # 来源 session，可为空（全局事实）
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class SessionRecord:
    session_key: str
    agent_id: str
    summary: str
    started_at: float
    ended_at: float = field(default_factory=time.time)


# ─── MemoryStore ──────────────────────────────────────────────────────────────

class MemoryStore:
    """
    SQLite 存储层，含 FTS5 全文索引（替代向量库）。

    两张表：
    - memories   : L3 长期记忆条目（fact / preference / tool_result）
    - sessions   : L2 会话摘要
    - memories_fts: FTS5 虚拟表，提供 BM25 全文检索

    面试话术：
    "我没用 sqlite-vec，用了 SQLite 内置的 FTS5。
    记忆检索本质是关键词触发，FTS5 内置 BM25 零额外依赖，
    <10000 条记忆的场景查询延迟 <5ms，完全够用。"
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS memories (
        id           TEXT PRIMARY KEY,
        agent_id     TEXT NOT NULL,
        session_key  TEXT DEFAULT '',
        content      TEXT NOT NULL,
        memory_type  TEXT NOT NULL,
        importance   REAL DEFAULT 0.5,
        created_at   REAL NOT NULL,
        accessed_at  REAL NOT NULL,
        access_count INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS sessions (
        session_key  TEXT PRIMARY KEY,
        agent_id     TEXT NOT NULL,
        summary      TEXT NOT NULL,
        started_at   REAL NOT NULL,
        ended_at     REAL NOT NULL
    );

    -- FTS5：trigram tokenizer 支持中文子串匹配（无需 jieba 分词）
    -- content= 指向源表，触发器保持同步
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, content='memories', content_rowid='rowid',
                   tokenize='trigram');

    -- 触发器：保持 FTS5 与 memories 表同步
    CREATE TRIGGER IF NOT EXISTS memories_ai
        AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
        END;

    CREATE TRIGGER IF NOT EXISTS memories_ad
        AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
        END;

    CREATE TRIGGER IF NOT EXISTS memories_au
        AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
        END;
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._path = str(db_path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self._DDL)
        self._conn.commit()

    # ── 长期记忆 写 ───────────────────────────────────────────────────────────

    def upsert(self, entry: MemoryEntry) -> None:
        """写入或覆盖更新（按 id 去重）。"""
        self._conn.execute(
            """
            INSERT INTO memories
                (id, agent_id, session_key, content, memory_type,
                 importance, created_at, accessed_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                content      = excluded.content,
                importance   = excluded.importance,
                accessed_at  = excluded.accessed_at,
                access_count = access_count + 1
            """,
            (
                entry.id, entry.agent_id, entry.session_key,
                entry.content, str(entry.memory_type),
                entry.importance, entry.created_at,
                entry.accessed_at, entry.access_count,
            ),
        )
        self._conn.commit()

    def delete(self, memory_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self._conn.commit()

    # ── 长期记忆 读 ───────────────────────────────────────────────────────────

    def fts_search(
        self,
        query: str,
        agent_id: str,
        limit: int = 20,
    ) -> list[tuple[MemoryEntry, float]]:
        """
        FTS5 全文检索，返回 (entry, bm25_score) 列表。
        bm25() 返回负值（越小越相关），这里取绝对值方便后续加权。
        """
        if not query.strip():
            return []
        rows = self._conn.execute(
            """
            SELECT m.*, -bm25(memories_fts) AS score
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.rowid
            WHERE memories_fts MATCH ?
              AND m.agent_id = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (self._fts_escape(query), agent_id, limit),
        ).fetchall()
        return [(self._row_to_entry(r), r["score"]) for r in rows]

    def get_recent(
        self,
        agent_id: str,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        """按 accessed_at 倒序取最近记录，可按类型过滤。"""
        if memory_type:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE agent_id=? AND memory_type=? "
                "ORDER BY accessed_at DESC LIMIT ?",
                (agent_id, str(memory_type), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE agent_id=? "
                "ORDER BY accessed_at DESC LIMIT ?",
                (agent_id, limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def touch(self, memory_id: str) -> None:
        """标记访问时间 + 计数（召回时调用）。"""
        self._conn.execute(
            "UPDATE memories SET accessed_at=?, access_count=access_count+1 WHERE id=?",
            (time.time(), memory_id),
        )
        self._conn.commit()

    # ── 会话摘要 写/读 ────────────────────────────────────────────────────────

    def upsert_session(self, record: SessionRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO sessions (session_key, agent_id, summary, started_at, ended_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                summary   = excluded.summary,
                ended_at  = excluded.ended_at
            """,
            (record.session_key, record.agent_id,
             record.summary, record.started_at, record.ended_at),
        )
        self._conn.commit()

    def get_recent_sessions(self, agent_id: str, limit: int = 5) -> list[SessionRecord]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE agent_id=? ORDER BY ended_at DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def count(self, agent_id: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE agent_id=?", (agent_id,)
        ).fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fts_escape(query: str) -> str:
        """
        为 trigram tokenizer 构造 FTS5 MATCH 查询。
        trigram 要求每个 token 至少 3 个字符，短于 3 的词静默跳过。
        多词用 AND 连接，提高精确度。
        """
        tokens = [t.strip() for t in query.split() if len(t.strip()) >= 3]
        if not tokens:
            return '""'
        escaped = [f'"{t}"' for t in tokens]
        return " AND ".join(escaped)

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            agent_id=row["agent_id"],
            session_key=row["session_key"] or "",
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            importance=row["importance"],
            created_at=row["created_at"],
            accessed_at=row["accessed_at"],
            access_count=row["access_count"],
        )

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_key=row["session_key"],
            agent_id=row["agent_id"],
            summary=row["summary"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
        )

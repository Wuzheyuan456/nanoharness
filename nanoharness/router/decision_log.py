from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from nanoharness.router.tiers import Tier


# ─── 路由决策记录 / Routing decision record ──────────────────────────────────────────────────────────────

@dataclass
class RouterDecision:
    trace_id: str
    session_key: str
    input_preview: str       # 用户消息前 100 字符 / first 100 chars of user message
    tier: Tier
    confidence: float        # 0.0 ~ 1.0，LLM 返回的置信度 / 0.0 ~ 1.0, confidence returned by LLM
    reason: str              # LLM 的分类理由（一句话） / LLM classification reason (one line)
    model_used: str          # 实际使用的模型 ID / model ID actually used
    method: str              # "llm" | "heuristic" | "fallback" / classification method
    latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)


# ─── DecisionLog：SQLite 持久化 / DecisionLog: SQLite persistence ────────────────────────────────────────────────

class DecisionLog:
    """
    路由决策的 append-only 持久化日志。 / Append-only persistent log of routing decisions.

    用途： / Use cases:
    - 离线分析路由准确率（人工标注 + 对比真实输出质量） / - Offline routing accuracy analysis (manual labeling + comparing real output quality)
    - 计算 cost 节省数据（T0 vs T1 单价差 × 调用次数） / - Compute cost savings (T0 vs T1 unit-price diff × call count)
    - Gradio 面板"路由决策可视化"的数据源 / - Data source for the Gradio "routing decision visualization" panel

    面试话术 / Interview talking point:
    "我把每次路由决策写入 SQLite，记录 tier / confidence / 分类理由 / 延迟。
    跑完 50 条测试集后可以出一张准确率报告和 cost 节省曲线，
    这是 LLM Router 设计合不合理的量化依据。"
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS router_decisions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id    TEXT NOT NULL,
        session_key TEXT NOT NULL,
        input_preview TEXT,
        tier        TEXT NOT NULL,
        confidence  REAL,
        reason      TEXT,
        model_used  TEXT,
        method      TEXT,
        latency_ms  REAL,
        ts          REAL
    );
    CREATE INDEX IF NOT EXISTS idx_trace ON router_decisions(trace_id);
    CREATE INDEX IF NOT EXISTS idx_session ON router_decisions(session_key);
    CREATE INDEX IF NOT EXISTS idx_tier ON router_decisions(tier);
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(self.DDL)
        self._conn.commit()

    def append(self, decision: RouterDecision) -> None:
        self._conn.execute(
            """
            INSERT INTO router_decisions
              (trace_id, session_key, input_preview, tier, confidence,
               reason, model_used, method, latency_ms, ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.trace_id,
                decision.session_key,
                decision.input_preview,
                str(decision.tier),
                decision.confidence,
                decision.reason,
                decision.model_used,
                decision.method,
                decision.latency_ms,
                decision.ts,
            ),
        )
        self._conn.commit()

    def query_by_session(self, session_key: str) -> list[RouterDecision]:
        rows = self._conn.execute(
            "SELECT * FROM router_decisions WHERE session_key=? ORDER BY ts",
            (session_key,),
        ).fetchall()
        return [self._row_to_decision(r) for r in rows]

    def cost_savings_report(self) -> dict:
        """
        统计各档位调用次数，估算相对全用 T2 的成本节省比例。 / Count calls per tier, estimate cost-savings ratio vs all-T2 baseline.
        基线：全用 T2（中等复杂度模型）。 / Baseline: all T2 (medium complexity model).
        粗略假设：T0 成本 = T2 × 0.033，T1 = T2 × 0.33，T3 = T2 × 5。 / Rough assumption: T0 cost = T2 × 0.033, T1 = T2 × 0.33, T3 = T2 × 5.
        """
        # 相对 T2 的成本比例 / Relative cost ratio to T2
        RELATIVE_COST = {
            Tier.T0: 0.033,  # 0.1 / 3
            Tier.T1: 0.33,   # 1.0 / 3
            Tier.T2: 1.0,
            Tier.T3: 5.0,    # 15.0 / 3
        }
        rows = self._conn.execute(
            "SELECT tier, COUNT(*) as cnt FROM router_decisions GROUP BY tier"
        ).fetchall()

        counts = {Tier(r[0]): r[1] for r in rows}
        total = sum(counts.values()) or 1
        actual_cost = sum(RELATIVE_COST.get(t, 1.0) * n for t, n in counts.items())
        baseline_cost = total * 1.0  # 假设全用 T2

        return {
            "total_calls": total,
            "tier_breakdown": {str(t): n for t, n in counts.items()},
            "actual_relative_cost": round(actual_cost, 2),
            "baseline_relative_cost": round(baseline_cost, 2),
            "savings_pct": round((1 - actual_cost / baseline_cost) * 100, 1),
        }

    def iter_all(self) -> Iterator[RouterDecision]:
        for row in self._conn.execute("SELECT * FROM router_decisions ORDER BY ts"):
            yield self._row_to_decision(row)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_decision(row: tuple) -> RouterDecision:
        # id, trace_id, session_key, input_preview, tier, confidence, / 行字段顺序
        # reason, model_used, method, latency_ms, ts / row field order
        return RouterDecision(
            trace_id=row[1],
            session_key=row[2],
            input_preview=row[3] or "",
            tier=Tier(row[4]),
            confidence=row[5] or 0.0,
            reason=row[6] or "",
            model_used=row[7] or "",
            method=row[8] or "",
            latency_ms=row[9] or 0.0,
            ts=row[10] or 0.0,
        )

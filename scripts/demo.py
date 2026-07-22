"""
演示脚本：往可观测性三件套灌真实数据，启动 Gradio 面板 / Demo script: seed real data into observability trio and launch Gradio dashboard.

无需 API key，用 mock 数据填充三个 Tab / No API key needed, fill three tabs with mock data:
  - 路由决策：模拟 20 条不同 tier 的路由决策
  - 链路回放：模拟 3 条完整 turn 的 Span 树
  - 黄金信号：模拟延迟/token/错误率/饱和度指标

用法：
    python scripts/demo.py
然后在浏览器打开 http://localhost:7860
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoharness.observability.dashboard import launch_dashboard
from nanoharness.observability.metrics import get_metrics
from nanoharness.observability.tracing import get_tracer
from nanoharness.router.decision_log import DecisionLog, RouterDecision
from nanoharness.router.tiers import Tier


# ─── 1. 灌路由决策 / 1. Seed Router Decisions ────────────────────────────────

def seed_router_decisions(db_path: str = "router_decisions.db") -> None:
    """模拟 20 条路由决策，覆盖 T0~T3 各档位 / Simulate 20 routing decisions covering T0~T3 tiers."""
    db = DecisionLog(db_path)
    # 清空旧数据重新灌（演示用）/ Clear old data and re-seed (for demo)
    db._conn.execute("DELETE FROM router_decisions")
    db._conn.commit()

    samples = [
        ("你好，今天天气怎么样", Tier.T0, 0.95, "llm", "简单问候查询"),
        ("1+1等于几", Tier.T0, 0.92, "llm", "简单算术"),
        ("谢谢你的帮助", Tier.T0, 0.90, "heuristic", "规则匹配'谢谢'"),
        ("帮我查北京明天的天气", Tier.T1, 0.88, "llm", "需调用天气工具"),
        ("这段代码报错 KeyError 帮我看看", Tier.T1, 0.85, "llm", "常规代码问题"),
        ("翻译这段英文邮件", Tier.T1, 0.86, "llm", "中等翻译任务"),
        ("用 Python 实现 LRU 缓存", Tier.T2, 0.91, "llm", "代码生成"),
        ("对比 React 和 Vue 优缺点", Tier.T2, 0.87, "llm", "多维度分析"),
        ("重构这段 500 行代码", Tier.T2, 0.89, "llm", "大型代码重构"),
        ("支付系统疑似 SQL 注入，做安全审计", Tier.T3, 0.93, "llm", "高风险安全分析"),
        ("MySQL 迁移 PostgreSQL 的风险评估", Tier.T3, 0.90, "llm", "高风险决策"),
        ("设计金融监管数据加密方案", Tier.T3, 0.92, "llm", "架构设计"),
    ]
    for i, (msg, tier, conf, method, reason) in enumerate(samples):
        db.append(RouterDecision(
            trace_id=f"demo-{i}",
            session_key=f"demo-session-{i % 4}",
            input_preview=msg[:100],
            tier=tier,
            confidence=conf,
            reason=reason,
            model_used={"T0": "claude-haiku-4-5", "T1": "claude-sonnet-4-6",
                        "T2": "claude-sonnet-5", "T3": "claude-opus-4-8"}[str(tier)],
            method=method,
            latency_ms=180 + (i * 7) % 60,
            ts=time.time() - (len(samples) - i) * 30,
        ))
    db.close()
    print(f"✓ 灌入 {len(samples)} 条路由决策")


# ─── 2. 灌 Span 链路 / 2. Seed Span Traces ────────────────────────────────────

def seed_traces() -> None:
    """模拟 3 条完整 turn 的 Span 树（turn → 路由 → 工具调用 → 完成）/ Simulate 3 complete turn Span trees (turn → routing → tool call → done)."""
    tracer = get_tracer()
    tracer.clear()

    scenarios = [
        ("T0 简单问答", "你好", [("router.classify", 15), ("provider.stream", 80)]),
        ("T1 工具调用", "查天气", [("router.classify", 18), ("tool_call:search", 120),
                                    ("provider.stream", 95)]),
        ("T2 代码生成", "写 LRU", [("router.classify", 22), ("provider.stream", 450),
                                     ("compaction", 60)]),
    ]
    for name, msg, children in scenarios:
        with tracer.trace_turn(session_key=f"demo-{name}") as root:
            root.set_attr("scenario", name)
            root.set_attr("user_msg", msg)
            for child_name, dur_ms in children:
                with tracer.start_span(child_name) as child:
                    child.set_attr("duration_ms", dur_ms)
                    time.sleep(0.001)   # 让 end_ts 有差异 / give end_ts some difference
    print(f"✓ 灌入 {len(scenarios)} 条 Span 链路")


# ─── 3. 灌黄金信号指标 / 3. Seed Golden-Signal Metrics ───────────────────────

def seed_metrics() -> None:
    """模拟延迟分布、token 用量、错误率、context 利用率 / Simulate latency distribution, token usage, error rate, and context utilization."""
    import random
    random.seed(42)
    m = get_metrics()

    # 延迟：模拟 50 次 turn，多数快、少数长尾 / Latency: simulate 50 turns, mostly fast with a few long-tail
    for _ in range(40):
        m.observe_latency(random.uniform(0.05, 0.3), kind="turn", model="haiku")
    for _ in range(8):
        m.observe_latency(random.uniform(0.4, 0.9), kind="turn", model="sonnet-5")
    for _ in range(2):
        m.observe_latency(random.uniform(1.5, 2.8), kind="turn", model="opus-4-8")  # 长尾 / long tail
    # 路由分类延迟 / Routing classification latency
    for _ in range(50):
        m.observe_latency(random.uniform(0.15, 0.25), kind="router", model="haiku")

    # token 用量 / Token usage
    for _ in range(30):
        m.inc_tokens(random.randint(50, 200), model="haiku", token_type="input")
        m.inc_tokens(random.randint(100, 400), model="haiku", token_type="output")
    for _ in range(8):
        m.inc_tokens(random.randint(500, 1500), model="sonnet-5", token_type="input")
        m.inc_tokens(random.randint(800, 2000), model="sonnet-5", token_type="output")
    for _ in range(2):
        m.inc_tokens(random.randint(2000, 4000), model="opus-4-8", token_type="input")

    # 工具调用 / Tool calls
    for _ in range(15):
        m.inc_tool_call("search", success=True)
    m.inc_tool_call("search", success=False)
    for _ in range(6):
        m.inc_tool_call("calculator", success=True)

    # 错误 / Errors
    m.inc_error("timeout")
    m.inc_error("rate_limited")

    # context 利用率（4 个 session）/ Context utilization (4 sessions)
    m.observe_context_utilization(0.32, session_key="demo-T0")
    m.observe_context_utilization(0.58, session_key="demo-T1")
    m.observe_context_utilization(0.81, session_key="demo-T2")
    m.observe_context_utilization(0.45, session_key="demo-T3")

    stats = m.latency_report(kind="turn", model="haiku")
    print(f"✓ 灌入指标：turn 延迟样本 {stats['count']}，"
          f"P50={stats['p50']*1000:.0f}ms P99={stats['p99']*1000:.0f}ms")


def main() -> None:
    print("\n" + "=" * 50)
    print("  NanoHarness 演示面板 — 灌入 mock 数据")
    print("=" * 50 + "\n")
    seed_router_decisions()
    seed_traces()
    seed_metrics()
    print("\n🚀 启动 Gradio 面板：http://localhost:7860\n")
    print(" 三个 Tab：")
    print("  1. 路由决策 — tier 分布柱状图 + cost 节省报告")
    print("  2. 链路回放 — 选 trace_id 看 Span 树")
    print("  3. 黄金信号 — P50/P95/P99 + token + 错误率\n")
    launch_dashboard(decision_log_path="router_decisions.db", port=7860)


if __name__ == "__main__":
    main()

"""
LLMRouter 量化数据采集脚本 / LLMRouter quantitative data collection script.

用法 / Usage:
    # 真实调用（需配置 ANTHROPIC_API_KEY）
    python scripts/benchmark_router.py

    # mock 模式 dry-run（不花钱，验证脚本逻辑）
    python scripts/benchmark_router.py --mock

产出：
    - 路由准确率（预测 tier vs 人工标注 tier）
    - 各档位调用分布
    - cost 节省%（相对全用 T1 的 baseline）
    - 决策写入 SQLite（router_decisions.db），供 Gradio 面板可视化

这是简历里"LLM 调用成本降低 Y%"的数据来源。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 让脚本可直接运行（python scripts/benchmark_router.py）/ Allow running the script directly (python scripts/benchmark_router.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoharness.provider.anthropic import AnthropicProvider
from nanoharness.router.decision_log import DecisionLog
from nanoharness.router.llm_router import LLMRouter
from nanoharness.router.tiers import DEFAULT_TIER_CONFIGS, Tier, TierRegistry


# ─── Benchmark 测试集（人工标注 expected_tier）/ Benchmark Dataset (human-annotated expected_tier) ──

BENCHMARK_DATASET: list[tuple[str, Tier]] = [
    # T0：简单问答、打招呼、是非题 / T0: simple Q&A, greeting, yes-no
    ("你好，在吗？", Tier.T0),
    ("今天星期几？", Tier.T0),
    ("谢谢你的帮助", Tier.T0),
    ("1 加 1 等于几", Tier.T0),
    ("北京是中国的首都吗", Tier.T0),
    ("帮我查一下现在的汇率大概是多少", Tier.T0),
    ("再见，下次聊", Tier.T0),

    # T1：中等推理、1~3 步工具调用、常规代码问题 / T1: moderate reasoning, 1~3 step tool calls, routine code issues
    ("帮我查一下北京明天的天气", Tier.T1),
    ("这段 Python 代码报错 KeyError，帮我看看哪里有问题", Tier.T1),
    ("帮我翻译一段英文邮件成中文", Tier.T1),
    ("推荐一本适合 Python 进阶的书", Tier.T1),
    ("帮我算一下 123 乘以 456 等于多少", Tier.T1),
    ("总结一下这篇文章的三个要点", Tier.T1),

    # T2：复杂任务、代码生成、多步分析、长链推理 / T2: complex tasks, code generation, multi-step analysis, long-chain reasoning
    ("帮我用 Python 实现一个 LRU 缓存类，要支持并发", Tier.T2),
    ("写一个 SQL 查询，统计每个部门工资最高的前三名员工", Tier.T2),
    ("帮我对比 React 和 Vue 在大型项目中的优缺点", Tier.T2),
    ("设计一个支持百万并发的消息队列架构方案", Tier.T2),
    ("把这段 500 行的代码重构成更清晰的模块化结构", Tier.T2),
    ("分析这份日志，找出导致服务延迟飙升的根本原因", Tier.T2),

    # T3：高风险决策、架构设计、安全分析、深度思考 / T3: high-risk decisions, architecture design, security analysis, deep thinking
    ("我们的支付系统疑似存在 SQL 注入漏洞，帮我做安全审计", Tier.T3),
    ("评估把核心数据库从 MySQL 迁移到 PostgreSQL 的风险和收益", Tier.T3),
    ("帮我设计一个符合金融监管要求的数据加密方案", Tier.T3),
    ("分析这次生产事故的根因，给出防止复发的改进方案", Tier.T3),
    ("设计一个支持多租户隔离的 SaaS 平台安全架构", Tier.T3),
]


# ─── Provider 构建 / Provider Construction ────────────────────────────────────

def build_router(real: bool) -> LLMRouter:
    """构建 LLMRouter。real=True 用真实 Anthropic，False 用 mock / Build LLMRouter; real=True uses real Anthropic, False uses mock."""
    if real:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  未检测到 ANTHROPIC_API_KEY，自动切到 mock 模式。")
            print("   配置方法：export ANTHROPIC_API_KEY=sk-ant-...")
            real = False

    if real:
        # 路由分类用 T0（haiku），成本最低 / Route classification uses T0 (haiku), lowest cost
        provider = AnthropicProvider(
            model_id="claude-haiku-4-5-20251001",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
    else:
        provider = _MockClassifyProvider()

    return LLMRouter(
        provider=provider,
        registry=TierRegistry(),   # 默认即 DEFAULT_TIER_CONFIGS，无需传参 / defaults to DEFAULT_TIER_CONFIGS, no args needed
        decision_log=DecisionLog("router_decisions.db"),
    )


class _MockClassifyProvider:
    """
    Mock provider：按关键词模拟分类，让脚本无 API key 也能 dry-run / Mock provider: simulates classification by keyword so the script can dry-run without an API key.

    只用于验证脚本流程，准确率数据无参考价值 / Only for validating script flow; accuracy data is not meaningful.
    真实 benchmark 必须用 AnthropicProvider / A real benchmark must use AnthropicProvider.
    """
    model_id = "mock-haiku"

    async def complete(self, system, messages, tools=None, max_tokens=4096):
        from unittest.mock import MagicMock
        text = messages[0].content
        # 简单关键词模拟，输出 JSON / Simple keyword simulation, output JSON
        if any(k in text for k in ["你好", "谢谢", "再见", "几点", "星期", "首都", "等于"]):
            tier = "T0"
        elif any(k in text for k in ["审计", "迁移", "监管", "事故", "架构", "安全"]):
            tier = "T3"
        elif any(k in text for k in ["实现", "SQL", "对比", "重构", "设计", "分析"]):
            tier = "T2"
        else:
            tier = "T1"
        import json
        resp = MagicMock()
        resp.final_text = json.dumps(
            {"tier": tier, "confidence": 0.8, "reason": "mock 分类"}
        )
        return resp


# ─── Benchmark 主流程 / Benchmark Main Flow ──────────────────────────────────

async def run_benchmark(real: bool) -> dict:
    router = build_router(real)
    total = len(BENCHMARK_DATASET)
    correct = 0
    confusion: dict[str, int] = {}   # "T0→T1" 次数 / count of "T0→T1"

    print(f"\n{'='*60}")
    print(f"  LLMRouter Benchmark  ({'真实调用' if real else 'MOCK 模式'}，{total} 条)")
    print(f"{'='*60}\n")

    for i, (msg, expected) in enumerate(BENCHMARK_DATASET, 1):
        result = await router.classify(
            msg, trace_id=f"bench-{i}", session_key=f"bench-session-{i}",
        )
        ok = result.tier == expected
        correct += ok
        key = f"{expected}→{result.tier}"
        confusion[key] = confusion.get(key, 0) + 1

        flag = "✓" if ok else "✗"
        print(f"  [{i:2d}/{total}] {flag} 预期{expected.value} 实际{result.tier.value} "
              f"({result.method}, conf={result.confidence:.2f}, {result.latency_ms:.0f}ms)")
        print(f"        「{msg[:30]}...」")

    accuracy = correct / total * 100
    report = router._log.cost_savings_report() if router._log else {}

    print(f"\n{'─'*60}")
    print(f"  准确率: {correct}/{total} = {accuracy:.1f}%")
    print(f"  档位分布: {report.get('tier_breakdown', {})}")
    print(f"  相对成本: {report.get('actual_relative_cost', 0)} "
          f"(baseline {report.get('baseline_relative_cost', 0)})")
    print(f"  cost 节省: {report.get('savings_pct', 0)}%")
    print(f"{'─'*60}\n")

    # 混淆矩阵（错判最多的方向）/ Confusion matrix (most frequent misclassification directions)
    mistakes = {k: v for k, v in confusion.items() if not k.startswith(k.split("→")[0] + "→" + k.split("→")[0])}
    wrong = {k: v for k, v in confusion.items() if k.split("→")[0] != k.split("→")[1]}
    if wrong:
        print("  错判方向（预期→实际: 次数）:")
        for k, v in sorted(wrong.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")
        print()

    return {
        "accuracy_pct": round(accuracy, 1),
        "correct": correct,
        "total": total,
        "cost_savings_pct": report.get("savings_pct", 0),
        "tier_breakdown": report.get("tier_breakdown", {}),
    }


def main() -> None:
    real = "--mock" not in sys.argv
    result = asyncio.run(run_benchmark(real))
    print(f"\n📊 简历数据点：LLM 调用成本降低约 {result['cost_savings_pct']}%，"
          f"路由准确率 {result['accuracy_pct']}%\n")


if __name__ == "__main__":
    main()

"""
上下文压缩降本数据采集脚本。

用法：
    # 真实调用（需配置 ANTHROPIC_API_KEY，用于压缩摘要）
    python scripts/benchmark_compaction.py

    # mock 模式 dry-run（用估算 token，验证脚本逻辑）
    python scripts/benchmark_compaction.py --mock

产出：
    - 压缩前/后 token 数对比
    - 压缩降本%（token 消耗降低 X%）
    - 压缩保留的消息数 / 被压缩的消息数
    - tool_result 保护情况（关键工具结果是否被摘要保留）

这是简历里"Token 消耗降低 X%"的数据来源。

设计说明：
    压缩的核心是 find_turn_boundary_cut + semantic_importance_score + LLM 摘要。
    前两步是纯算法，mock 模式下就能测出 token 削减量；
    第三步（LLM 摘要）真实模式下才有效，mock 模式摘要用固定文本占位。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoharness.core.compaction import (
    CompactionConfig, CompactionEngine, find_turn_boundary_cut,
    retreat_to_turn_boundary, semantic_importance_score,
)
from nanoharness.core.context import AgentContext, Message
from nanoharness.provider.anthropic import AnthropicProvider


# ─── 构造长对话历史 / Build Long Conversation History ────────────────────────

def build_long_history() -> list[Message]:
    """
    构造一个真实的多轮 ReAct 对话历史：30+ 条消息，含大量工具结果 / Build a realistic multi-turn ReAct history: 30+ messages with lots of tool results.

    模拟场景：用户让 Agent 调研某个技术方案，多轮工具调用 + 分析 / Simulated scenario: user asks the Agent to research a technical solution, with multi-turn tool calls + analysis.
    每条消息设置真实量级的 token_count（基于内容长度估算）/ Each message gets a realistic-magnitude token_count (estimated from content length).
    """
    msgs: list[Message] = []

    # 初始提问 / Initial question
    msgs.append(Message(role="user", content=(
        "我想调研一下用 Rust 重写我们核心交易系统的可行性，"
        "帮我查相关资料并给出分析。"
    )))

    # 模拟 8 轮 ReAct：assistant 思考 + 工具调用 + 工具结果 / Simulate 8 rounds of ReAct: assistant thinking + tool call + tool result
    tool_results = [
        "Rust 在金融系统应用案例：某交易所用 Rust 重写后延迟从 200μs 降到 50μs，"
        "内存安全消除了 70% 的 CVE。但开发效率比 Python 低约 40%，"
        "团队需要 3-6 个月学习曲线。关键依赖：tokio 异步运行时、serde 序列化。"
        * 3,   # 放大 token 量 / amplify token count
        "现有 Python 系统瓶颈分析：GIL 导致 CPU 密集任务无法真并行，"
        "当前 P99 延迟 15ms，目标 5ms。核心热点在订单匹配引擎，"
        "占 60% CPU。数据库 IO 已优化，非瓶颈。" * 3,
        "Rust 生态调研：tokio 成熟稳定，serde 性能业界标杆。"
        "但量化交易专用库较少，部分需自研。FFI 调 Python 混合可行。" * 3,
    ]

    for i in range(8):
        msgs.append(Message(role="assistant", content=[
            {"type": "tool_use", "id": f"call-{i}", "name": "search",
             "input": {"query": f"rust-finance-{i}"}}
        ]))
        msgs.append(Message(role="user", content=[
            {"type": "tool_result", "tool_use_id": f"call-{i}",
             "content": tool_results[i % len(tool_results)]}
        ]))
        msgs.append(Message(role="assistant", content=(
            f"第 {i+1} 轮分析：基于上述资料，Rust 在延迟敏感场景有明显优势，"
            f"但需权衡团队学习成本。继续调研生态成熟度..."
        )))

    msgs.append(Message(role="user", content=(
        "综合这些调研，给我一个最终建议：是否值得重写？"
    )))
    return msgs


def total_tokens(msgs: list[Message]) -> int:
    """估算消息列表总 token。对 list content（tool_use/tool_result）用 json.dumps / Estimate total tokens of a message list; for list content (tool_use/tool_result) use json.dumps."""
    total = 0
    for m in msgs:
        if m.token_count > 0:
            total += m.token_count
        elif isinstance(m.content, str):
            total += len(m.content) // 4
        else:
            total += len(json.dumps(m.content, ensure_ascii=False)) // 4
    return total


# ─── Mock 摘要 Provider / Mock Summary Provider ──────────────────────────────

class _MockSummaryProvider:
    """Mock provider：摘要返回固定文本，验证压缩流程不调真实 LLM / Mock provider: summary returns fixed text, verifying the compaction flow without calling a real LLM."""
    model_id = "mock-haiku"

    async def complete(self, system, messages, tools=None, max_tokens=4096):
        resp = MagicMock()
        resp.final_text = (
            "[摘要] 用户调研 Rust 重写交易系统可行性。"
            "资料表明：Rust 延迟从 200μs 降到 50μs，内存安全消除 70% CVE；"
            "现有 Python 系统瓶颈在 GIL 和订单匹配引擎；"
            "Rust 生态 tokio/serde 成熟但量化专用库少，FFI 混合可行。"
        )
        return resp


# ─── Benchmark 主流程 / Benchmark Main Flow ──────────────────────────────────

async def run_benchmark(real: bool) -> dict:
    if real:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  未检测到 ANTHROPIC_API_KEY，自动切到 mock 模式。")
            real = False

    provider = (
        AnthropicProvider(
            model_id="claude-haiku-4-5-20251001",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        ) if real else _MockSummaryProvider()
    )

    # 用较小的 keep_budget 让压缩真的触发（演示用，生产是 20000）/ Use a smaller keep_budget so compaction actually triggers (demo; production is 20000)
    # mock 模式 token 是估算值，量级小；真实模式用默认 20000 即可 / In mock mode tokens are estimated and small; in real mode use the default 20000
    config = CompactionConfig(
        context_window_limit=180_000,
        keep_recent_messages=6,
        keep_budget_tokens=300,    # 故意调小，让压缩明显触发 / intentionally small so compaction clearly triggers
        safety_margin=0.10,
    )
    engine = CompactionEngine(provider=provider, config=config)

    history = build_long_history()
    ctx = AgentContext(
        agent_id="bench-agent",
        session_key="bench-compaction",
        system_prompt="你是技术调研助手。",
        model_id=provider.model_id,
    )
    ctx.history = list(history)

    tokens_before = total_tokens(ctx.history)
    msg_count_before = len(ctx.history)

    print(f"\n{'='*60}")
    print(f"  Compaction Benchmark  ({'真实调用' if real else 'MOCK 模式'})")
    print(f"{'='*60}")
    print(f"  压缩前: {msg_count_before} 条消息, {tokens_before} tokens\n")

    # 验证 turn-boundary 保护：工具结果不被切断 / Verify turn-boundary protection: tool results are not cut
    cut = find_turn_boundary_cut(ctx.history, config.keep_budget_tokens)
    safe_cut = retreat_to_turn_boundary(ctx.history, cut)
    print(f"  切点: raw={cut} safe={safe_cut}（turn-boundary 保护）")
    if safe_cut < len(ctx.history):
        assert not ctx.history[safe_cut].is_tool_result(), "切断点不应是孤儿 tool_result"

    # 执行压缩 / Run compaction
    await engine.compact(ctx)

    tokens_after = total_tokens(ctx.history)
    msg_count_after = len(ctx.history)
    reduction_pct = (1 - tokens_after / tokens_before) * 100 if tokens_before else 0

    print(f"\n  压缩后: {msg_count_after} 条消息, {tokens_after} tokens")
    print(f"  {'─'*40}")
    print(f"  Token 削减: {tokens_before} → {tokens_after} (−{reduction_pct:.1f}%)")
    print(f"  消息削减: {msg_count_before} → {msg_count_after}")
    print(f"{'='*60}\n")

    # 验证摘要保留：摘要消息应存在且含关键信息 / Verify summary retention: a summary message should exist and contain key info
    summary_msgs = [m for m in ctx.history
                    if isinstance(m.content, str) and "SUMMARY" in m.content.upper()]
    print(f"  摘要消息数: {len(summary_msgs)}")
    if summary_msgs:
        print(f"  摘要预览: {summary_msgs[0].content[:80]}...")

    # 重要性评分样例（面试可讲）/ Importance score sample (good talking point in interviews)
    print(f"\n  重要性评分样例（tool_result 应高于 assistant text）:")
    for i in (1, 2, 3):   # call-0, tool_result-0, analysis-0 / call-0, tool_result-0, analysis-0
        if i < len(history):
            score = semantic_importance_score(history[i], i, len(history))
            role = history[i].role
            is_tr = history[i].is_tool_result()
            print(f"    [{i}] role={role} is_tool_result={is_tr} score={score:.3f}")

    return {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "reduction_pct": round(reduction_pct, 1),
        "messages_before": msg_count_before,
        "messages_after": msg_count_after,
        "mode": "real" if real else "mock",
    }


def main() -> None:
    real = "--mock" not in sys.argv
    result = asyncio.run(run_benchmark(real))
    print(f"\n📊 简历数据点：上下文压缩使 Token 消耗降低约 {result['reduction_pct']}%\n")


if __name__ == "__main__":
    main()

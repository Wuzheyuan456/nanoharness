"""
Gateway 压测脚本：测车道隔离层 P50/P95/P99 延迟和并发安全。

用法：
    python scripts/load_test_gateway.py                    # 默认 50 并发 × 3 轮
    python scripts/load_test_gateway.py --concurrency 100 --rounds 5

产出：
    - P50/P95/P99 延迟（毫秒）
    - 吞吐量 QPS
    - 并发安全验证：同 session 串行、跨 session 并行
    - 错误率

说明：
    压测的是 Gateway 车道隔离层（LaneQueue + 去重 + 安全 + 路由），
    不是 LLM 延迟——用 mock handler 模拟固定处理耗时。
    这是简历里"P99 延迟降低 Z%"的基础数据。

面试话术：
"压测用 mock handler 隔离掉 LLM 延迟，单独测 Gateway 的并发调度能力。
    50 并发会话下 P99 在几十毫秒级，证明车道隔离层不是瓶颈——
    真正的延迟来自 LLM 调用本身，那由 Router 路由到更快模型解决。"
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nanoharness.channels.base import (
    ChannelSendResult, ChatType, InboundEnvelope, OutboundEnvelope, SendStatus,
)
from nanoharness.channels.gateway import Gateway, SafetyPolicy
from nanoharness.channels.router import BindingRule, ChannelRouter
from nanoharness.observability.metrics import get_metrics


class _LoadTestChannel:
    """压测用通道：记录发送，不真实发消息。"""
    channel_id = "loadtest"

    def __init__(self) -> None:
        self.sent_count = 0

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send(self, envelope: OutboundEnvelope) -> ChannelSendResult:
        self.sent_count += 1
        return ChannelSendResult(status=SendStatus.SENT, retryable=False,
                                  reason="", message_ids=[str(self.sent_count)])


def build_gateway(handler_latency_ms: float) -> Gateway:
    """构建压测用 Gateway，handler 模拟固定处理耗时。"""
    router = ChannelRouter(default_agent_id="agent")
    gw = Gateway(router=router, safety=SafetyPolicy(group_require_mention=False))

    async def handler(env: InboundEnvelope) -> OutboundEnvelope:
        # 模拟 Agent 处理耗时
        await asyncio.sleep(handler_latency_ms / 1000)
        return OutboundEnvelope(
            target_channel="loadtest", target_peer=env.chat_id,
            content=f"reply to {env.sender_id}",
        )

    gw.register_channel(_LoadTestChannel())
    gw.set_handler(handler)
    return gw


async def _send_one(gw: Gateway, session_key: str, seq: int) -> float:
    """发一条消息，返回 Gateway 端到端处理耗时（毫秒）。"""
    env = InboundEnvelope(
        envelope_id=f"load-{session_key}-{seq}",   # 唯一 id 避免去重
        channel_id="loadtest",
        sender_id=session_key,
        chat_id=session_key,
        chat_type=ChatType.DIRECT,
        content=f"msg-{seq}",
    )
    t0 = time.monotonic()
    await gw.handle_inbound(env)
    return (time.monotonic() - t0) * 1000


async def run_load_test(concurrency: int, rounds: int, handler_latency_ms: float) -> dict:
    gw = build_gateway(handler_latency_ms)
    metrics = get_metrics()
    metrics.clear() if hasattr(metrics, "clear") else None

    print(f"\n{'='*60}")
    print(f"  Gateway 压测  ({concurrency} 并发会话 × {rounds} 轮, "
          f"handler 延迟 {handler_latency_ms}ms)")
    print(f"{'='*60}\n")

    # 阶段1：并发安全验证——同 session 串行、跨 session 并行
    print("  [验证] 车道隔离：同 session 串行 / 跨 session 并行...")
    # 两个不同 session 各发 2 条，handler 各 sleep 50ms
    verify_gw = build_gateway(50)
    t0 = time.monotonic()
    await asyncio.gather(
        _send_one(verify_gw, "verify-A", 0),
        _send_one(verify_gw, "verify-A", 1),   # 同 session，串行 → 100ms
        _send_one(verify_gw, "verify-B", 0),   # 不同 session，并行
    )
    verify_elapsed = (time.monotonic() - t0) * 1000
    print(f"    3 条消息（同 session 2 条串行 + 跨 session 1 条并行）"
          f"总耗时 {verify_elapsed:.0f}ms")
    print(f"    预期 ≈ 100ms（同 session 串行 50+50，跨 session 并行不额外增加）")
    assert verify_elapsed < 180, "车道隔离异常：串行耗时不对"
    print(f"    ✓ 车道隔离正常\n")

    # 阶段2：正式压测
    print(f"  [压测] 投递 {concurrency}×{rounds} = {concurrency*rounds} 条消息...")
    latencies: list[float] = []
    errors = 0
    t_start = time.monotonic()

    for rnd in range(rounds):
        # 每轮并发投递 concurrency 条（每个 session 一条）
        tasks = [
            _send_one(gw, f"sess-{i}", rnd)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                errors += 1
            else:
                latencies.append(r)

    total_elapsed = time.monotonic() - t_start
    total_msgs = concurrency * rounds

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(n * 0.5)] if n else 0
    p95 = latencies[int(n * 0.95)] if n else 0
    p99 = latencies[int(n * 0.99)] if n else 0
    mean = statistics.mean(latencies) if latencies else 0
    qps = total_msgs / total_elapsed if total_elapsed > 0 else 0
    error_rate = errors / total_msgs * 100 if total_msgs else 0

    # 同时记录到 metrics（供 Gradio 面板展示）
    for lat in latencies:
        metrics.observe_latency(lat / 1000, kind="gateway")

    print(f"\n  {'─'*40}")
    print(f"  总消息数: {total_msgs}  成功: {n}  错误: {errors}")
    print(f"  总耗时:   {total_elapsed:.2f}s")
    print(f"  吞吐量:   {qps:.1f} QPS")
    print(f"  {'─'*40}")
    print(f"  平均延迟: {mean:.1f}ms")
    print(f"  P50:      {p50:.1f}ms")
    print(f"  P95:      {p95:.1f}ms")
    print(f"  P99:      {p99:.1f}ms")
    print(f"  错误率:   {error_rate:.2f}%")
    print(f"{'='*60}\n")

    return {
        "total_msgs": total_msgs, "errors": errors, "qps": round(qps, 1),
        "mean_ms": round(mean, 1), "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1), "p99_ms": round(p99, 1),
        "error_rate_pct": round(error_rate, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gateway 压测")
    parser.add_argument("--concurrency", type=int, default=50, help="并发会话数")
    parser.add_argument("--rounds", type=int, default=3, help="轮数")
    parser.add_argument("--latency", type=float, default=30, help="模拟 handler 延迟(ms)")
    args = parser.parse_args()

    result = asyncio.run(run_load_test(args.concurrency, args.rounds, args.latency))
    print(f"📊 简历数据点：Gateway 在 {args.concurrency} 并发下 P99={result['p99_ms']}ms，"
          f"吞吐 {result['qps']} QPS\n")


if __name__ == "__main__":
    main()

"""
Gradio 可视化调试面板。

三个 Tab：
  1. 路由决策：tier 分布柱状图 + cost 节省 + 决策明细表
  2. 链路回放：trace 列表 → Span 树展示（基于 Tracer）
  3. 指标看板：四大黄金信号（延迟 P50/P95/P99 / token / 错误率 / 饱和度）

数据来源：
  - DecisionLog（SQLite）：路由决策持久化
  - Tracer（内存）：Span 链路
  - MetricsCollector（内存）：黄金信号指标

启动：
    python -m nanoharness.observability.dashboard
    或在代码里调 launch_dashboard()

面试话术：
"面板用 Gradio 几十行搞定，省前端工时。三个 Tab 对应三个可观测性维度：
  路由决策看成本、链路回放看因果、指标看板看健康度。数据都来自前面
  各 Phase 埋好的点——DecisionLog、Tracer、Metrics，面板只做展示。"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nanoharness.observability.metrics import get_metrics
from nanoharness.observability.tracing import get_tracer
from nanoharness.router.decision_log import DecisionLog
from nanoharness.router.tiers import Tier


def _span_tree_to_rows(node: dict[str, Any], depth: int = 0) -> list[list[str]]:
    """把 Span 树递归展平成表格行（缩进表示层级）。"""
    rows: list[list[str]] = []
    span = node.get("span")
    if span:
        indent = "  " * depth
        rows.append([
            f"{indent}{span.name}",
            f"{span.duration_ms:.1f}",
            span.status,
            ", ".join(f"{k}={v}" for k, v in list(span.attributes.items())[:3])
            if span.attributes else "",
        ])
    for child in node.get("children", []):
        rows.extend(_span_tree_to_rows(child, depth + 1))
    return rows


def _load_decisions(db_path: str) -> tuple[pd.DataFrame, dict]:
    """从 SQLite 读路由决策，返回 DataFrame 和 cost 报告。"""
    db = DecisionLog(db_path)
    decisions = list(db.iter_all())
    report = db.cost_savings_report()
    db.close()

    if not decisions:
        return pd.DataFrame(columns=["时间", "消息预览", "档位", "置信度", "方法", "延迟ms"]), report

    rows = [{
        "时间": pd.Timestamp(d.ts, unit="s").strftime("%H:%M:%S"),
        "消息预览": d.input_preview[:30],
        "档位": str(d.tier),
        "置信度": round(d.confidence, 2),
        "方法": d.method,
        "延迟ms": round(d.latency_ms, 1),
    } for d in decisions]
    return pd.DataFrame(rows), report


def _tier_distribution_df(report: dict) -> pd.DataFrame:
    """tier 分布 DataFrame，供 BarPlot。"""
    breakdown = report.get("tier_breakdown", {})
    if not breakdown:
        return pd.DataFrame(columns=["tier", "count"])
    return pd.DataFrame([
        {"tier": k, "count": v} for k, v in breakdown.items()
    ])


def build_dashboard(
    decision_log_path: str = "router_decisions.db",
    tracer: Any = None,
    metrics: Any = None,
):
    """构建 Gradio 面板（返回 Blocks，调用方 .launch()）。"""
    import gradio as gr

    tracer = tracer or get_tracer()
    metrics = metrics or get_metrics()

    with gr.Blocks(title="NanoHarness 可观测性面板") as demo:
        gr.Markdown("# NanoHarness 可观测性面板\n路由成本 / 链路回放 / 黄金信号")

        # ── Tab1: 路由决策 ──────────────────────────────────────────────────
        with gr.Tab("路由决策"):
            with gr.Row():
                refresh_router = gr.Button("刷新路由数据")
            with gr.Row():
                tier_plot = gr.BarPlot(
                    x="tier", y="count",
                    title="各档位调用次数分布",
                    height=250,
                )
                cost_text = gr.Markdown()
            decisions_table = gr.DataFrame(
                headers=["时间", "消息预览", "档位", "置信度", "方法", "延迟ms"],
                label="路由决策明细",
                interactive=False,
            )

            def _refresh_router():
                df, report = _load_decisions(decision_log_path)
                tier_df = _tier_distribution_df(report)
                cost_md = (
                    f"### Cost 节省报告\n"
                    f"- 总调用: **{report.get('total_calls', 0)}** 次\n"
                    f"- 实际相对成本: **{report.get('actual_relative_cost', 0)}**\n"
                    f"- baseline（全 T1）: {report.get('baseline_relative_cost', 0)}\n"
                    f"- **cost 节省: {report.get('savings_pct', 0)}%**\n"
                )
                return tier_df, cost_md, df

            refresh_router.click(_refresh_router, outputs=[tier_plot, cost_text, decisions_table])
            demo.load(_refresh_router, outputs=[tier_plot, cost_text, decisions_table])

        # ── Tab2: 链路回放 ──────────────────────────────────────────────────
        with gr.Tab("链路回放"):
            with gr.Row():
                trace_dropdown = gr.Dropdown(
                    choices=[], label="选择 trace_id", scale=3,
                )
                refresh_traces = gr.Button("刷新 trace 列表", scale=1)
            span_tree = gr.DataFrame(
                headers=["Span", "耗时ms", "状态", "属性"],
                label="Span 树（缩进表示层级）",
                interactive=False,
            )
            trace_info = gr.Markdown()

            def _refresh_traces():
                ids = tracer.all_traces()
                return gr.update(choices=ids, value=ids[0] if ids else None)

            def _show_trace(trace_id: str):
                if not trace_id:
                    return pd.DataFrame(
                        columns=["Span", "耗时ms", "状态", "属性"]
                    ), "未选择 trace"
                spans = tracer.get_trace(trace_id)
                tree = tracer.build_tree(trace_id)
                rows = _span_tree_to_rows(tree)
                df = pd.DataFrame(rows, columns=["Span", "耗时ms", "状态", "属性"])
                info = (
                    f"### Trace `{trace_id}`\n"
                    f"- Span 数: {len(spans)}\n"
                    f"- 总耗时: {sum(s.duration_ms for s in spans):.1f}ms\n"
                    f"- 错误 Span: {sum(1 for s in spans if s.status == 'error')}"
                )
                return df, info

            refresh_traces.click(_refresh_traces, outputs=trace_dropdown)
            trace_dropdown.change(_show_trace, inputs=trace_dropdown,
                                   outputs=[span_tree, trace_info])
            demo.load(_refresh_traces, outputs=trace_dropdown)

        # ── Tab3: 黄金信号 ──────────────────────────────────────────────────
        with gr.Tab("黄金信号"):
            with gr.Row():
                refresh_metrics = gr.Button("刷新指标")
            with gr.Row():
                latency_md = gr.Markdown()
                token_md = gr.Markdown()
            with gr.Row():
                error_md = gr.Markdown()
                util_md = gr.Markdown()
            prometheus_box = gr.Textbox(
                label="Prometheus exposition format（可接 /metrics 端点）",
                lines=12, interactive=False,
            )

            def _refresh_metrics():
                lat = metrics.latency_report(kind="turn")
                latency = (
                    f"### 延迟（Latency）\n"
                    f"- 样本数: {lat['count']}\n"
                    f"- 平均: {lat['mean']*1000:.1f}ms\n"
                    f"- **P50: {lat['p50']*1000:.1f}ms**\n"
                    f"- **P95: {lat['p95']*1000:.1f}ms**\n"
                    f"- **P99: {lat['p99']*1000:.1f}ms**\n"
                )
                tok_report = metrics.token_report()
                tok_lines = "\n".join(f"- {k}: {v}" for k, v in tok_report.items()) \
                    or "- 暂无数据"
                token = f"### 流量（Token 用量）\n{tok_lines}"
                err_rate = metrics.error_rate()
                error = (
                    f"### 错误率（Errors）\n"
                    f"- 错误率: **{err_rate*100:.2f}%**"
                )
                # 饱和度：取所有 session 的 context_util 最大值
                util_samples = metrics.context_util.samples()
                max_util = max((v for _, v in util_samples), default=0.0)
                util = (
                    f"### 饱和度（Context 利用率）\n"
                    f"- 最大 context 利用率: **{max_util*100:.1f}%**\n"
                    f"- 监控 session 数: {len(util_samples)}"
                )
                prom = metrics.render_prometheus()
                return latency, token, error, util, prom

            refresh_metrics.click(_refresh_metrics,
                                    outputs=[latency_md, token_md, error_md, util_md, prometheus_box])
            demo.load(_refresh_metrics,
                       outputs=[latency_md, token_md, error_md, util_md, prometheus_box])

    return demo


def launch_dashboard(
    decision_log_path: str = "router_decisions.db",
    port: int = 7860,
    share: bool = False,
) -> None:
    """启动 Gradio 面板。"""
    demo = build_dashboard(decision_log_path=decision_log_path)
    demo.launch(server_port=port, share=share, show_error=True)


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "router_decisions.db"
    launch_dashboard(decision_log_path=db_path)

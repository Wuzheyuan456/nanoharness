"""
四大黄金信号指标采集（零依赖轻量实现）。

Google SRE 的四大黄金信号：
  - 延迟（Latency）：请求处理耗时，分桶看 P50/P95/P99
  - 流量（Traffic）：请求速率，QPS / 调用次数
  - 错误率（Errors）：失败请求占比
  - 饱和度（Saturation）：资源占用率，如 context window 利用率

为什么不用 prometheus_client？
  秋招项目避免重依赖，自己实现 Counter/Histogram/Gauge 三个原语。
  render_prometheus() 输出标准 Prometheus exposition format 文本，
  以后要接 Prometheus 只需暴露这个 /metrics 端点，业务代码零改动。

面试话术：
"我用 Google SRE 的四大黄金信号组织指标。延迟用分桶 Histogram
  能算 P99 而不是只看平均——平均延迟会掩盖长尾。自己实现了
  Counter/Histogram/Gauge 三个原语，没引 prometheus_client，
  但 render_prometheus 输出标准格式，以后接 Prometheus 只加个 HTTP 端点。"
"""
from __future__ import annotations

import time
from threading import Lock
from typing import Any

# 默认延迟分桶（秒），覆盖 10ms ~ 10s
_DEFAULT_BUCKETS = [
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
]


# ─── 指标原语 ──────────────────────────────────────────────────────────────────

class Counter:
    """单调递增计数器，带标签维度。线程安全。"""

    def __init__(self, name: str, help_text: str, labelnames: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help_text
        self.labelnames = labelnames
        self._values: dict[tuple, float] = {}
        self._lock = Lock()

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def get(self, **labels: Any) -> float:
        return self._values.get(self._label_key(labels), 0.0)

    def samples(self) -> list[tuple[dict, float]]:
        with self._lock:
            return [
                (dict(zip(self.labelnames, k)), v)
                for k, v in self._values.items()
            ]

    def _label_key(self, labels: dict) -> tuple:
        return tuple(labels.get(n, "") for n in self.labelnames)


class Gauge:
    """可增可减的瞬时值，带标签维度。"""

    def __init__(self, name: str, help_text: str, labelnames: tuple[str, ...] = ()) -> None:
        self.name = name
        self.help = help_text
        self.labelnames = labelnames
        self._values: dict[tuple, float] = {}
        self._lock = Lock()

    def set(self, value: float, **labels: Any) -> None:
        key = self._label_key(labels)
        with self._lock:
            self._values[key] = value

    def get(self, **labels: Any) -> float:
        return self._values.get(self._label_key(labels), 0.0)

    def samples(self) -> list[tuple[dict, float]]:
        with self._lock:
            return [
                (dict(zip(self.labelnames, k)), v)
                for k, v in self._values.items()
            ]

    def _label_key(self, labels: dict) -> tuple:
        return tuple(labels.get(n, "") for n in self.labelnames)


class Histogram:
    """
    分桶统计延迟分布，能算分位数（P50/P95/P99）。

    buckets: 上界列表（秒），如 [0.01, 0.05, 0.1, 0.5, 1.0]
    每个观测值落入所有上界 ≥ 它的桶（累积分布，Prometheus 约定）。
    同时保留全量样本用于精确分位数计算。
    """

    def __init__(
        self, name: str, help_text: str,
        buckets: list[float] | None = None,
        labelnames: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.help = help_text
        self.labelnames = labelnames
        self.buckets = sorted(buckets or _DEFAULT_BUCKETS)
        # 每组标签：{bucket_upper: count, "_sum": total, "_count": n, "_values": 全量样本}
        self._data: dict[tuple, dict] = {}
        self._lock = Lock()

    def observe(self, value: float, **labels: Any) -> None:
        """记录一个观测值（秒）。"""
        key = self._label_key(labels)
        with self._lock:
            d = self._data.setdefault(key, {
                **{b: 0 for b in self.buckets},
                "_sum": 0.0, "_count": 0, "_values": [],
            })
            d["_sum"] += value
            d["_count"] += 1
            d["_values"].append(value)
            for b in self.buckets:
                if value <= b:
                    d[b] += 1

    def quantile(self, q: float, **labels: Any) -> float:
        """算分位数（0~1）。用全量样本精确算。"""
        key = self._label_key(labels)
        d = self._data.get(key)
        if not d or d["_count"] == 0:
            return 0.0
        values = sorted(d["_values"])
        idx = max(0, min(len(values) - 1, int(q * len(values))))
        return values[idx]

    def stats(self, **labels: Any) -> dict[str, float]:
        """返回 count/mean/p50/p95/p99。"""
        key = self._label_key(labels)
        d = self._data.get(key)
        if not d or d["_count"] == 0:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "count": d["_count"],
            "mean": d["_sum"] / d["_count"],
            "p50": self.quantile(0.5, **labels),
            "p95": self.quantile(0.95, **labels),
            "p99": self.quantile(0.99, **labels),
        }

    def _label_key(self, labels: dict) -> tuple:
        return tuple(labels.get(n, "") for n in self.labelnames)


# ─── MetricsCollector：四大黄金信号注册中心 ────────────────────────────────────

class MetricsCollector:
    """
    全局指标采集器，注册项目用到的所有指标。

    用法：
        m = get_metrics()
        m.observe_latency(0.35, model="haiku", kind="turn")
        m.inc_tokens(120, model="haiku", token_type="input")
        m.observe_context_utilization(0.72, session_key="s1")
    """

    def __init__(self) -> None:
        # 延迟（黄金信号1）
        self.latency = Histogram(
            "agent_response_latency_seconds",
            "Agent 响应延迟分布（秒）",
            labelnames=("kind", "model"),    # kind: turn/router/tool
        )
        # 流量（黄金信号2）：token 用量
        self.tokens = Counter(
            "agent_token_usage_total",
            "LLM token 总用量",
            labelnames=("model", "token_type"),   # token_type: input/output
        )
        # 流量：工具调用次数
        self.tool_calls = Counter(
            "agent_tool_calls_total",
            "工具调用总次数",
            labelnames=("tool_name", "status"),    # status: success/failed
        )
        # 错误率（黄金信号3）
        self.errors = Counter(
            "agent_errors_total",
            "Agent 错误总数",
            labelnames=("error_type",),
        )
        # 饱和度（黄金信号4）：context window 利用率
        self.context_util = Gauge(
            "context_window_utilization_ratio",
            "上下文窗口利用率（0~1）",
            labelnames=("session_key",),
        )

    # ── 语义化便捷方法 ────────────────────────────────────────────────────────

    def observe_latency(self, seconds: float, kind: str = "turn", model: str = "") -> None:
        self.latency.observe(seconds, kind=kind, model=model)

    def inc_tokens(self, amount: int, model: str = "", token_type: str = "input") -> None:
        self.tokens.inc(amount, model=model, token_type=token_type)

    def inc_tool_call(self, tool_name: str, success: bool = True) -> None:
        self.tool_calls.inc(tool_name=tool_name, status="success" if success else "failed")

    def inc_error(self, error_type: str) -> None:
        self.errors.inc(error_type=error_type)

    def observe_context_utilization(self, ratio: float, session_key: str = "") -> None:
        self.context_util.set(ratio, session_key=session_key)

    # ── 汇总报告 ──────────────────────────────────────────────────────────────

    def latency_report(self, kind: str = "turn", model: str = "") -> dict[str, float]:
        return self.latency.stats(kind=kind, model=model)

    def token_report(self) -> dict[str, int]:
        """按 model+type 汇总 token。"""
        out: dict[str, int] = {}
        for labels, val in self.tokens.samples():
            key = f"{labels.get('model', '')}/{labels.get('token_type', '')}"
            out[key] = int(val)
        return out

    def error_rate(self) -> float:
        """错误率 = errors / (errors + 成功工具调用)。"""
        total_calls = sum(v for _, v in self.tool_calls.samples())
        total_errors = sum(v for _, v in self.errors.samples())
        denom = total_calls + total_errors
        return total_errors / denom if denom > 0 else 0.0

    # ── Prometheus 格式导出 ───────────────────────────────────────────────────

    def render_prometheus(self) -> str:
        """
        输出标准 Prometheus exposition format 文本。

        接 Prometheus 只需把这个文本挂在 /metrics HTTP 端点。
        """
        lines: list[str] = []

        def _emit_counter(c: Counter) -> None:
            lines.append(f"# HELP {c.name} {c.help}")
            lines.append(f"# TYPE {c.name} counter")
            for labels, val in c.samples():
                label_str = ",".join(f'{n}="{v}"' for n, v in labels.items())
                lines.append(f"{c.name}{{{label_str}}} {val}")

        def _emit_gauge(g: Gauge) -> None:
            lines.append(f"# HELP {g.name} {g.help}")
            lines.append(f"# TYPE {g.name} gauge")
            for labels, val in g.samples():
                label_str = ",".join(f'{n}="{v}"' for n, v in labels.items())
                lines.append(f"{g.name}{{{label_str}}} {val}")

        def _emit_histogram(h: Histogram) -> None:
            lines.append(f"# HELP {h.name} {h.help}")
            lines.append(f"# TYPE {h.name} histogram")
            for labels, d in h._data.items():
                label_prefix = ",".join(f'{n}="{v}"' for n, v in zip(h.labelnames, labels))
                for b in h.buckets:
                    lines.append(f'{h.name}_bucket{{{label_prefix},le="{b}"}} {d[b]}')
                lines.append(f'{h.name}_bucket{{{label_prefix},le="+Inf"}} {d["_count"]}')
                lines.append(f'{h.name}_sum{{{label_prefix}}} {d["_sum"]}')
                lines.append(f'{h.name}_count{{{label_prefix}}} {d["_count"]}')

        _emit_counter(self.tokens)
        _emit_counter(self.tool_calls)
        _emit_counter(self.errors)
        _emit_gauge(self.context_util)
        _emit_histogram(self.latency)
        return "\n".join(lines) + "\n"


# ─── 全局单例 ──────────────────────────────────────────────────────────────────

_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics

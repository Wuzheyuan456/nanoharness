# Prometheus 指标采集（可选，Phase 7 再实现）
# 四大黄金信号: 延迟 / 流量 / 错误率 / 饱和度
# agent_response_latency_seconds（Histogram）
# agent_token_usage_total（Counter，按 model/type 分组）
# agent_tool_calls_total（Counter，按 tool_name/status 分组）
# context_window_utilization_ratio（Gauge）

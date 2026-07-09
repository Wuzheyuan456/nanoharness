# OpenTelemetry 全链路追踪
# get_tracer(): 初始化 OTel tracer
# trace_turn(): context manager，为每次 turn 创建根 Span
# trace_tool_call(): 为每次工具调用创建子 Span（记录工具名 / 延迟 / token 用量）
# trace_compaction(): 记录压缩触发和压缩后 token 变化

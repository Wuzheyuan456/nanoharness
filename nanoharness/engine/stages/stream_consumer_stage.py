# StreamConsumerStage: 驱动 NanoCore.run_turn()，消费 async generator
# 处理流式 token / 工具调用事件 / in-turn compaction 触发
# 把 AgentEvent 转发给 Hook 和 Channel 出站层

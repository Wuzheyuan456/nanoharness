# Supervisor 模式编排
# Orchestrator: 接收复杂任务 → LLM 拆解 → 从 AgentRegistry 按能力路由 → spawn Worker NanoCore
# spawn(): 用 agent_factory 注入解耦（不直接 import NanoCore）
# 深度限制 + 并发限制双重保护
# SubagentHandle: asyncio.Task + done_callback 状态管理

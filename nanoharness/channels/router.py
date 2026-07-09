# 消息路由引擎
# BindingRule: 声明式规则（channel / chat_type / sender_pattern / group_pattern → agent_id）
# Router.resolve(): 按优先级匹配规则，返回目标 agent_id
# session_key 生成: DM → agent:dm:channel:sender_id / Group → agent:group:channel:group_id

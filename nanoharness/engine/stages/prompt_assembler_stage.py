# PromptAssemblerStage: 组装最终发给 LLM 的 system prompt
# 注入记忆召回结果（memory_search tool result 形式）
# 注入技能列表（L1 元数据层，~100 token/技能）
# 调用 LLM Router 决定本次使用的模型档位

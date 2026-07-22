# PromptAssemblerStage: 组装最终发给 LLM 的 system prompt / PromptAssemblerStage: assemble the final system prompt sent to the LLM
# 注入记忆召回结果（memory_search tool result 形式） / Inject memory recall results (in memory_search tool result form)
# 注入技能列表（L1 元数据层，~100 token/技能） / Inject skill list (L1 metadata layer, ~100 token/skill)
# 调用 LLM Router 决定本次使用的模型档位 / Call LLM Router to decide the model tier for this turn

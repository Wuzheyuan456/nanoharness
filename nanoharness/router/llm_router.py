# LLM-based 难度分类器（不依赖本地 ONNX，用一次 T0 级别的 LLM call 打标签）
# classify(): 输入用户消息 → 返回 (tier, confidence, reason)
# 降级链: LLM 分类 → 超时 → 规则 heuristic → 默认 T1
# ThreadPoolExecutor 隔离（预留，当前 LLM call 本身是 async）
# 路由结果写入 decision_log

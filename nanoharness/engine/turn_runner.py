# TurnRunner: 分阶段编排一次完整的 turn
# 持有 per-session asyncio.Lock + ContextVar 重入检测
# 按顺序调用各 Stage: input → compaction → prompt_assembler → provider → stream_consumer → finalizer
# 负责 Hook 的注册和触发时机

# Lane Queue：车道式会话隔离队列
# 同一 session_key 的消息串行处理（asyncio.Lock per session）
# 不同 session 可并行处理
# ContextVar 重入检测：防止 subagent 用同 session_key 时死锁

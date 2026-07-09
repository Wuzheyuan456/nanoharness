# MemoryManager: 记忆系统对外统一接口
# prefetch(): 每轮 LLM 调用前，判断是否触发召回
# flush(): 会话结束时触发巩固和持久化
# 四层记忆结构: L0 Bootstrap / L1 Session / L2 Retrieval Index / L3 Tool Cache

# 记忆生命周期 Hooks（接入 CompactionHook）
# before_compact(): 扫描即将被压缩的对话，提取核心事实强制写入 MEMORY.md（防止 Failure Mode B）
# sync_turn(): 工具执行后自动捕获关键操作，写入短期记忆日志

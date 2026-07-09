# SQLite + sqlite-vec 存储层
# MemoryStore: insert / search / delete
# 分块策略: 按 token 数近似分块（Tokens * 4 字符）
# 增量更新: 文件级 Hash 比对，未变更文件跳过 embedding 计算
# 去重: chunk_id 覆盖更新

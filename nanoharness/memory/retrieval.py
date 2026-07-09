# 两阶段精准召回
# Stage 1: BM25 关键词检索 + sqlite-vec 向量检索（宽召回 × 10）
# Stage 2: RRF（倒数排名融合）重排 → Top-K 精读
# 召回结果包装成 memory_search tool result 注入 context（不直接塞 system prompt）
# evergreen 文件（MEMORY.md）不做时间衰减

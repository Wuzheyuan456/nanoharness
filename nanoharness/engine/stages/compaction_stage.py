# CompactionStage: preflight 压缩（turn 开始前，读 DB 判断是否需要压缩）
# 与 in-turn compaction 分离：preflight 读 DB，in-turn 操作 agent._history
# 通过 has_attempted_compaction_this_turn() 防止双重压缩

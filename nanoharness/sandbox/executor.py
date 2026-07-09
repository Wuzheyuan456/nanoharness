# 工具沙箱执行器
# subprocess + resource limit（CPU / 内存 / 文件访问白名单）
# 高危命令黑名单（rm -rf / curl / wget 等正则拦截）
# 审计日志：每次执行记录 session_id / tool_name / 入参摘要 / 结果 / 耗时
# Docker 隔离（预留接口，Phase 5 后实现）

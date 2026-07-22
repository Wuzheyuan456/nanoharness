# 工具沙箱执行器 / Tool sandbox executor
# subprocess + resource limit（CPU / 内存 / 文件访问白名单）/ subprocess + resource limit (CPU / memory / file access whitelist)
# 高危命令黑名单（rm -rf / curl / wget 等正则拦截）/ High-risk command blacklist (regex interception of rm -rf / curl / wget etc.)
# 审计日志：每次执行记录 session_id / tool_name / 入参摘要 / 结果 / 耗时 / Audit log: each execution records session_id / tool_name / input summary / result / latency
# Docker 隔离（预留接口，Phase 5 后实现）/ Docker isolation (reserved interface, to be implemented after Phase 5)

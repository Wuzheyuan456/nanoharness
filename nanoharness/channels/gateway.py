# Gateway 控制平面：所有通道消息的统一入口
# 处理流水线: 回声检测 → 去重 → 安全检查 → 路由决策 → LaneQueue 分发
# 插件注册: register_channel(plugin: BaseChannel)
# 安全策略: DM 配对验证 / 白名单 / 黑名单 / 群聊需 @机器人

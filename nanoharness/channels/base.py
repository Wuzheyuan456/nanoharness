# Channel 抽象层（对齐 opensquilla channels/contract.py）
# InboundEnvelope: 统一入站消息格式（envelope_id / channel_id / sender_id / content / chat_type）
# OutboundEnvelope: 统一出站消息格式（target_channel / target_peer / content）
# ChannelSendResult: 结构化发送结果（status / retryable / reason）
# BaseChannel Protocol: start() / stop() / send() / parse_message()
# ChatType: DIRECT / GROUP / CHANNEL / THREAD

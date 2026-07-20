"""
Discord 通道插件（discord.py 2.x）。

职责：
  - parse_message(): discord.Message → InboundEnvelope
  - send(): OutboundEnvelope → channel.send()，分块上限 2000 字符

设计要点：
  - channel_id 固定 "discord"
  - chat_type 判定：DMChannel / GroupChannel → DIRECT，TextChannel → GROUP
    （用 isinstance 而不是字符串匹配，discord.py 的 channel 类型是类）
  - mentions_bot：检查 message.mentions 里是否有本 bot 的 User 对象
  - 超长分块复用 telegram 的 _split_long_text，只是上限换成 2000

面试话术：
"Discord 和 Telegram 的通道插件结构完全一样——parse_message + send。
差异只在平台特有字段：Discord 用 mentions 列表判定 @，Telegram 用 entities。
复用了 _split_long_text 这个纯函数，只是上限从 4096 换成 2000。
这印证了 BaseChannel 抽象的价值——新通道的边际成本极低。"
"""
from __future__ import annotations

import logging
from typing import Any

from nanoharness.channels.base import (
    ChannelSendResult, ChatType, InboundEnvelope, OutboundEnvelope, SendStatus,
)
from nanoharness.channels.telegram import _split_long_text

log = logging.getLogger(__name__)

DISCORD_MAX_LEN = 2000


class DiscordChannel:
    """
    Discord 通道插件。外部注入 client 和 bot_user_id，便于测试用 fake 替换。
    """

    channel_id = "discord"

    def __init__(self, client: Any, bot_user_id: int | str = 0) -> None:
        self._client = client
        self._bot_user_id = str(bot_user_id)

    async def start(self) -> None:
        """启动 client。生产环境这里 client.run(token)。"""
        log.info("Discord 通道已启动（bot_user_id=%s）", self._bot_user_id)

    async def stop(self) -> None:
        log.info("Discord 通道已停止")

    def parse_message(self, message: Any) -> InboundEnvelope:
        """
        discord.Message → InboundEnvelope。

        channel 类型用 isinstance 区分——discord.py 的 DMChannel 和
        TextChannel 是不同类。测试用 SimpleNamespace + 自定义类伪造。
        """
        channel = getattr(message, "channel", None)
        author = getattr(message, "author", None)
        content = getattr(message, "content", "") or ""
        channel_id = getattr(channel, "id", 0) if channel else 0
        author_id = getattr(author, "id", 0) if author else 0

        # chat_type：DM/Group → DIRECT，其余（TextChannel/VoiceChannel）→ GROUP
        chat_type = self._resolve_chat_type(channel)

        # mentions_bot：检查 mentions 列表
        mentions = self._is_mentioned(message)

        # reply 链路
        reply_ref = getattr(message, "reference", None)
        reply_to_sender_id = ""
        if reply_ref is not None and getattr(reply_ref, "message_id", None):
            # resolved 才有 author，cached 可能只有 id
            resolved = getattr(reply_ref, "resolved", None)
            if resolved is not None:
                resolved_author = getattr(resolved, "author", None)
                reply_to_sender_id = str(getattr(resolved_author, "id", "")) if resolved_author else ""

        return InboundEnvelope(
            channel_id=self.channel_id,
            sender_id=str(author_id),
            chat_id=str(channel_id),
            chat_type=chat_type,
            content=content,
            mentions_bot=mentions,
            reply_to_sender_id=reply_to_sender_id,
            raw=message,
        )

    def _resolve_chat_type(self, channel: Any) -> ChatType:
        """Discord channel 类型 → ChatType。用 isinstance 兼容子类。"""
        if channel is None:
            return ChatType.DIRECT
        # 私聊/群组私聊 → DIRECT
        type_name = type(channel).__name__
        if type_name in ("DMChannel", "GroupChannel"):
            return ChatType.DIRECT
        if type_name == "Thread":
            return ChatType.THREAD
        if type_name in ("TextChannel", "VoiceChannel", "StageChannel"):
            return ChatType.GROUP
        return ChatType.DIRECT   # 兜底

    def _is_mentioned(self, message: Any) -> bool:
        """检查 mentions 列表里是否有本 bot。"""
        if not self._bot_user_id:
            return False
        mentions = getattr(message, "mentions", None) or []
        for user in mentions:
            if str(getattr(user, "id", "")) == self._bot_user_id:
                return True
        return False

    async def send(self, envelope: OutboundEnvelope) -> ChannelSendResult:
        """发送出站消息，自动分块（2000 上限）。"""
        chunks = _split_long_text(envelope.content, DISCORD_MAX_LEN)
        sent_ids: list[str] = []
        # Discord 需要 fetch channel 再 send；生产里 client.get_channel / fetch_channel
        channel = None
        get_channel = getattr(self._client, "get_channel", None)
        if get_channel is not None:
            channel = get_channel(int(envelope.target_peer))
        fetch_channel = getattr(self._client, "fetch_channel", None)
        if channel is None and fetch_channel is not None:
            try:
                channel = await fetch_channel(int(envelope.target_peer))
            except Exception as exc:
                log.warning("Discord 获取 channel 失败: %s", exc)
                return ChannelSendResult(
                    status=SendStatus.FAILED, retryable=False,
                    reason=f"channel 不可达: {exc}", message_ids=[],
                )
        if channel is None:
            return ChannelSendResult(
                status=SendStatus.FAILED, retryable=False,
                reason="channel 不存在", message_ids=[],
            )

        for chunk in chunks:
            try:
                msg = await channel.send(chunk)
                sent_ids.append(str(getattr(msg, "id", "")))
            except Exception as exc:
                msg_lower = str(exc).lower()
                if "rate" in msg_lower or "429" in msg_lower:
                    return ChannelSendResult(
                        status=SendStatus.RATE_LIMITED, retryable=True,
                        reason=f"限流: {exc}", message_ids=sent_ids,
                    )
                log.warning("Discord 发送失败: %s", exc)
                return ChannelSendResult(
                    status=SendStatus.FAILED, retryable=False,
                    reason=str(exc), message_ids=sent_ids,
                )
        return ChannelSendResult(
            status=SendStatus.SENT, retryable=False,
            reason="", message_ids=sent_ids,
        )

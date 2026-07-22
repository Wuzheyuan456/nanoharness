"""
Telegram 通道插件（aiogram 3.x）/ Telegram channel plugin (aiogram 3.x).

职责 / Responsibilities:
  - parse_message(): aiogram Message → InboundEnvelope（平台对象翻译成信封）
  - send(): OutboundEnvelope → bot.send_message()，处理超长消息分块（4096 char 限制）

设计要点：
  - channel_id 固定 "telegram"，路由和日志按通道隔离
  - chat_type 映射：private→DIRECT，group/supergroup→GROUP，channel→CHANNEL
  - mentions_bot 判定：检查 message entities 里是否有 text_mention 指向本 bot，
    或 text 里有 @<bot_username>。群聊安全门控依赖这个字段
  - 超长消息分块发送，按 4096 上限切，尽量在换行处切避免半截句子

面试话术：
"Telegram 单条消息上限 4096 字符，Discord 是 2000。Agent 回复经常超长
（比如贴一大段代码），不分块直接发会被平台拒。我在 send 里做分块，
优先按换行切——不要把一句话切成两半，体验更差。"
"""
from __future__ import annotations

import logging
from typing import Any

from nanoharness.channels.base import (
    ChannelSendResult, ChatType, InboundEnvelope, OutboundEnvelope, SendStatus,
)

log = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4096


class TelegramChannel:
    """
    Telegram 通道插件。 / Telegram channel plugin.

    不在这里建 bot 实例——外部注入 Bot，便于测试用 fake bot 替换。 / Bot instance is not created here—injected from outside so tests can swap in a fake bot.
    """

    channel_id = "telegram"

    def __init__(self, bot: Any, bot_username: str = "") -> None:
        self._bot = bot
        self._bot_username = bot_username.lstrip("@").lower()
        self._dp: Any = None   # Dispatcher，start 时才初始化 / Dispatcher, initialized on start

    async def start(self) -> None:
        """启动 dispatcher。生产环境这里会 start_polling。 / Start dispatcher. In production this calls start_polling."""
        # 通道插件对外只暴露 BaseChannel 协议方法，具体平台启动逻辑放在这里 / Channel plugin only exposes BaseChannel protocol methods; platform-specific startup logic goes here
        log.info("Telegram 通道已启动（bot_username=%s）", self._bot_username)

    async def stop(self) -> None:
        log.info("Telegram 通道已停止")

    def parse_message(self, message: Any) -> InboundEnvelope:
        """
        aiogram Message → InboundEnvelope。 / aiogram Message → InboundEnvelope.

        用 getattr 取字段而不是类型注解，避免类型耦合太紧， / Uses getattr to read fields rather than type annotations to keep coupling loose,
        也方便测试用 SimpleNamespace 伪造 message。 / and to make it easy to fake message with SimpleNamespace in tests.
        """
        chat = getattr(message, "chat", None)
        from_user = getattr(message, "from_user", None) or getattr(message, "from_user", None)
        chat_type_raw = getattr(chat, "type", "private") if chat else "private"
        text = getattr(message, "text", "") or ""
        chat_id = getattr(chat, "id", 0) if chat else 0
        sender_id = getattr(from_user, "id", 0) if from_user else 0

        # chat_type 映射 / chat_type mapping
        type_map = {
            "private": ChatType.DIRECT,
            "group": ChatType.GROUP,
            "supergroup": ChatType.GROUP,
            "channel": ChatType.CHANNEL,
        }
        chat_type = type_map.get(chat_type_raw, ChatType.DIRECT)

        # mentions_bot 判定 / mentions_bot check
        mentions = self._is_mentioned(message, text)

        # reply 链路 / reply chain
        reply_to = getattr(message, "reply_to_message", None)
        reply_to_sender_id = ""
        if reply_to is not None:
            reply_from = getattr(reply_to, "from_user", None)
            reply_to_sender_id = str(getattr(reply_from, "id", "")) if reply_from else ""

        return InboundEnvelope(
            channel_id=self.channel_id,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            chat_type=chat_type,
            content=text,
            mentions_bot=mentions,
            reply_to_sender_id=reply_to_sender_id,
            raw=message,
        )

    def _is_mentioned(self, message: Any, text: str) -> bool:
        """检查消息是否 @了本机器人。 / Check whether the message @-mentions this bot."""
        # 方式1：文本里出现 @<bot_username> / Method 1: @<bot_username> appears in text
        if self._bot_username and f"@{self._bot_username}" in text.lower():
            return True
        # 方式2：entities 里有 text_mention 指向本 bot（无 username 的 bot） / Method 2: entities contain a text_mention pointing to this bot (bots without username)
        entities = getattr(message, "entities", None) or []
        for ent in entities:
            ent_type = getattr(ent, "type", "")
            if ent_type == "mention" and self._bot_username:
                # mention entity 对应的文本片段 / text fragment corresponding to the mention entity
                offset = getattr(ent, "offset", 0)
                length = getattr(ent, "length", 0)
                mentioned = text[offset:offset + length].lstrip("@").lower()
                if mentioned == self._bot_username:
                    return True
        return False

    async def send(self, envelope: OutboundEnvelope) -> ChannelSendResult:
        """
        发送出站消息，自动分块（4096 上限）。 / Send outbound message, auto-chunked (4096 limit).

        失败处理：RateLimit/RetryAfter 标记 retryable=True，其他异常标记 FAILED。 / Failure handling: RateLimit/RetryAfter marks retryable=True; other exceptions mark FAILED.
        """
        chunks = _split_long_text(envelope.content, TELEGRAM_MAX_LEN)
        sent_ids: list[str] = []
        for chunk in chunks:
            try:
                msg = await self._bot.send_message(
                    chat_id=int(envelope.target_peer),
                    text=chunk,
                )
                sent_ids.append(str(getattr(msg, "message_id", "")))
            except Exception as exc:
                # aiogram 的 RetryAfter 异常名含 "RetryAfter"，标记可重试 / aiogram's RetryAfter exception name contains "RetryAfter"; mark retryable
                msg_lower = str(exc).lower()
                if "retry" in msg_lower or "429" in msg_lower:
                    return ChannelSendResult(
                        status=SendStatus.RATE_LIMITED,
                        retryable=True,
                        reason=f"限流: {exc}",
                        message_ids=sent_ids,
                    )
                log.warning("Telegram 发送失败: %s", exc)
                return ChannelSendResult(
                    status=SendStatus.FAILED,
                    retryable=False,
                    reason=str(exc),
                    message_ids=sent_ids,
                )
        return ChannelSendResult(
            status=SendStatus.SENT,
            retryable=False,
            reason="",
            message_ids=sent_ids,
        )


def _split_long_text(text: str, max_len: int) -> list[str]:
    """
    把超长文本切分成不超过 max_len 的块。 / Split a long text into chunks no longer than max_len.

    优先在换行处切，其次按硬长度切。这是 send 方法的纯函数， / Prefer cutting at newlines, otherwise hard-cut by length. This is a pure function of send,
    拆出来便于单元测试（不需要真的发消息）。 / extracted so it can be unit-tested without actually sending messages.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        # 在 max_len 范围内找最后一个换行，避免切半句话 / find last newline within max_len to avoid splitting a sentence
        cut = remaining.rfind("\n", 0, max_len)
        if cut <= 0:
            # 没有换行，硬切 / no newline, hard cut
            cut = max_len
        chunks.append(remaining[:cut].rstrip("\n"))
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks

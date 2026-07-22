"""
通道抽象层（对齐 opensquilla channels/contract.py）。

设计目标：把不同 IM 平台（Telegram / Discord / ...）的消息统一成
两套信封结构，上层 Gateway 只处理信封，不感知具体平台。

面试话术：
"InboundEnvelope / OutboundEnvelope 是通道层的'通用语'。
Telegram 收到的是 aiogram.Message，Discord 收到的是 discord.Message，
结构完全不同。通道插件负责把它们翻译成 InboundEnvelope，
Gateway 只认信封——加一个新平台只需要写一个 parse_message 适配器。"
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ChatType(str, Enum):
    """会话类型，决定 session_key 生成策略和群聊安全策略 / Chat type, determines session_key generation strategy and group-chat safety strategy."""
    DIRECT = "direct"     # 私聊：会话隔离按 sender_id / direct message: session isolation by sender_id
    GROUP = "group"       # 群聊：会话隔离按 group_id，且通常需要 @机器人 / group: session isolation by group_id, and usually requires @bot
    CHANNEL = "channel"   # 频道广播（如 Telegram Channel）：只读为主 / channel broadcast (e.g. Telegram Channel): mostly read-only
    THREAD = "thread"     # 论坛帖：按 thread_id 隔离 / forum thread: isolation by thread_id


class SendStatus(str, Enum):
    """发送结果状态，区分可重试与不可重试错误 / Send result status, distinguishing retryable from non-retryable errors."""
    SENT = "sent"             # 发送成功 / sent successfully
    FAILED = "failed"         # 不可重试失败（如被禁言）/ non-retryable failure (e.g. muted)
    RATE_LIMITED = "rate_limited"  # 可重试：限流，建议退避后重发 / retryable: rate limited, recommend retry after backoff


@dataclass(frozen=True)
class InboundEnvelope:
    """入站消息信封：平台无关的统一格式 / Inbound message envelope: platform-agnostic unified format."""
    envelope_id: str = ""             # 信封唯一 id（用于去重），空则自动生成 / unique envelope id (for dedup); auto-generated if empty
    channel_id: str = ""              # 来源通道 id，如 "telegram" / "discord" / source channel id, e.g. "telegram" / "discord"
    sender_id: str = ""               # 发送者 id（用户/机器人）/ sender id (user/bot)
    chat_id: str = ""                 # 会话 id（私聊=用户id，群聊=群id）/ chat id (DM=user id, group=group id)
    chat_type: ChatType = ChatType.DIRECT
    content: str = ""
    mentions_bot: bool = False        # 群聊中是否 @了本机器人（安全门控用）/ whether the bot was @-mentioned in group chat (used for safety gating)
    reply_to_sender_id: str = ""      # 被回复消息的发送者 id（可选）/ sender id of the replied-to message (optional)
    raw: Any = None                   # 原始平台对象（调试用，不参与业务逻辑）/ original platform object (for debugging, not used in business logic)

    def __post_init__(self) -> None:
        if not self.envelope_id:
            # frozen dataclass 用 object.__setattr__ 绕过只读限制 / frozen dataclass uses object.__setattr__ to bypass the read-only restriction
            object.__setattr__(self, "envelope_id", f"env-{uuid.uuid4().hex[:12]}")


@dataclass(frozen=True)
class OutboundEnvelope:
    """出站消息信封 / Outbound message envelope."""
    target_channel: str = ""          # 目标通道 id / target channel id
    target_peer: str = ""            # 目标会话 id（私聊=用户id，群聊=群id）/ target chat id (DM=user id, group=group id)
    content: str = ""
    reply_to_envelope_id: str = ""   # 引用回复哪条入站信封（可选）/ inbound envelope id being replied to (optional)


@dataclass
class ChannelSendResult:
    """通道 send() 的结构化返回，区分可重试与不可重试失败 / Structured return of channel send(), distinguishing retryable from non-retryable failures."""
    status: SendStatus
    retryable: bool = False        # RATE_LIMITED → True，其余 → False / RATE_LIMITED → True, others → False
    reason: str = ""
    message_ids: list[str] = field(default_factory=list)  # 平台返回的消息 id 列表 / list of message ids returned by the platform


@runtime_checkable
class BaseChannel(Protocol):
    """通道插件协议。Gateway 通过这个协议操作任意通道 / Channel plugin protocol. The Gateway operates any channel through this protocol."""
    channel_id: str

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send(self, envelope: OutboundEnvelope) -> ChannelSendResult: ...


@runtime_checkable
class MessageParser(Protocol):
    """可选的 parse_message 协议，通道插件把平台原始消息转成 InboundEnvelope / Optional parse_message protocol; the channel plugin converts the platform's raw message into an InboundEnvelope."""
    def parse_message(self, raw: Any) -> InboundEnvelope: ...

"""
Gateway 控制平面：所有通道消息的统一入口 / Gateway control plane: unified entry for all channel messages.

处理流水线（每条入站消息依次经过）/ Processing pipeline (each inbound message passes through in order):
  1. 通道查找：按 envelope.channel_id 找到注册的通道插件
  2. 去重：envelope_id 去重窗口，防止 webhook 重投导致重复处理
  3. 安全检查：发送者白名单 + 群聊 @机器人 门控
  4. 路由决策：ChannelRouter.resolve → agent_id
  5. 车道分发：LaneQueue 按 session_key 串行执行 handler
  6. 回复发送：handler 返回 OutboundEnvelope → 通道插件 send()

Gateway 不依赖具体 Agent 实现——handler 是注入的 callable，
输入 InboundEnvelope，输出 OutboundEnvelope。这层解耦让 Gateway
可以独立测试（用 fake handler + fake channel），也方便以后换掉 Agent 层。

面试话术：
"Gateway 是经典的责任链：去重→安全→路由→分发，每一步都是
独立可测的策略。handler 是注入的，Gateway 不知道背后是 TurnRunner
还是别的什么——这让通道层和 Agent 层彻底解耦。
群聊默认要求 @机器人，不然群里每句话都会触发 Agent，既吵又烧钱。"
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from nanoharness.channels.base import (
    BaseChannel, ChannelSendResult, InboundEnvelope, OutboundEnvelope,
    SendStatus,
)
from nanoharness.channels.lane_queue import LaneQueue
from nanoharness.channels.router import ChannelRouter, make_session_key

log = logging.getLogger(__name__)

# handler 签名：输入入站信封，返回出站信封（或 None 表示不回复） / handler signature: takes inbound envelope, returns outbound envelope (or None for no reply)
InboundHandler = Callable[[InboundEnvelope], Awaitable[OutboundEnvelope | None]]


@dataclass
class SafetyPolicy:
    """安全策略，可热更新。 / Safety policy, hot-updatable."""
    allowed_senders: set[str] = field(default_factory=set)   # 空=不限制 / empty = no restriction
    blocked_senders: set[str] = field(default_factory=set)
    group_require_mention: bool = True    # 群聊默认要求 @机器人 / groups require @bot by default


@dataclass
class DedupResult:
    """去重检查结果。 / Dedup check result."""
    duplicate: bool
    reason: str = ""


class DedupWindow:
    """
    固定容量的去重窗口（LRU）。 / Fixed-capacity dedup window (LRU).

    用 OrderedDict 实现：超容量时淘汰最久未访问的 envelope_id。 / Implemented with OrderedDict: evicts least-recently-accessed envelope_id when over capacity.
    生产级可换成 Redis SETEX，这里内存版够用且零依赖。 / Production grade can switch to Redis SETEX; the in-memory version is enough here and zero-dependency.
    """

    def __init__(self, capacity: int = 2048) -> None:
        self._capacity = capacity
        self._seen: OrderedDict[str, None] = OrderedDict()

    def check(self, envelope_id: str) -> DedupResult:
        if envelope_id in self._seen:
            # 命中：移到末尾表示最近访问 / hit: move to end to mark recently accessed
            self._seen.move_to_end(envelope_id)
            return DedupResult(duplicate=True, reason=f"envelope_id {envelope_id} 已处理过")
        self._seen[envelope_id] = None
        if len(self._seen) > self._capacity:
            self._seen.popitem(last=False)   # 淘汰最老 / evict oldest
        return DedupResult(duplicate=False)

    def __len__(self) -> int:
        return len(self._seen)


class Gateway:
    """
    多通道网关控制平面。 / Multi-channel gateway control plane.

    用法： / Usage:
        gw = Gateway(router=ChannelRouter(...))
        gw.register_channel(telegram_channel)
        gw.register_channel(discord_channel)
        gw.set_handler(my_agent_handler)
        await gw.start()
        # 通道收到消息后调用 gw.handle_inbound(envelope) / call gw.handle_inbound(envelope) when a channel receives a message
    """

    def __init__(
        self,
        router: ChannelRouter,
        lane_queue: LaneQueue | None = None,
        safety: SafetyPolicy | None = None,
        dedup_capacity: int = 2048,
    ) -> None:
        self._router = router
        self._lane = lane_queue or LaneQueue()
        self._safety = safety or SafetyPolicy()
        self._dedup = DedupWindow(capacity=dedup_capacity)
        self._channels: dict[str, BaseChannel] = {}
        self._handler: InboundHandler | None = None

    # ── 配置 / Configuration ──────────────────────────────────────────────────────────────────

    def register_channel(self, channel: BaseChannel) -> None:
        """注册通道插件，channel_id 必须唯一。 / Register a channel plugin; channel_id must be unique."""
        if channel.channel_id in self._channels:
            raise ValueError(f"通道 {channel.channel_id} 已注册")
        self._channels[channel.channel_id] = channel

    def set_handler(self, handler: InboundHandler) -> None:
        """注入入站消息处理器。 / Inject the inbound message handler."""
        self._handler = handler

    def get_channel(self, channel_id: str) -> BaseChannel | None:
        return self._channels.get(channel_id)

    # ── 生命周期 / Lifecycle ──────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """启动所有已注册通道。 / Start all registered channels."""
        for channel in self._channels.values():
            await channel.start()
        log.info("Gateway 启动完成，已注册 %d 个通道", len(self._channels))

    async def stop(self) -> None:
        for channel in self._channels.values():
            await channel.stop()

    # ── 主处理流程 / Main pipeline ────────────────────────────────────────────────────────────

    async def handle_inbound(self, envelope: InboundEnvelope) -> ChannelSendResult | None:
        """
        处理一条入站信封，返回发送结果（不回复时返回 None）。 / Process one inbound envelope; returns send result (None when no reply).

        流水线异常不会向上抛，全部降级为日志——保证单条消息异常不拖垮网关。 / Pipeline exceptions are not re-raised; all degrade to logs—so a single message error won't take down the gateway.
        """
        # 1. 去重 / 1. Dedup
        dedup = self._dedup.check(envelope.envelope_id)
        if dedup.duplicate:
            log.debug("丢弃重复消息: %s", dedup.reason)
            return None

        # 2. 安全检查 / 2. Safety check
        blocked_reason = self._check_safety(envelope)
        if blocked_reason:
            log.info("安全门控拦截: %s (sender=%s)", blocked_reason, envelope.sender_id)
            return None

        # 3. 路由 / 3. Routing
        agent_id = self._router.resolve(envelope)
        session_key = make_session_key(agent_id, envelope)

        # 4. 车道分发（串行同会话，并行跨会话） / 4. Lane dispatch (serial within session, parallel across sessions)
        if self._handler is None:
            log.warning("未设置 handler，丢弃消息")
            return None

        try:
            outbound = await self._lane.dispatch(
                session_key,
                lambda: self._handler(envelope),
            )
        except Exception as exc:
            log.warning("handler 处理异常（session=%s）: %s", session_key, exc)
            return None

        if outbound is None:
            return None   # handler 主动不回复 / handler chose not to reply

        # 5. 发送回复 / 5. Send reply
        channel = self._channels.get(outbound.target_channel)
        if channel is None:
            log.warning("目标通道未注册: %s", outbound.target_channel)
            return None

        return await channel.send(outbound)

    def _check_safety(self, envelope: InboundEnvelope) -> str:
        """
        安全门控。返回非空字符串表示拦截原因，空串表示放行。 / Safety gate. Non-empty string means block reason; empty string means allow.

        检查项： / Checks:
          - 黑名单 / blacklist
          - 白名单（非空时生效） / whitelist (effective when non-empty)
          - 群聊 @机器人门控 / group @bot gate
        """
        s = self._safety
        if envelope.sender_id in s.blocked_senders:
            return f"发送者在黑名单: {envelope.sender_id}"
        if s.allowed_senders and envelope.sender_id not in s.allowed_senders:
            return f"发送者不在白名单: {envelope.sender_id}"
        if (s.group_require_mention
                and envelope.chat_type.value == "group"
                and not envelope.mentions_bot):
            return "群聊未 @机器人"
        return ""

    # ── 可观测性辅助 / Observability helpers ──────────────────────────────────────────────────

    @property
    def dedup_size(self) -> int:
        """当前去重窗口条目数，供监控面板展示。 / Current dedup window entry count, for monitoring dashboards."""
        return len(self._dedup)

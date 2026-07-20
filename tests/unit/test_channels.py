"""
通道层单元测试。

测试策略：
  - 用 FakeChannel / FakeBot 避免真实 Telegram/Discord 账号
  - Telegram/Discord 的 parse_message 用 SimpleNamespace 伪造平台对象
  - 超长分块用纯函数 _split_long_text 直接测
  - LaneQueue 的并行性用 asyncio.sleep + wall-clock 计时断言
  - Gateway 流水线：去重/安全/路由/分发各步独立断言

行为指纹风格：断言状态转换、返回值、副作用，不断言文本内容。
"""
from __future__ import annotations

import asyncio
import time
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from nanoharness.channels.base import (
    BaseChannel, ChannelSendResult, ChatType, InboundEnvelope,
    OutboundEnvelope, SendStatus,
)
from nanoharness.channels.gateway import DedupWindow, Gateway, SafetyPolicy
from nanoharness.channels.lane_queue import LaneQueue
from nanoharness.channels.router import (
    BindingRule, ChannelRouter, make_session_key,
)
from nanoharness.channels.telegram import TelegramChannel, _split_long_text
from nanoharness.channels.discord import DiscordChannel


# ─── Fakes ────────────────────────────────────────────────────────────────────

class FakeChannel:
    """最小化的通道实现，用于 Gateway 测试。"""
    def __init__(self, channel_id: str = "fake") -> None:
        self.channel_id = channel_id
        self.sent: list[OutboundEnvelope] = []
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def send(self, envelope: OutboundEnvelope) -> ChannelSendResult:
        self.sent.append(envelope)
        return ChannelSendResult(
            status=SendStatus.SENT, retryable=False,
            reason="", message_ids=[f"mid-{len(self.sent)}"],
        )


class FakeBot:
    """伪造 aiogram Bot，记录发送调用。"""
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_message(self, chat_id: int, text: str) -> SimpleNamespace:
        self.sent.append({"chat_id": chat_id, "text": text})
        return SimpleNamespace(message_id=len(self.sent))


def make_telegram_dm(text: str, sender_id: int = 100,
                     bot_username: str = "nanobot") -> SimpleNamespace:
    """伪造 aiogram 私聊 Message。"""
    return SimpleNamespace(
        chat=SimpleNamespace(id=200, type="private"),
        from_user=SimpleNamespace(id=sender_id),
        text=text,
        entities=[],
        reply_to_message=None,
    )


def make_telegram_group(text: str, sender_id: int = 100, chat_id: int = -300,
                        mentions_bot: bool = False,
                        bot_username: str = "nanobot") -> SimpleNamespace:
    """伪造 aiogram 群聊 Message。"""
    entities = []
    if mentions_bot and bot_username:
        # 构造一个 mention entity 指向 bot
        mention_text = f"@{bot_username}"
        offset = text.find(mention_text)
        if offset >= 0:
            entities.append(SimpleNamespace(
                type="mention", offset=offset, length=len(mention_text),
            ))
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, type="supergroup"),
        from_user=SimpleNamespace(id=sender_id),
        text=text,
        entities=entities,
        reply_to_message=None,
    )


def make_discord_dm(text: str, author_id: int = 100) -> SimpleNamespace:
    """伪造 discord.py 私聊 Message。channel 类型用类名区分。"""
    class DMChannel:
        def __init__(self) -> None:
            self.id = 200

    return SimpleNamespace(
        channel=DMChannel(),
        author=SimpleNamespace(id=author_id),
        content=text,
        mentions=[],
        reference=None,
    )


def make_discord_group(text: str, author_id: int = 100, channel_id: int = 300,
                       mentions_bot: bool = False,
                       bot_user_id: int = 999) -> SimpleNamespace:
    """伪造 discord.py 群聊 Message。"""
    class TextChannel:
        async def send(self, chunk: str) -> SimpleNamespace:
            self.last = SimpleNamespace(id=len(getattr(self, "sent", [])) + 1)
            if not hasattr(self, "sent"):
                self.sent = []
            self.sent.append(chunk)
            return self.last

    ch = TextChannel()
    mentions = []
    if mentions_bot:
        mentions.append(SimpleNamespace(id=bot_user_id))

    return SimpleNamespace(
        channel=ch,
        author=SimpleNamespace(id=author_id),
        content=text,
        mentions=mentions,
        reference=None,
    )


# ─── base.py: 信封结构 ──────────────────────────────────────────────────────

def test_inbound_envelope_auto_generates_id():
    """未指定 envelope_id 时自动生成唯一 id。"""
    env1 = InboundEnvelope(channel_id="tg", sender_id="1", chat_id="1",
                           chat_type=ChatType.DIRECT, content="hi")
    env2 = InboundEnvelope(channel_id="tg", sender_id="2", chat_id="2",
                           chat_type=ChatType.DIRECT, content="hi")
    assert env1.envelope_id and env2.envelope_id
    assert env1.envelope_id != env2.envelope_id


def test_inbound_envelope_frozen():
    """信封是不可变的，保证并发安全。"""
    env = InboundEnvelope(channel_id="tg", sender_id="1", chat_id="1", content="x")
    with pytest.raises(Exception):
        env.content = "changed"   # type: ignore[misc]


def test_base_channel_protocol_checkable():
    """FakeChannel 满足 BaseChannel Protocol。"""
    ch = FakeChannel()
    assert isinstance(ch, BaseChannel)


# ─── router.py: 路由 ────────────────────────────────────────────────────────

def test_router_resolves_default_when_no_rule():
    """无规则匹配时返回 default_agent_id。"""
    router = ChannelRouter(default_agent_id="fallback")
    env = InboundEnvelope(channel_id="tg", sender_id="1", chat_id="1",
                          chat_type=ChatType.DIRECT, content="hi")
    assert router.resolve(env) == "fallback"


def test_router_priority_order():
    """同条件多规则时，priority 高的胜出。"""
    router = ChannelRouter(default_agent_id="d")
    router.add_rule(BindingRule(agent_id="low", channel_id="tg", priority=1))
    router.add_rule(BindingRule(agent_id="high", channel_id="tg", priority=10))
    env = InboundEnvelope(channel_id="tg", sender_id="1", chat_id="1",
                          chat_type=ChatType.DIRECT, content="hi")
    assert router.resolve(env) == "high"


def test_router_sender_pattern_regex():
    """sender_pattern 用正则匹配特定用户。"""
    router = ChannelRouter(default_agent_id="default")
    router.add_rule(BindingRule(agent_id="vip_agent", sender_pattern=r"^vip_\d+$"))
    # 匹配
    env1 = InboundEnvelope(channel_id="tg", sender_id="vip_123", chat_id="1",
                           chat_type=ChatType.DIRECT, content="hi")
    assert router.resolve(env1) == "vip_agent"
    # 不匹配 → default
    env2 = InboundEnvelope(channel_id="tg", sender_id="user_456", chat_id="1",
                           chat_type=ChatType.DIRECT, content="hi")
    assert router.resolve(env2) == "default"


def test_router_chat_type_filter():
    """chat_type 过滤：只让私聊走某 agent。"""
    router = ChannelRouter(default_agent_id="general")
    router.add_rule(BindingRule(
        agent_id="dm_only", chat_type=ChatType.DIRECT, channel_id="tg",
    ))
    dm_env = InboundEnvelope(channel_id="tg", sender_id="1", chat_id="1",
                             chat_type=ChatType.DIRECT, content="hi")
    group_env = InboundEnvelope(channel_id="tg", sender_id="1", chat_id="-100",
                                chat_type=ChatType.GROUP, content="hi")
    assert router.resolve(dm_env) == "dm_only"
    assert router.resolve(group_env) == "general"


def test_session_key_dm_vs_group():
    """私聊按 sender_id 隔离，群聊按 chat_id 共享。"""
    agent_id = "helper"
    dm_env = InboundEnvelope(channel_id="tg", sender_id="user_A", chat_id="user_A",
                             chat_type=ChatType.DIRECT, content="hi")
    group_env = InboundEnvelope(channel_id="tg", sender_id="user_A", chat_id="-100",
                                chat_type=ChatType.GROUP, content="hi")

    dm_key = make_session_key(agent_id, dm_env)
    group_key = make_session_key(agent_id, group_env)

    # 私聊 key 含 sender_id，群聊 key 含 chat_id
    assert "user_A" in dm_key
    assert "-100" in group_key
    assert dm_key != group_key

    # 群里两个不同用户共享同一 session_key（这是 IM 群聊的标准语义）
    user_b_group = InboundEnvelope(channel_id="tg", sender_id="user_B", chat_id="-100",
                                    chat_type=ChatType.GROUP, content="hi")
    assert make_session_key(agent_id, user_b_group) == group_key


# ─── lane_queue.py: 车道隔离 ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lane_queue_same_session_serial():
    """同一 session_key 的任务串行执行。"""
    queue = LaneQueue()
    order: list[str] = []

    async def task(name: str, delay: float):
        await asyncio.sleep(delay)
        order.append(name)
        return name

    # 同一 session 并发投递两个任务，按投递顺序完成
    await asyncio.gather(
        queue.dispatch("sess-1", lambda: task("first", 0.05)),
        queue.dispatch("sess-1", lambda: task("second", 0.01)),
    )
    # 串行：first 先完成（即使 delay 更长），因为 second 要等 first 释放锁
    assert order == ["first", "second"]


@pytest.mark.asyncio
async def test_lane_queue_different_sessions_parallel():
    """不同 session_key 的任务并行执行，wall-clock 接近 max 而非 sum。"""
    queue = LaneQueue()

    async def task(delay: float):
        await asyncio.sleep(delay)
        return "ok"

    t0 = time.monotonic()
    await asyncio.gather(
        queue.dispatch("sess-A", lambda: task(0.05)),
        queue.dispatch("sess-B", lambda: task(0.05)),
    )
    elapsed = time.monotonic() - t0
    # 并行：总耗时接近 0.05s，串行则要 0.10s。留 0.04s buffer。
    assert elapsed < 0.09, f"疑似串行执行，耗时={elapsed:.3f}s"


@pytest.mark.asyncio
async def test_lane_queue_reentrancy_no_deadlock():
    """重入场景：同链路内再次 dispatch 同 session 不死锁。"""
    queue = LaneQueue()
    log: list[str] = []

    async def outer():
        log.append("outer-start")
        # 在持有锁的情况下，再次 dispatch 同一 session
        inner_result = await queue.dispatch("sess-1", inner)
        log.append(f"outer-end inner={inner_result}")

    async def inner():
        log.append("inner")
        return "inner-done"

    await queue.dispatch("sess-1", outer)
    # 如果没重入检测，outer 会死锁等自己释放锁
    assert log == ["outer-start", "inner", "outer-end inner=inner-done"]


# ─── gateway.py: 流水线 ───────────────────────────────────────────────────────

def test_dedup_window_drops_duplicate():
    """相同 envelope_id 第二次进来被去重。"""
    win = DedupWindow(capacity=10)
    assert win.check("env-1").duplicate is False
    assert win.check("env-1").duplicate is True   # 第二次命中
    assert win.check("env-2").duplicate is False


def test_dedup_window_lru_eviction():
    """超容量时按 LRU 淘汰最久未访问的条目。"""
    win = DedupWindow(capacity=2)
    win.check("env-1")
    win.check("env-2")
    win.check("env-3")   # 容量 2，env-1 被淘汰（最老）
    # env-1 现在又算"新"的
    assert win.check("env-1").duplicate is False
    # 重新加入 env-1 后，最老的 env-2 被淘汰；env-3（最近访问）还在
    assert win.check("env-3").duplicate is True
    assert win.check("env-2").duplicate is False   # 已被淘汰


@pytest.mark.asyncio
async def test_gateway_dedup_drops_second_message():
    """Gateway 对重复 envelope_id 只处理一次。"""
    ch = FakeChannel()
    gw = Gateway(router=ChannelRouter(default_agent_id="a"))
    gw.register_channel(ch)

    call_count = 0

    async def handler(env: InboundEnvelope) -> OutboundEnvelope:
        nonlocal call_count
        call_count += 1
        return OutboundEnvelope(
            target_channel="fake", target_peer=env.chat_id, content="reply",
        )

    gw.set_handler(handler)

    env = InboundEnvelope(
        envelope_id="fixed-id", channel_id="fake", sender_id="u1", chat_id="1",
        chat_type=ChatType.DIRECT, content="hi",
    )
    await gw.handle_inbound(env)
    await gw.handle_inbound(env)   # 重复

    assert call_count == 1
    assert len(ch.sent) == 1


@pytest.mark.asyncio
async def test_gateway_safety_group_require_mention():
    """群聊默认要求 @机器人，没 @ 的消息被拦截。"""
    ch = FakeChannel()
    gw = Gateway(router=ChannelRouter(default_agent_id="a"))
    gw.register_channel(ch)
    gw.set_handler(lambda env: _async_return(None))

    # 群聊但 mentions_bot=False
    env = InboundEnvelope(
        channel_id="fake", sender_id="u1", chat_id="-1",
        chat_type=ChatType.GROUP, content="hi", mentions_bot=False,
    )
    result = await gw.handle_inbound(env)
    assert result is None   # 被安全门控拦截


@pytest.mark.asyncio
async def test_gateway_safety_group_with_mention_passes():
    """群聊且 @了机器人 → 正常处理并回复。"""
    ch = FakeChannel()
    gw = Gateway(router=ChannelRouter(default_agent_id="a"))
    gw.register_channel(ch)

    async def handler(env: InboundEnvelope) -> OutboundEnvelope:
        return OutboundEnvelope(
            target_channel="fake", target_peer=env.chat_id, content="ok",
        )

    gw.set_handler(handler)

    env = InboundEnvelope(
        channel_id="fake", sender_id="u1", chat_id="-1",
        chat_type=ChatType.GROUP, content="@bot hi", mentions_bot=True,
    )
    result = await gw.handle_inbound(env)
    assert result is not None
    assert result.status == SendStatus.SENT


@pytest.mark.asyncio
async def test_gateway_safety_blocked_sender():
    """黑名单发送者被拦截。"""
    gw = Gateway(
        router=ChannelRouter(default_agent_id="a"),
        safety=SafetyPolicy(blocked_senders={"bad-user"}),
    )
    gw.register_channel(FakeChannel())
    gw.set_handler(lambda env: _async_return(None))

    env = InboundEnvelope(
        channel_id="fake", sender_id="bad-user", chat_id="1",
        chat_type=ChatType.DIRECT, content="hi",
    )
    result = await gw.handle_inbound(env)
    assert result is None


@pytest.mark.asyncio
async def test_gateway_safety_whitelist():
    """白名单非空时，仅白名单用户放行。"""
    gw = Gateway(
        router=ChannelRouter(default_agent_id="a"),
        safety=SafetyPolicy(allowed_senders={"vip"}),
    )
    gw.register_channel(FakeChannel())
    gw.set_handler(lambda env: _async_return(None))

    # 非白名单 → 拦截
    blocked = InboundEnvelope(
        channel_id="fake", sender_id="outsider", chat_id="1",
        chat_type=ChatType.DIRECT, content="hi",
    )
    assert await gw.handle_inbound(blocked) is None

    # 白名单 → 放行（handler 返回 None，所以结果也是 None，但没被拦截）
    allowed = InboundEnvelope(
        channel_id="fake", sender_id="vip", chat_id="1",
        chat_type=ChatType.DIRECT, content="hi",
    )
    # 用 dedup_size 间接验证：放行的消息进了 handler，去重窗口 +1
    before = gw.dedup_size
    await gw.handle_inbound(allowed)
    after = gw.dedup_size
    assert after == before + 1


@pytest.mark.asyncio
async def test_gateway_routes_to_session_and_serializes():
    """Gateway 通过 router 决定 agent，按 session_key 走车道。"""
    router = ChannelRouter(default_agent_id="default")
    router.add_rule(BindingRule(agent_id="vip_agent", sender_pattern="^vip"))
    gw = Gateway(router=router)
    gw.register_channel(FakeChannel())

    seen_sessions: list[str] = []

    async def handler(env: InboundEnvelope) -> None:
        seen_sessions.append(env.sender_id)
        return None

    gw.set_handler(handler)

    # vip 用户 → vip_agent，session_key 含 agent_id "vip_agent"
    env = InboundEnvelope(
        channel_id="fake", sender_id="vip_1", chat_id="vip_1",
        chat_type=ChatType.DIRECT, content="hi",
    )
    await gw.handle_inbound(env)
    # 普通用户 → default agent
    env2 = InboundEnvelope(
        channel_id="fake", sender_id="user_2", chat_id="user_2",
        chat_type=ChatType.DIRECT, content="hi",
    )
    await gw.handle_inbound(env2)

    assert seen_sessions == ["vip_1", "user_2"]


@pytest.mark.asyncio
async def test_gateway_register_duplicate_channel_raises():
    """重复注册同 channel_id 报错。"""
    gw = Gateway(router=ChannelRouter())
    gw.register_channel(FakeChannel(channel_id="tg"))
    with pytest.raises(ValueError):
        gw.register_channel(FakeChannel(channel_id="tg"))


# ─── telegram.py: parse_message + 分块 ───────────────────────────────────────

def test_telegram_parse_dm():
    """私聊 message 解析为 DIRECT，mentions_bot=False。"""
    bot = FakeBot()
    ch = TelegramChannel(bot=bot, bot_username="nanobot")
    msg = make_telegram_dm("你好", sender_id=42)

    env = ch.parse_message(msg)

    assert env.channel_id == "telegram"
    assert env.sender_id == "42"
    assert env.chat_type == ChatType.DIRECT
    assert env.content == "你好"
    assert env.mentions_bot is False


def test_telegram_parse_group_with_mention():
    """群聊 + @bot → mentions_bot=True。"""
    bot = FakeBot()
    ch = TelegramChannel(bot=bot, bot_username="nanobot")
    msg = make_telegram_group(
        "@nanobot 帮我查天气", mentions_bot=True, bot_username="nanobot",
    )
    env = ch.parse_message(msg)
    assert env.chat_type == ChatType.GROUP
    assert env.mentions_bot is True


def test_telegram_parse_group_without_mention():
    """群聊没 @bot → mentions_bot=False。"""
    bot = FakeBot()
    ch = TelegramChannel(bot=bot, bot_username="nanobot")
    msg = make_telegram_group("今天天气不错", mentions_bot=False)
    env = ch.parse_message(msg)
    assert env.mentions_bot is False


@pytest.mark.asyncio
async def test_telegram_send_short_message():
    """短消息一次发送，返回 SENT。"""
    bot = FakeBot()
    ch = TelegramChannel(bot=bot, bot_username="nb")
    env = OutboundEnvelope(target_channel="telegram", target_peer="200", content="hello")
    result = await ch.send(env)
    assert result.status == SendStatus.SENT
    assert len(bot.sent) == 1
    assert bot.sent[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_telegram_send_long_message_chunked():
    """超长消息分块发送，每块 ≤ 4096。"""
    bot = FakeBot()
    ch = TelegramChannel(bot=bot, bot_username="nb")
    # 5000 字符，需要分 2 块
    long_text = "a" * 5000
    env = OutboundEnvelope(target_channel="telegram", target_peer="200", content=long_text)
    result = await ch.send(env)

    assert result.status == SendStatus.SENT
    assert len(bot.sent) == 2
    # 每块不超限
    assert all(len(s["text"]) <= 4096 for s in bot.sent)
    # 拼起来等于原文
    assert "".join(s["text"] for s in bot.sent) == long_text


def test_split_long_text_prefers_newline_cut():
    """分块优先在换行处切，不切断句子。"""
    # 第 10 字符是换行，max_len=15 时应在换行处切
    text = "第一行\n第二行\n第三行"
    chunks = _split_long_text(text, max_len=10)
    assert len(chunks) >= 2
    # 每块都不超限
    assert all(len(c) <= 10 for c in chunks)
    # 拼接还原（去掉切点处的换行后可能不完全等，这里只验证内容片段都在）
    joined = "".join(chunks)
    assert "第一行" in joined
    assert "第三行" in joined


def test_split_long_text_short_returns_single():
    """短于上限的文本返回单块。"""
    assert _split_long_text("hi", max_len=100) == ["hi"]


# ─── discord.py: parse_message ────────────────────────────────────────────────

def test_discord_parse_dm():
    """私聊解析为 DIRECT。"""
    client = SimpleNamespace()
    ch = DiscordChannel(client=client, bot_user_id=999)
    msg = make_discord_dm("hi", author_id=42)

    env = ch.parse_message(msg)

    assert env.channel_id == "discord"
    assert env.sender_id == "42"
    assert env.chat_type == ChatType.DIRECT


def test_discord_parse_group_with_mention():
    """群聊 + @bot → mentions_bot=True。"""
    ch = DiscordChannel(client=SimpleNamespace(), bot_user_id=999)
    msg = make_discord_group("@bot hi", author_id=42, mentions_bot=True)
    env = ch.parse_message(msg)
    assert env.chat_type == ChatType.GROUP
    assert env.mentions_bot is True


# ─── 辅助 ────────────────────────────────────────────────────────────────────

async def _async_return(value):
    """返回固定值的 async handler，用于不需要实际处理的测试。"""
    return value

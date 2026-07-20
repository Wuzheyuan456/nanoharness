"""
消息路由引擎。

声明式 BindingRule：把入站信封匹配到目标 agent_id。
匹配维度：channel_id / chat_type / sender_pattern / group_pattern，全部可选，
全 None = 兜底规则。按注册顺序匹配，第一个命中的胜出。

session_key 生成：
  - 私聊：agent:{agent_id}:dm:{channel_id}:{sender_id}
    按"用户×agent"隔离，同一用户和不同 agent 对话互不串台
  - 群聊：agent:{agent_id}:group:{channel_id}:{chat_id}
    按"群×agent"隔离，群里所有人共享一个会话上下文

面试话术：
"路由用声明式规则而不是 if-else，加规则只 append 不改代码。
session_key 的设计是个细节：私聊按 sender_id 隔离保证用户隐私
（A 和 bot 的对话 B 看不到），群聊按 chat_id 共享让群里所有人
有共同上下文——这是 IM 机器人的标准语义。"
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from nanoharness.channels.base import ChatType, InboundEnvelope


@dataclass(frozen=True)
class BindingRule:
    """
    声明式绑定规则。所有字段可选，None 表示"匹配任意"。

    例如只让 Telegram 私聊走 agent "helper"：
        BindingRule(channel_id="telegram", chat_type=ChatType.DIRECT, agent_id="helper")
    """
    agent_id: str
    channel_id: str | None = None       # 限定通道，None=任意通道
    chat_type: ChatType | None = None   # 限定会话类型，None=任意
    sender_pattern: str | None = None   # 发送者 id 正则，None=任意发送者
    group_pattern: str | None = None     # 群 id 正则，None=任意群
    priority: int = 0                    # 优先级，高者优先；同优先级按注册顺序


def _match_pattern(pattern: str | None, value: str) -> bool:
    """正则匹配，None 视为通配。"""
    if pattern is None:
        return True
    return re.search(pattern, value) is not None


class ChannelRouter:
    """
    通道路由器：按规则把信封解析到 agent_id。

    匹配规则：
      1. 所有字段都匹配的规则胜出
      2. 多个匹配时，priority 高的胜出
      3. 同 priority 按注册顺序，第一个胜出
      4. 无任何匹配 → default_agent_id
    """

    def __init__(self, default_agent_id: str = "default") -> None:
        self._rules: list[BindingRule] = []
        self._default = default_agent_id

    def add_rule(self, rule: BindingRule) -> None:
        self._rules.append(rule)
        # 按优先级降序排序，priority 高的在前
        self._rules.sort(key=lambda r: -r.priority)

    def resolve(self, envelope: InboundEnvelope) -> str:
        """返回目标 agent_id。"""
        for rule in self._rules:
            if self._matches(rule, envelope):
                return rule.agent_id
        return self._default

    @staticmethod
    def _matches(rule: BindingRule, env: InboundEnvelope) -> bool:
        if rule.channel_id is not None and rule.channel_id != env.channel_id:
            return False
        if rule.chat_type is not None and rule.chat_type != env.chat_type:
            return False
        if not _match_pattern(rule.sender_pattern, env.sender_id):
            return False
        if not _match_pattern(rule.group_pattern, env.chat_id):
            return False
        return True


def make_session_key(agent_id: str, envelope: InboundEnvelope) -> str:
    """
    根据会话类型生成 session_key。

    私聊：按 sender_id 隔离（用户×agent 独立上下文）
    群聊/频道/帖：按 chat_id 共享（群内共享上下文）
    """
    if envelope.chat_type == ChatType.DIRECT:
        return f"agent:{agent_id}:dm:{envelope.channel_id}:{envelope.sender_id}"
    # 群聊/频道/帖统一用 chat_id 作为会话隔离维度
    return f"agent:{agent_id}:group:{envelope.channel_id}:{envelope.chat_id}"

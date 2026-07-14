from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentCard:
    """
    Agent 的能力声明卡片，对齐 Google A2A 协议的 capabilities/endpoint 设计。

    注册中心靠它做能力路由：Orchestrator 不依赖具体 Agent 类，
    只通过 required_capability 字符串从 Registry 查询匹配的 Card。

    面试话术：
    "AgentCard 是 Agent 的自我描述文档，类似 OpenAPI spec 之于 REST 服务。
    新增一个 Agent 只需要注册一张 Card，Orchestrator 代码零修改。
    capabilities 是松散的字符串标签，不是枚举，方便业务自定义扩展。"
    """
    agent_id: str
    description: str
    capabilities: list[str]                          # 能力标签，如 ["code_review", "search"]
    system_prompt: str
    default_tier: str = "T1"                         # 对应 TierRegistry 中的 T0~T3
    tools: dict[str, Any] = field(default_factory=dict)
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    max_concurrent: int = 3                          # 同一 agent 最多同时运行几个实例


class AgentRegistry:
    """
    全局 Agent Card 注册中心。

    Orchestrator 通过 lookup_by_capability() 找到合适的 Agent Card，
    再根据 Card 的配置 spawn 对应的 Worker NanoCore 实例。

    面试话术：
    "Registry 是 Orchestrator 和具体 Agent 之间的解耦层。
    Orchestrator 说'我需要一个能 code_review 的 Agent'，
    Registry 返回匹配的 Card，Orchestrator 不知道背后是哪个类、哪个模型。
    这也是面向 A2A 协议设计的基础——以后 Agent 可以是远程服务，
    Card 里加一个 endpoint 字段就能透明切换本地/远程。"
    """

    def __init__(self) -> None:
        self._cards: dict[str, AgentCard] = {}   # agent_id → AgentCard

    def register(self, card: AgentCard) -> None:
        """注册一张 AgentCard，重复 agent_id 会覆盖旧版本。"""
        self._cards[card.agent_id] = card

    def lookup(self, agent_id: str) -> AgentCard | None:
        return self._cards.get(agent_id)

    def lookup_by_capability(self, capability: str) -> list[AgentCard]:
        """返回所有包含指定能力标签的 AgentCard（可能多个）。"""
        return [card for card in self._cards.values() if capability in card.capabilities]

    def list_all(self) -> list[AgentCard]:
        return list(self._cards.values())

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._cards

"""
Agent 调度器：统一的 InboundHandler 实现，根据消息复杂度自动路由到 TurnRunner 或 Orchestrator。

设计动机：
  Gateway.set_handler() 接受一个 InboundHandler（Callable[[InboundEnvelope], Awaitable[OutboundEnvelope|None]]）。
  AgentDispatcher 实现这个接口，在内部做复杂度判断：
    - 简单任务 → TurnRunner（单 Agent ReAct 循环）
    - 复杂任务 → Orchestrator（多 Agent 拆解/路由/并行/综合）
  Gateway 完全不感知背后是哪条路径——handler 是注入的，Gateway 只认接口。

  判断逻辑故意放在 dispatcher 而非 gateway：gateway 是通道控制平面，不应该懂业务复杂度；
  dispatcher 是业务调度层，它懂任务特征。
"""
from __future__ import annotations

import logging
from typing import Any

from nanoharness.agents.orchestrator import Orchestrator
from nanoharness.channels.base import InboundEnvelope, OutboundEnvelope
from nanoharness.channels.router import make_session_key
from nanoharness.core.context import AgentContext
from nanoharness.core.event_store import DoneEvent
from nanoharness.engine.turn_runner import TurnRunner

log = logging.getLogger(__name__)

# 触发 Orchestrator 路由的关键词（暗示并行多子任务）/ Keywords that trigger Orchestrator routing (implying parallel sub-tasks)
_COMPLEX_KEYWORDS: tuple[str, ...] = ("分别", "同时", "并行", "另外还要")

_DEFAULT_SYSTEM_PROMPT = "你是一个智能助手，请尽力完成用户的任务。"


class AgentDispatcher:
    """
    统一 Agent 调度器，实现 InboundHandler 接口，可直接注入 Gateway.set_handler()。

    路由策略（两档）：
      - 消息长度 > complex_threshold 或包含复杂度关键词 → Orchestrator（多 Agent）
      - 否则 → TurnRunner（单 Agent）
    未传入 orchestrator 时永远走 TurnRunner（单 Agent 模式）。

    面试话术 / Interview talking point：
    "Gateway 的 handler 是注入的 callable，不感知背后是单 Agent 还是多 Agent。
    AgentDispatcher 实现了这个 callable，在内部做复杂度判断——
    简短问题走 TurnRunner 单 Agent，包含'分别/同时/并行'等关键词或超过阈值长度的请求
    走 Orchestrator 多 Agent 拆解。判断逻辑在 dispatcher 里，Gateway 零修改。
    这是策略模式和依赖注入的结合：Gateway 定义接口，dispatcher 实现策略，两者都能独立测试。

    Gateway's handler is an injected callable, oblivious to whether it's a single Agent or multi-Agent behind it.
    AgentDispatcher implements that callable and makes complexity decisions internally —
    short questions go to TurnRunner single-Agent, while requests containing keywords like
    'separately/simultaneously/in parallel' or exceeding the length threshold go to the Orchestrator multi-Agent.
    The decision logic lives in the dispatcher; Gateway needs zero changes.
    This is a combination of the Strategy pattern and dependency injection: Gateway defines the interface,
    dispatcher implements the strategy, and both can be tested independently."
    """

    def __init__(
        self,
        turn_runner: TurnRunner,
        orchestrator: Orchestrator | None = None,
        agent_id: str = "assistant",
        model_id: str = "claude-haiku-20240307",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        complex_threshold: int = 200,                    # 消息超过此字符数视为复杂任务 / Messages over this char count are treated as complex
        complex_keywords: tuple[str, ...] = _COMPLEX_KEYWORDS,
    ) -> None:
        self._turn_runner = turn_runner
        self._orchestrator = orchestrator
        self._agent_id = agent_id
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._threshold = complex_threshold
        self._keywords = complex_keywords

    async def __call__(self, envelope: InboundEnvelope) -> OutboundEnvelope | None:
        """InboundHandler 实现：处理一条入站信封，返回出站回复或 None（不回复）。"""
        ctx = self._build_ctx(envelope)

        if self._orchestrator and self._is_complex(envelope.content):
            log.info("路由到 Orchestrator（复杂任务）: session=%s", ctx.session_key)
            result = await self._orchestrator.run(envelope.content, ctx)
            text = result.final_synthesis
        else:
            log.info("路由到 TurnRunner（简单任务）: session=%s", ctx.session_key)
            text = await self._run_turn(ctx, envelope.content)

        if not text:
            return None
        return OutboundEnvelope(
            target_channel=envelope.channel_id,
            target_peer=envelope.chat_id,
            content=text,
            reply_to_envelope_id=envelope.envelope_id,
        )

    def _build_ctx(self, envelope: InboundEnvelope) -> AgentContext:
        """从 InboundEnvelope 构造 AgentContext，session_key 遵循通道路由惯例。"""
        session_key = make_session_key(self._agent_id, envelope)
        return AgentContext(
            agent_id=self._agent_id,
            session_key=session_key,
            system_prompt=self._system_prompt,
            model_id=self._model_id,
        )

    def _is_complex(self, content: str) -> bool:
        """启发式复杂度判断：长消息或包含并行任务关键词，任一满足则走 Orchestrator。"""
        if len(content) > self._threshold:
            return True
        return any(kw in content for kw in self._keywords)

    async def _run_turn(self, ctx: AgentContext, user_message: str) -> str:
        """驱动 TurnRunner async generator，提取 DoneEvent.final_text。"""
        text = ""
        async for event in self._turn_runner.run(ctx, user_message):
            if isinstance(event, DoneEvent):
                text = event.final_text
        return text

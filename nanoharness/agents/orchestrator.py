from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from nanoharness.agents.registry import AgentCard, AgentRegistry
from nanoharness.core.context import AgentContext, Message
from nanoharness.core.event_store import DoneEvent
from nanoharness.core.nano_core import NanoCore
from nanoharness.provider.base import LLMProvider

log = logging.getLogger(__name__)

# ─── 深度保护：防止 Orchestrator 内部再嵌套 Orchestrator 无限递归 ────────────────
# 每个 asyncio Task 持有自己的副本，不同用户会话互不干扰（同 TurnRunner._LOCK_OWNER 设计）
_ORCHESTRATION_DEPTH: ContextVar[int] = ContextVar("_orchestration_depth", default=0)

# ─── 系统提示 ──────────────────────────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """\
你是任务分解专家。把用户的复杂任务拆成 2~5 个可以独立并行执行的子任务。

只输出 JSON，格式如下：
{
  "subtasks": [
    {
      "description": "子任务的详细描述，包含足够的上下文",
      "required_capability": "需要的能力关键词（如 code_review / search / summarize / math）"
    }
  ]
}

注意：
- 每个子任务必须是自包含的，不依赖其他子任务的输出
- 如果任务本身已经足够简单，输出 1 个子任务即可
- required_capability 要精确，这决定了哪个 Agent 会处理这个子任务
"""

_SYNTHESIZE_SYSTEM = """\
你是一个高级 AI 助手。以下是一个复杂任务被拆成子任务后，各个 Worker Agent 分别完成的结果。
请综合所有子任务的输出，给出一个完整、连贯的最终答复，不要重复每个子任务的标题。
"""


# ─── 数据模型 ──────────────────────────────────────────────────────────────────

@dataclass
class SubtaskSpec:
    """Orchestrator 对一个子任务的描述，由 LLM 分解生成。"""
    index: int
    description: str
    required_capability: str


@dataclass
class SubtaskResult:
    """一个 Worker 完成子任务后的结果。"""
    spec: SubtaskSpec
    agent_id: str
    output: str
    success: bool
    elapsed_ms: float


@dataclass
class OrchestratorResult:
    """整个 Supervisor 编排的最终结果。"""
    original_task: str
    subtask_results: list[SubtaskResult]
    final_synthesis: str
    total_elapsed_ms: float

    @property
    def succeeded_count(self) -> int:
        return sum(1 for r in self.subtask_results if r.success)

    @property
    def failed_count(self) -> int:
        return len(self.subtask_results) - self.succeeded_count


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Supervisor 模式编排器。

    四步流程：
    1. 拆解（Decompose）：一次 T0 级 LLM call，把复杂任务分解为 N 个子任务规格（JSON）
    2. 路由（Route）：从 AgentRegistry 按 required_capability 匹配 AgentCard
    3. 执行（Execute）：asyncio.gather 并行 spawn N 个 Worker NanoCore，收集结果
    4. 综合（Synthesize）：T1 级 LLM call 把 N 份输出合并为最终答复

    面试话术：
    "Orchestrator 的核心是分离关注点：LLM 知道如何拆解任务（第一步），
    Registry 知道哪个 Agent 有对应能力（第二步），
    NanoCore 知道如何执行（第三步），三者互不耦合。
    asyncio.gather 保证独立子任务真并行，不是串行模拟。
    ContextVar 深度计数确保嵌套 Orchestrator 最多 max_depth 层，
    防止 Agent 无限 spawn Agent 耗尽资源。"
    """

    def __init__(
        self,
        registry: AgentRegistry,
        provider_factory: dict[str, LLMProvider],   # tier_str → provider（如 {"T0": ..., "T1": ...}）
        max_workers: int = 5,                        # 单次编排最多并发的 Worker 数
        max_depth: int = 2,                          # 最大嵌套编排深度
        fallback_capability: str = "general",        # 无能力匹配时的兜底标签
    ) -> None:
        self._registry = registry
        self._providers = provider_factory
        self._max_workers = max_workers
        self._max_depth = max_depth
        self._fallback_capability = fallback_capability

    async def run(
        self,
        task: str,
        parent_ctx: AgentContext,
    ) -> OrchestratorResult:
        """
        主入口。接受复杂任务，返回 OrchestratorResult。
        parent_ctx 只用于继承 session_key 前缀（worker 会用新的 session_key）。
        """
        depth = _ORCHESTRATION_DEPTH.get()
        if depth >= self._max_depth:
            # 深度超限：不再拆解，直接交给兜底 Agent 执行原始任务
            log.warning("编排深度超限（%d/%d），降级为直接执行", depth, self._max_depth)
            result = await self._run_direct(task, parent_ctx.session_key)
            return OrchestratorResult(
                original_task=task,
                subtask_results=[result],
                final_synthesis=result.output,
                total_elapsed_ms=result.elapsed_ms,
            )

        t0 = time.monotonic()
        token = _ORCHESTRATION_DEPTH.set(depth + 1)

        try:
            # 1. 拆解任务
            decompose_provider = self._get_provider("T0")
            specs = await self._decompose(task, decompose_provider)
            specs = specs[: self._max_workers]   # 截断，避免瞬发过多 API call
            log.info("任务拆解完成: 共 %d 个子任务", len(specs))

            # 2+3. 路由 + 并行执行
            results: list[SubtaskResult] = list(
                await asyncio.gather(*[
                    self._run_worker(spec, self._route(spec), parent_ctx.session_key)
                    for spec in specs
                ])
            )

            # 4. 综合
            synth_provider = self._get_provider("T1")
            synthesis = await self._synthesize(task, results, synth_provider)

            return OrchestratorResult(
                original_task=task,
                subtask_results=results,
                final_synthesis=synthesis,
                total_elapsed_ms=(time.monotonic() - t0) * 1000,
            )
        finally:
            _ORCHESTRATION_DEPTH.reset(token)

    # ── 拆解 ──────────────────────────────────────────────────────────────────

    async def _decompose(self, task: str, provider: LLMProvider) -> list[SubtaskSpec]:
        """调用 LLM 把任务拆成子任务列表，解析失败时返回单任务兜底。"""
        try:
            resp = await provider.complete(
                system=_DECOMPOSE_SYSTEM,
                messages=[Message(role="user", content=f"请拆解以下任务：\n\n{task}")],
                max_tokens=512,
            )
            data = _parse_json(resp.final_text)
        except Exception as exc:
            log.warning("任务拆解 LLM 调用失败: %s", exc)
            data = None

        if data is None:
            return [SubtaskSpec(0, task, self._fallback_capability)]

        specs: list[SubtaskSpec] = []
        for i, item in enumerate(data.get("subtasks", [])):
            if isinstance(item, dict) and item.get("description"):
                specs.append(SubtaskSpec(
                    index=i,
                    description=item["description"],
                    required_capability=item.get("required_capability", self._fallback_capability),
                ))

        return specs or [SubtaskSpec(0, task, self._fallback_capability)]

    # ── 路由 ──────────────────────────────────────────────────────────────────

    def _route(self, spec: SubtaskSpec) -> AgentCard:
        """按 required_capability 找 AgentCard，多级兜底保证永不失败。"""
        candidates = self._registry.lookup_by_capability(spec.required_capability)
        if candidates:
            return candidates[0]
        general = self._registry.lookup_by_capability(self._fallback_capability)
        if general:
            return general[0]
        return _FALLBACK_CARD

    # ── 执行 Worker ───────────────────────────────────────────────────────────

    async def _run_worker(
        self,
        spec: SubtaskSpec,
        card: AgentCard,
        parent_session_key: str,
    ) -> SubtaskResult:
        """
        用 AgentCard 的配置 spawn 一个 Worker NanoCore 执行子任务。

        Worker 用独立 session_key（parent_key#worker-{i}-{hex}），
        避免与父 session 的 TurnRunner 锁产生冲突。
        """
        t0 = time.monotonic()
        worker_session = f"{parent_session_key}#worker-{spec.index}-{uuid.uuid4().hex[:6]}"

        provider = self._get_provider(card.default_tier)
        if provider is None:
            return SubtaskResult(
                spec=spec,
                agent_id=card.agent_id,
                output="[错误：未找到对应 provider，检查 provider_factory 配置]",
                success=False,
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        ctx = AgentContext(
            agent_id=card.agent_id,
            session_key=worker_session,
            system_prompt=card.system_prompt,
            model_id=provider.model_id,
        )
        core = NanoCore(
            ctx=ctx,
            provider=provider,
            tools=card.tools,
            tool_definitions=card.tool_definitions,
        )

        output = ""
        success = True
        try:
            async for event in core.run_turn(spec.description):
                if isinstance(event, DoneEvent):
                    output = event.final_text
        except Exception as exc:
            success = False
            output = f"[Worker 错误: {exc}]"
            log.warning("子任务 %d (%s) 失败: %s", spec.index, card.agent_id, exc)

        return SubtaskResult(
            spec=spec,
            agent_id=card.agent_id,
            output=output,
            success=success,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    async def _run_direct(self, task: str, parent_session_key: str) -> SubtaskResult:
        """深度超限时的降级路径：不拆解，直接执行。"""
        spec = SubtaskSpec(0, task, self._fallback_capability)
        card = self._route(spec)
        return await self._run_worker(spec, card, parent_session_key)

    # ── 综合 ──────────────────────────────────────────────────────────────────

    async def _synthesize(
        self,
        original_task: str,
        results: list[SubtaskResult],
        provider: LLMProvider,
    ) -> str:
        """把各子任务结果合并为连贯的最终答复。"""
        parts = [f"原始任务：{original_task}\n"]
        for r in results:
            status = "✓" if r.success else "✗"
            parts.append(f"子任务 {r.spec.index + 1} [{status}] {r.spec.description}\n{r.output}")
        prompt = "\n\n".join(parts)

        try:
            resp = await provider.complete(
                system=_SYNTHESIZE_SYSTEM,
                messages=[Message(role="user", content=prompt)],
                max_tokens=1024,
            )
            return resp.final_text
        except Exception as exc:
            log.warning("综合 LLM 调用失败，降级拼接: %s", exc)
            return "\n\n".join(
                f"[子任务 {r.spec.index + 1}]\n{r.output}" for r in results
            )

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _get_provider(self, tier: str) -> LLMProvider | None:
        """按档位取 provider，找不到时逐级向上兜底。"""
        for t in [tier, "T1", "T0"]:
            p = self._providers.get(t)
            if p is not None:
                return p
        return next(iter(self._providers.values()), None)


# ─── 模块级常量 ───────────────────────────────────────────────────────────────

_FALLBACK_CARD = AgentCard(
    agent_id="__fallback__",
    description="内置通用兜底 Agent，无任何 Agent 注册时使用",
    capabilities=["general"],
    system_prompt="你是一个通用助手，请尽力完成用户交代的任务。",
    default_tier="T1",
)


# ─── JSON 解析工具（与 LLMRouter / SessionConsolidator 保持一致的两步容错策略） ──

def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None

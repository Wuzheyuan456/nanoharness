from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from nanoharness.core.context import AgentContext
from nanoharness.core.event_store import DoneEvent
from nanoharness.core.nano_core import NanoCore
from nanoharness.provider.base import LLMProvider

log = logging.getLogger(__name__)

# ─── 系统提示 ──────────────────────────────────────────────────────────────────

_REVIEWER_SYSTEM = """\
你是一名经验丰富的代码审查工程师。请对提交的代码进行独立审查，重点关注：
- 正确性（逻辑错误、边界条件、潜在 bug）
- 安全性（注入、越界、资源泄漏）
- 可维护性（命名、复杂度、可读性）

只输出 JSON，格式如下：
{
  "issues": ["严重问题或 bug，每条一句话"],
  "suggestions": ["改进建议，每条一句话"],
  "verdict": "approve 或 request_changes 或 reject"
}
"""

_JUDGE_SYSTEM = """\
你是首席代码审查工程师。以下是两位工程师对同一段代码的独立审查意见。
请你：
1. 识别两份意见中存在分歧的具体点
2. 综合判断每个分歧
3. 输出最终合并审查报告

只输出 JSON，格式如下：
{
  "disagreements": ["分歧点描述，每条一句话"],
  "final_verdict": "approve 或 request_changes 或 reject",
  "final_report": "完整的最终审查报告（自然语言，2~5 句话）"
}
"""


# ─── 数据模型 ──────────────────────────────────────────────────────────────────

@dataclass
class ReviewOpinion:
    """一个 Reviewer 的结构化审查意见。"""
    reviewer_label: str                          # "A" 或 "B"
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    verdict: str = "request_changes"             # approve | request_changes | reject
    raw_text: str = ""                           # LLM 原始输出（调试用）


@dataclass
class DebateResult:
    """辩论模式代码审查的最终结果。"""
    code_snippet: str                            # 被审查代码（截断存储）
    reviewer_a: ReviewOpinion
    reviewer_b: ReviewOpinion
    disagreements: list[str]                     # 两人分歧点列表
    final_verdict: str                           # Judge 的最终裁定
    final_report: str                            # Judge 的合并报告
    total_elapsed_ms: float

    @property
    def reviewers_agreed(self) -> bool:
        return self.reviewer_a.verdict == self.reviewer_b.verdict


# ─── DebateOrchestrator ───────────────────────────────────────────────────────

class DebateOrchestrator:
    """
    辩论模式代码审查编排器。

    流程：
    1. 两个 Reviewer NanoCore 用 asyncio.gather 并行独立审查同一段代码
       （完全独立的 session_key 和 AgentContext，不共享任何历史）
    2. Judge NanoCore 收集两份意见，识别分歧，输出最终裁定

    面试话术：
    "辩论模式的价值在于独立视角。如果两个 Reviewer 用同一个 session，
    第二个会受第一个历史的影响，本质上还是一次审查。
    我给他们完全独立的 session_key 和 AgentContext，真正的并行独立评估。
    Judge 的 prompt 明确说'这是两位不同工程师的意见'，
    它的工作是识别分歧，而不是取两者的平均值。
    如果两人都说 approve，Judge 基本会 approve；
    如果一人 approve、一人 reject，Judge 要具体分析分歧在哪。"
    """

    def __init__(
        self,
        provider: LLMProvider,                    # Reviewer 使用的模型
        judge_provider: LLMProvider | None = None, # Judge 可以用更强的模型（默认同 provider）
        reviewer_system: str = _REVIEWER_SYSTEM,
        judge_system: str = _JUDGE_SYSTEM,
    ) -> None:
        self._provider = provider
        self._judge_provider = judge_provider or provider
        self._reviewer_system = reviewer_system
        self._judge_system = judge_system

    async def review(
        self,
        code: str,
        context: str = "",          # 可选背景描述：功能说明、修改目的等
        session_prefix: str = "debate",
    ) -> DebateResult:
        """
        主入口。并行跑两个 Reviewer，再交给 Judge 综合。
        context 注入到 prompt 中帮助 Reviewer 理解代码意图。
        """
        t0 = time.monotonic()
        review_prompt = _build_review_prompt(code, context)

        # 两个 Reviewer 完全并行，各自独立
        opinion_a, opinion_b = await asyncio.gather(
            self._run_reviewer("A", review_prompt, session_prefix),
            self._run_reviewer("B", review_prompt, session_prefix),
        )
        log.info(
            "Reviewer A: %s | Reviewer B: %s | 一致: %s",
            opinion_a.verdict, opinion_b.verdict,
            opinion_a.verdict == opinion_b.verdict,
        )

        # Judge 综合两份意见
        disagreements, verdict, report = await self._judge(
            review_prompt, opinion_a, opinion_b
        )

        return DebateResult(
            code_snippet=code[:500],
            reviewer_a=opinion_a,
            reviewer_b=opinion_b,
            disagreements=disagreements,
            final_verdict=verdict,
            final_report=report,
            total_elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    # ── Reviewer ──────────────────────────────────────────────────────────────

    async def _run_reviewer(
        self,
        label: str,
        review_prompt: str,
        session_prefix: str,
    ) -> ReviewOpinion:
        """启动一个独立 Reviewer NanoCore，解析其 JSON 意见。"""
        # 每个 Reviewer 独立 session，避免共享历史
        session_key = f"{session_prefix}#reviewer-{label}-{uuid.uuid4().hex[:6]}"
        ctx = AgentContext(
            agent_id=f"reviewer-{label}",
            session_key=session_key,
            system_prompt=self._reviewer_system,
            model_id=self._provider.model_id,
        )
        core = NanoCore(ctx=ctx, provider=self._provider, tools={}, tool_definitions=[])

        raw_text = ""
        try:
            async for event in core.run_turn(review_prompt):
                if isinstance(event, DoneEvent):
                    raw_text = event.final_text
        except Exception as exc:
            log.warning("Reviewer %s 异常: %s", label, exc)

        return _parse_opinion(label, raw_text)

    # ── Judge ─────────────────────────────────────────────────────────────────

    async def _judge(
        self,
        review_prompt: str,
        opinion_a: ReviewOpinion,
        opinion_b: ReviewOpinion,
    ) -> tuple[list[str], str, str]:
        """Judge 综合两份意见，返回 (disagreements, final_verdict, final_report)。"""
        judge_prompt = (
            f"被审查的代码：\n{review_prompt}\n\n"
            f"审查者 A 的意见：\n"
            f"  问题：{opinion_a.issues}\n"
            f"  建议：{opinion_a.suggestions}\n"
            f"  裁定：{opinion_a.verdict}\n\n"
            f"审查者 B 的意见：\n"
            f"  问题：{opinion_b.issues}\n"
            f"  建议：{opinion_b.suggestions}\n"
            f"  裁定：{opinion_b.verdict}"
        )

        session_key = f"judge-{uuid.uuid4().hex[:8]}"
        ctx = AgentContext(
            agent_id="judge",
            session_key=session_key,
            system_prompt=self._judge_system,
            model_id=self._judge_provider.model_id,
        )
        core = NanoCore(ctx=ctx, provider=self._judge_provider, tools={}, tool_definitions=[])

        raw_text = ""
        try:
            async for event in core.run_turn(judge_prompt):
                if isinstance(event, DoneEvent):
                    raw_text = event.final_text
        except Exception as exc:
            log.warning("Judge 异常: %s", exc)

        return _parse_judge(raw_text, opinion_a, opinion_b)


# ─── 解析函数（模块级，方便单独测试） ──────────────────────────────────────────

def _parse_opinion(label: str, raw_text: str) -> ReviewOpinion:
    """从 LLM 输出解析 ReviewOpinion，JSON 解析失败时降级到原始文字。"""
    data = _parse_json(raw_text)
    if data:
        return ReviewOpinion(
            reviewer_label=label,
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            verdict=data.get("verdict", "request_changes"),
            raw_text=raw_text,
        )
    # 降级：把原始文字当作唯一 issue
    return ReviewOpinion(
        reviewer_label=label,
        issues=[raw_text[:200]] if raw_text else ["审查失败，无输出"],
        verdict="request_changes",
        raw_text=raw_text,
    )


def _parse_judge(
    raw_text: str,
    opinion_a: ReviewOpinion,
    opinion_b: ReviewOpinion,
) -> tuple[list[str], str, str]:
    """解析 Judge 输出。JSON 解析失败时用规则合并两份意见。"""
    data = _parse_json(raw_text)
    if data:
        return (
            data.get("disagreements", []),
            data.get("final_verdict", "request_changes"),
            data.get("final_report", raw_text[:500]),
        )

    # 降级：规则合并
    # 两人一致直接采用；有任意一人 reject 则 reject；否则 request_changes
    if opinion_a.verdict == opinion_b.verdict:
        merged_verdict = opinion_a.verdict
    elif "reject" in (opinion_a.verdict, opinion_b.verdict):
        merged_verdict = "reject"
    else:
        merged_verdict = "request_changes"

    all_issues = list(dict.fromkeys(opinion_a.issues + opinion_b.issues))   # dedup 保序
    report = (
        f"共发现 {len(all_issues)} 个问题（A: {len(opinion_a.issues)} 条，"
        f"B: {len(opinion_b.issues)} 条）。最终裁定：{merged_verdict}。"
    )
    return ([], merged_verdict, report)


def _build_review_prompt(code: str, context: str) -> str:
    parts = ["请审查以下代码："]
    if context:
        parts.append(f"背景说明：{context}")
    parts.append(f"```\n{code}\n```")
    return "\n\n".join(parts)


def _parse_json(text: str) -> dict | None:
    """两步容错解析，保持项目内一致的 JSON 解析策略。"""
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

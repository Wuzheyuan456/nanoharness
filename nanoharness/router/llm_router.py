from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from nanoharness.core.context import Message
from nanoharness.router.decision_log import DecisionLog, RouterDecision
from nanoharness.router.tiers import DEFAULT_TIER_CONFIGS, Tier, TierRegistry

log = logging.getLogger(__name__)

# ─── 分类结果 ──────────────────────────────────────────────────────────────────

@dataclass
class ClassifyResult:
    tier: Tier
    confidence: float    # 0.0 ~ 1.0
    reason: str
    method: str          # "llm" | "heuristic" | "fallback"
    latency_ms: float = 0.0


# ─── 规则启发式降级 ────────────────────────────────────────────────────────────

# 关键词 → 档位，从高到低匹配，第一个命中的优先
_HEURISTIC_RULES: list[tuple[list[str], Tier]] = [
    # T3 特征：风险、决策、大型代码重构、需要深度推理
    (["重构", "架构设计", "安全漏洞", "重大决策", "生产事故", "refactor", "architecture"], Tier.T3),
    # T2 特征：代码生成、多步骤分析、长文档
    (["写代码", "实现", "生成", "分析", "比较", "code", "implement", "analyze"], Tier.T2),
    # T0 特征：打招呼、简单查询、是/否问题
    (["你好", "hello", "hi", "谢谢", "thanks", "几点", "今天", "天气", "是不是", "对吗"], Tier.T0),
]


def heuristic_classify(message: str) -> ClassifyResult:
    """
    基于关键词的快速规则分类，作为 LLM 分类的降级兜底。
    """
    msg_lower = message.lower()
    for keywords, tier in _HEURISTIC_RULES:
        if any(k in msg_lower for k in keywords):
            return ClassifyResult(
                tier=tier,
                confidence=0.6,
                reason=f"规则匹配关键词 → {tier}",
                method="heuristic",
            )
    return ClassifyResult(
        tier=Tier.T1,
        confidence=0.5,
        reason="无规则命中，默认 T1",
        method="fallback",
    )


# ─── LLM 路由器 ────────────────────────────────────────────────────────────────

# 发给分类 LLM 的 system prompt，要求严格输出 JSON
_CLASSIFIER_SYSTEM = """\
你是一个任务难度分类器。根据用户消息，判断处理该任务需要的模型档位，输出 JSON。

档位定义：
- T0：简单问答、打招呼、单步查询、是/否判断，无需工具调用
- T1：中等推理、需要 1~3 步工具调用、常规代码问题
- T2：复杂任务、长链推理、代码生成/重构、多轮工具调用
- T3：高风险决策、架构设计、安全分析、需要深度思考

严格只输出如下 JSON，不要有任何其他文字：
{"tier": "T0|T1|T2|T3", "confidence": 0.0~1.0, "reason": "一句话理由"}
"""


class LLMRouter:
    """
    基于一次 T0 级别 LLM call 的任务难度分类器。

    降级链（按优先级）：
    1. LLM 分类（claude-haiku，~200ms，成本极低）
    2. 超时 → 规则启发式（关键词匹配，<1ms）
    3. 解析失败 → fallback T1

    面试话术：
    "我没有用 opensquilla 那种本地 ONNX 模型，原因是 BGE 模型需要 Git LFS
    依赖，部署门槛高。我的方案是用 Haiku 做一次分类调用，成本约 $0.00005，
    延迟约 200ms，换来零依赖、可解释的分类理由。这是一个刻意的工程取舍。"
    """

    def __init__(
        self,
        provider: object,                    # LLMProvider，用 T0 模型
        registry: TierRegistry | None = None,
        decision_log: DecisionLog | None = None,
        timeout: float = 2.0,                # 超时后降级到 heuristic
    ) -> None:
        self._provider = provider
        self._registry = registry or TierRegistry()
        self._log = decision_log
        self._timeout = timeout

    async def classify(
        self,
        message: str,
        trace_id: str = "",
        session_key: str = "",
    ) -> ClassifyResult:
        """
        输入用户消息，返回 ClassifyResult。
        无论 LLM 是否成功，都保证返回一个有效结果。
        """
        t0 = time.monotonic()
        result = await self._try_llm_classify(message)
        result.latency_ms = (time.monotonic() - t0) * 1000

        if self._log:
            self._log.append(RouterDecision(
                trace_id=trace_id,
                session_key=session_key,
                input_preview=message[:100],
                tier=result.tier,
                confidence=result.confidence,
                reason=result.reason,
                model_used=self._registry.model_id(Tier.T0),
                method=result.method,
                latency_ms=result.latency_ms,
            ))

        log.debug("路由分类: tier=%s confidence=%.2f method=%s latency=%.0fms",
                  result.tier, result.confidence, result.method, result.latency_ms)
        return result

    async def _try_llm_classify(self, message: str) -> ClassifyResult:
        try:
            resp = await asyncio.wait_for(
                self._provider.complete(   # type: ignore[attr-defined]
                    system=_CLASSIFIER_SYSTEM,
                    messages=[Message(role="user", content=message)],
                    max_tokens=128,
                ),
                timeout=self._timeout,
            )
            return self._parse_response(resp.final_text)
        except asyncio.TimeoutError:
            log.warning("路由 LLM 超时（%.1fs），降级到规则分类", self._timeout)
            return heuristic_classify(message)
        except Exception as exc:
            log.warning("路由 LLM 异常（%s），降级到规则分类", exc)
            return heuristic_classify(message)

    @staticmethod
    def _parse_response(text: str) -> ClassifyResult:
        """
        解析 LLM 返回的 JSON。

        容错策略：
        - 先 json.loads 整段文本
        - 失败则用正则提取第一个 {...} 块
        - 再失败则 fallback T1（method="fallback"）
        """
        def _extract(raw: str) -> dict | None:
            try:
                return json.loads(raw.strip())
            except json.JSONDecodeError:
                pass
            m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return None  # 明确返回 None 表示解析完全失败

        data = _extract(text)
        if data is None:
            log.debug("路由响应解析失败，原文: %s", text[:100])
            return ClassifyResult(
                tier=Tier.T1, confidence=0.5,
                reason="LLM 响应解析失败，默认 T1", method="fallback",
            )
        try:
            tier = Tier(data.get("tier", "T1"))
            confidence = float(data.get("confidence", 0.7))
            reason = str(data.get("reason", ""))
            return ClassifyResult(tier=tier, confidence=confidence, reason=reason, method="llm")
        except (ValueError, TypeError):
            return ClassifyResult(
                tier=Tier.T1, confidence=0.5,
                reason="tier 字段无效，默认 T1", method="fallback",
            )

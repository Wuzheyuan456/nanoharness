from __future__ import annotations

import time
from dataclasses import dataclass

from nanoharness.router.tiers import Tier

# ─── 路由策略配置 / Routing policy config ────────────────────────────────────

_TIER_ORDER = [Tier.T0, Tier.T1, Tier.T2, Tier.T3]


@dataclass
class RoutingPolicy:
    """
    分类结果的后处理策略。 / Post-processing policy applied to raw classification results.

    面试话术 / Interview talking point:
    "分类器给了 tier 和 confidence，但直接用置信度 0.4 的结果等于没有 confidence 概念。
    RoutingPolicy 是分类之后的决策层：置信度不够就升档，KV cache 窗口内不降档——
    这对应 opensquilla RoutingPolicyEngine 的前两个 stage，
    我只取通用 harness 需要的部分。"
    """
    confidence_threshold: float = 0.6    # 低于此置信度 → 升一档 / below this → escalate by one tier
    anti_downgrade_window_s: float = 1800.0  # 30 分钟内不降档（KV cache 保护） / no downgrade within 30 min (KV cache protection)


# ─── 策略函数 / Policy functions ────────────────────────────────────────────

def _confidence_gate(
    result: "ClassifyResult",
    policy: RoutingPolicy,
) -> "ClassifyResult":
    """
    置信度低于阈值时升一档。 / Escalate by one tier when confidence is below the threshold.

    只对 LLM 分类结果生效（规则/fallback 本身就是保守估计，不再升档）。 /
    Only applies to LLM results — heuristic/fallback are already conservative, no escalation needed.
    """
    from nanoharness.router.llm_router import ClassifyResult
    if "llm" not in result.method:
        return result  # 规则或 fallback 不升档 / rule/fallback: no escalation
    if result.confidence >= policy.confidence_threshold:
        return result
    idx = _TIER_ORDER.index(result.tier)
    escalated = _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]
    return ClassifyResult(
        tier=escalated,
        confidence=result.confidence,
        reason=result.reason,
        method=result.method + "+confidence_escalated",
        latency_ms=result.latency_ms,
    )


def _anti_downgrade(
    result: "ClassifyResult",
    policy: RoutingPolicy,
    session_key: str,
    cache: dict[str, tuple[Tier, float]],
) -> "ClassifyResult":
    """
    保护 KV cache：窗口内不降档。 / KV cache protection: no downgrade within the window.

    上轮路由到 T2，本轮分类到 T0 → 保持 T2。超出时间窗口则放行。 /
    If the last turn was T2 and this one classifies as T0 → hold at T2. Allow if window expired.
    """
    from nanoharness.router.llm_router import ClassifyResult
    if not session_key or session_key not in cache:
        return result
    prev_tier, ts = cache[session_key]
    if time.monotonic() - ts > policy.anti_downgrade_window_s:
        return result  # 缓存过期，放行 / cache expired, allow
    prev_idx = _TIER_ORDER.index(prev_tier)
    cur_idx = _TIER_ORDER.index(result.tier)
    if cur_idx >= prev_idx:
        return result  # 没有降档，放行 / no downgrade
    return ClassifyResult(
        tier=prev_tier,
        confidence=result.confidence,
        reason=result.reason,
        method=result.method + "+anti_downgrade",
        latency_ms=result.latency_ms,
    )


def _update_cache(session_key: str, tier: Tier, cache: dict[str, tuple[Tier, float]]) -> None:
    """更新会话 tier 缓存，超过 1000 条时清理最旧的 200 条。 / Update session tier cache; evict oldest 200 entries when over 1000."""
    cache[session_key] = (tier, time.monotonic())
    if len(cache) > 1000:
        oldest = sorted(cache.items(), key=lambda kv: kv[1][1])[:200]
        for k, _ in oldest:
            del cache[k]


def apply_routing_policy(
    result: "ClassifyResult",
    policy: RoutingPolicy,
    session_key: str,
    session_tier_cache: dict[str, tuple[Tier, float]],
) -> "ClassifyResult":
    """
    按顺序应用所有路由策略阶段并更新缓存。 / Apply all routing policy stages in order and update the cache.

    阶段顺序 / Stage order:
    1. confidence_gate  — 低置信度升档 / escalate on low confidence
    2. anti_downgrade   — KV cache 保护 / KV cache protection
    3. update_cache     — 记录本轮最终 tier / record final tier for this turn
    """
    result = _confidence_gate(result, policy)
    result = _anti_downgrade(result, policy, session_key, session_tier_cache)
    _update_cache(session_key, result.tier, session_tier_cache)
    return result

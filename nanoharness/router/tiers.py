from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


# ─── 难度档位枚举 / Difficulty tier enum ──────────────────────────────────────────────────────────────

class Tier(StrEnum):
    T0 = "T0"   # 简单问答：打招呼、查事实、单步计算 / simple QA: greeting, fact lookup, single-step calc
    T1 = "T1"   # 中等推理：多步骤任务、需要 1~3 次工具调用 / medium reasoning: multi-step tasks, 1~3 tool calls
    T2 = "T2"   # 复杂任务：长链推理、代码生成、多轮工具调用 / complex tasks: long-chain reasoning, codegen, multi-turn tool calls
    T3 = "T3"   # 高风险决策：需要 extended thinking、高精度要求 / high-stakes decisions: needs extended thinking, high accuracy


# ─── 每个档位的模型配置 / Per-tier model config ────────────────────────────────────────────────────────

@dataclass
class TierConfig:
    tier: Tier
    model_id: str
    max_tokens: int
    thinking_budget: int = 0   # >0 时启用 extended thinking（仅 T3） / enable extended thinking when >0 (T3 only)
    description: str = ""


# 默认映射表 — 可通过 nanoharness.toml 覆盖 / default mapping table — overridable via nanoharness.toml
DEFAULT_TIER_CONFIGS: dict[Tier, TierConfig] = {
    Tier.T0: TierConfig(
        tier=Tier.T0,
        model_id="claude-sonnet-4-6",
        max_tokens=1024,
        description="简单问答，最低成本",
    ),
    Tier.T1: TierConfig(
        tier=Tier.T1,
        model_id="claude-sonnet-4-6",
        max_tokens=4096,
        description="中等推理，默认档位",
    ),
    Tier.T2: TierConfig(
        tier=Tier.T2,
        model_id="claude-sonnet-4-6",
        max_tokens=8192,
        description="复杂任务，长链推理",
    ),
    Tier.T3: TierConfig(
        tier=Tier.T3,
        model_id="claude-opus-4-7",
        max_tokens=16384,
        thinking_budget=10000,
        description="高风险决策，extended thinking",
    ),
}


# ─── Prompt 策略 / Prompt policy ───────────────────────────────────────────────────────────────

class PromptPolicy(StrEnum):
    P0 = "P0"   # 直接回答，不加推理引导 / answer directly, no reasoning guidance
    P1 = "P1"   # 加"请分步骤思考"引导 / add "think step by step" guidance
    P2 = "P2"   # 加"请先列出所有可能方案再选最优"引导 / add "list all options then pick the best" guidance


# 档位 → 推荐 Prompt 策略 / tier → recommended prompt policy
TIER_TO_POLICY: dict[Tier, PromptPolicy] = {
    Tier.T0: PromptPolicy.P0,
    Tier.T1: PromptPolicy.P1,
    Tier.T2: PromptPolicy.P1,
    Tier.T3: PromptPolicy.P2,
}

POLICY_HINTS: dict[PromptPolicy, str] = {
    PromptPolicy.P0: "",
    PromptPolicy.P1: "请分步骤思考后再给出答案。",
    PromptPolicy.P2: "请先列出所有可能的方案，评估各自优缺点，再选择最优方案详细执行。",
}


# ─── TierRegistry：支持运行时覆盖 / TierRegistry: supports runtime override ─────────────────────────────────────────────

class TierRegistry:
    """
    持有当前生效的档位配置，支持从 toml/env 覆盖默认值。 / Holds currently effective tier configs, supports overriding defaults via toml/env.

    面试话术 / Interview talking point:
    "模型映射和引擎逻辑完全分离。运营同学可以改 nanoharness.toml
    里的 model_id，不需要动任何 Python 代码。"
    """

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        self._configs = {k: v for k, v in DEFAULT_TIER_CONFIGS.items()}
        if overrides:
            self._apply_overrides(overrides)

    def get(self, tier: Tier) -> TierConfig:
        return self._configs[tier]

    def model_id(self, tier: Tier) -> str:
        return self._configs[tier].model_id

    def policy_hint(self, tier: Tier) -> str:
        policy = TIER_TO_POLICY[tier]
        return POLICY_HINTS[policy]

    def _apply_overrides(self, overrides: dict[str, Any]) -> None:
        # overrides 格式：{"T0": {"model_id": "..."}, "T1": {...}} / overrides format: {"T0": {"model_id": "..."}, "T1": {...}}
        for tier_str, fields in overrides.items():
            try:
                tier = Tier(tier_str)
                cfg = self._configs[tier]
                for k, v in fields.items():
                    if hasattr(cfg, k):
                        object.__setattr__(cfg, k, v)
            except (ValueError, KeyError):
                pass  # 未知档位直接忽略，不影响运行 / unknown tiers are silently ignored, no runtime impact


# 全局默认实例（可被替换） / global default instance (replaceable)
default_registry = TierRegistry()

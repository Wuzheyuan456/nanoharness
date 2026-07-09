from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from nanoharness.core.context import AgentContext, Message

log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class CompactionConfig:
    """
    Decoupled from AgentContext so compaction can use a cheaper model
    (e.g. claude-haiku-4-5) without affecting the main turn model.
    """
    compaction_model: str = "claude-haiku-4-5-20251001"
    context_window_limit: int = 180_000       # tokens before compaction triggers
    keep_recent_messages: int = 10            # always preserve last N messages
    keep_budget_tokens: int = 20_000          # target remaining tokens after compact
    safety_margin: float = 0.10              # extra buffer on top of keep_budget


# ─── Turn-boundary helpers ────────────────────────────────────────────────────

def find_turn_boundary_cut(messages: list[Message], keep_budget_tokens: int) -> int:
    """
    Walk backwards through messages, accumulating token cost.
    Return the index of the first message to KEEP (everything before is compacted).
    """
    accumulated = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        cost = msg.token_count if msg.token_count > 0 else _estimate_tokens(msg)
        accumulated += cost
        if accumulated > keep_budget_tokens:
            return i + 1
    return 0  # keep everything (shouldn't happen if caller checks budget)


def retreat_to_turn_boundary(messages: list[Message], cut: int) -> int:
    """
    Adjust cut point so we never split a tool_use / tool_result pair.

    Rule: if messages[cut] is a tool_result, or if messages[cut-1] is a
    tool_use, back up one more step.
    """
    while cut < len(messages):
        if messages[cut].is_tool_result():
            cut -= 1
            continue
        if cut > 0 and messages[cut - 1].is_tool_use():
            cut -= 1
            continue
        break
    return max(0, cut)


# ─── Semantic importance scoring ──────────────────────────────────────────────

def semantic_importance_score(msg: Message, index: int, total: int) -> float:
    """
    Heuristic importance score [0, 1].

    Weights:
    - Tool results get 0.8 base  (concrete facts the model acted on)
    - User messages get 0.6 base (original intent)
    - Assistant text gets 0.4 base (reasoning, can be re-derived)
    - Position decay: messages closer to the end score higher
    - Short messages (< 50 chars) get a 0.5× penalty (likely filler)
    """
    base = 0.4
    if msg.is_tool_result():
        base = 0.8
    elif msg.role == "user":
        base = 0.6

    # Recency boost: linear from 0 (oldest) to 1 (newest)
    recency = (index + 1) / total if total > 1 else 1.0

    content_len = len(msg.content) if isinstance(msg.content, str) else len(json.dumps(msg.content))
    brevity_penalty = 0.5 if content_len < 50 else 1.0

    return base * (0.5 + 0.5 * recency) * brevity_penalty


# ─── Compaction Engine ────────────────────────────────────────────────────────

class CompactionEngine:
    """
    Compress ctx.history when approaching the context window limit.

    Strategy:
    1. find_turn_boundary_cut() — identify what to cut
    2. retreat_to_turn_boundary() — ensure no orphaned tool pairs
    3. Score remaining candidates by semantic_importance_score()
    4. Call a cheap LLM (compaction_model) to summarize the cut portion
    5. Prepend the summary as a system-style user message

    Interview talking point:
    "Instead of naive truncation I score each message's semantic contribution
    — tool results carry the most information density, so they survive longest.
    The turn-boundary retreat prevents malformed API payloads."
    """

    def __init__(self, provider: Any, config: CompactionConfig | None = None) -> None:
        self._provider = provider
        self.config = config or CompactionConfig()
        # Expose to NanoCore._should_compact()
        self.context_window_limit = self.config.context_window_limit

    async def compact(self, ctx: AgentContext) -> None:
        messages = ctx.history
        if len(messages) <= self.config.keep_recent_messages:
            return

        keep_budget = self.config.keep_budget_tokens
        cut = find_turn_boundary_cut(messages, keep_budget)
        cut = retreat_to_turn_boundary(messages, cut)

        if cut == 0:
            log.warning("compact: could not find safe cut point, skipping")
            return

        to_compress = messages[:cut]
        to_keep = messages[cut:]

        # Sort by importance to help the summariser focus on high-value content
        scored = sorted(
            enumerate(to_compress),
            key=lambda t: semantic_importance_score(t[1], t[0], len(to_compress)),
            reverse=True,
        )
        # Pass top-60% to the summariser (avoid flooding cheap model)
        top_n = max(1, int(len(scored) * 0.6))
        top_messages = [m for _, m in scored[:top_n]]

        summary_text = await self._summarise(top_messages, ctx.system_prompt)

        summary_message = Message(
            role="user",
            content=f"[CONTEXT SUMMARY — earlier conversation compressed]\n{summary_text}",
        )
        ctx.history = [summary_message] + to_keep
        log.info(
            "compact: %d messages → summary + %d kept (cut=%d)",
            len(to_compress), len(to_keep), cut,
        )

    async def _summarise(self, messages: list[Message], system_hint: str) -> str:
        prompt = (
            "Summarize the following conversation excerpt concisely.\n"
            "Preserve: key facts discovered, decisions made, tool results, "
            "and any important context needed for the ongoing task.\n"
            "Be brief but accurate.\n\n"
            "Conversation:\n" + self._format_for_summary(messages)
        )
        try:
            resp = await self._provider.complete(
                system=f"You are a concise summariser. Context: {system_hint[:200]}",
                messages=[Message(role="user", content=prompt)],
                max_tokens=1024,
            )
            return resp.final_text
        except Exception as exc:
            log.error("compact summarise failed: %s", exc)
            return "[Summary unavailable — compaction model error]"

    @staticmethod
    def _format_for_summary(messages: list[Message]) -> str:
        lines = []
        for m in messages:
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)[:500]
            lines.append(f"{m.role.upper()}: {content}")
        return "\n".join(lines)


# ─── Token estimate helper ────────────────────────────────────────────────────

def _estimate_tokens(msg: Message) -> int:
    if isinstance(msg.content, str):
        return len(msg.content) // 4
    return len(json.dumps(msg.content)) // 4

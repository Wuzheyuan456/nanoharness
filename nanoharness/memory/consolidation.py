from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from nanoharness.core.context import AgentContext, Message
from nanoharness.memory.store import MemoryEntry, MemoryStore, MemoryType, SessionRecord

log = logging.getLogger(__name__)

# LLM 提炼 session 摘要用的 prompt / prompt for LLM to distill session summary
_CONSOLIDATION_SYSTEM = """\
你是一个记忆整理助手。请从以下对话中提取：
1. 一段简洁的会话摘要（2~4句话，描述做了什么、得出了什么结论）
2. 值得长期记住的事实或用户偏好（每条一行，格式：[类型] 内容）
   类型可以是：fact / preference

只输出 JSON，格式如下：
{
  "summary": "...",
  "facts": [
    {"type": "fact", "content": "..."},
    {"type": "preference", "content": "..."}
  ]
}
"""


class SessionConsolidator:
    """
    会话结束后异步触发的记忆巩固（简化版 Dream 机制）。 / Async memory consolidation triggered after a session ends (simplified Dream mechanism).

    流程： / Flow:
    1. 用 LLM（建议 Haiku，成本低）提炼整个 session 的摘要和关键事实 / 1. Use LLM (Haiku recommended, low cost) to distill the whole session's summary and key facts
    2. 摘要写入 L2（sessions 表） / 2. Write summary into L2 (sessions table)
    3. 事实/偏好写入 L3（memories 表） / 3. Write facts/preferences into L3 (memories table)
    4. 全程异步后台运行，不阻塞用户响应 / 4. Runs fully async in the background, never blocks user responses

    面试话术 / Interview talking point:
    "Dream 机制来自神经科学——大脑在睡眠时巩固白天的记忆。
    我的简化版是 session 结束后异步触发一次 LLM call，
    把对话提炼成结构化摘要写入 SQLite。
    下次 session 开始时，FTS5 能检索到这段摘要，
    用户不需要重复介绍自己。"
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: Any,                   # LLMProvider，建议用 Haiku / LLMProvider, Haiku recommended
        max_history_chars: int = 6000,   # 传给 LLM 的对话上限（防超长） / conversation char cap sent to LLM (guard against overflow)
    ) -> None:
        self._store = store
        self._provider = provider
        self._max_chars = max_history_chars

    async def consolidate(self, ctx: AgentContext, started_at: float) -> None:
        """
        同步入口（内部用 asyncio.create_task 转为后台）。 / Sync entry (internally spawned to background via asyncio.create_task).
        由 MemoryManager.flush() 调用。 / Called by MemoryManager.flush().
        """
        asyncio.create_task(self._run(ctx, started_at))

    async def consolidate_and_wait(self, ctx: AgentContext, started_at: float) -> None:
        """同步等待版本，用于测试或需要确认写入完成的场景。 / Sync-await version, for tests or when write completion must be confirmed."""
        await self._run(ctx, started_at)

    async def _run(self, ctx: AgentContext, started_at: float) -> None:
        if not ctx.history:
            return

        history_text = self._format_history(ctx.history)
        try:
            result = await self._llm_consolidate(history_text, ctx.system_prompt)
        except Exception as exc:
            log.warning("记忆巩固 LLM 调用失败，跳过: %s", exc)
            # 降级：直接存截断的对话作为摘要 / fallback: store truncated conversation as summary
            result = {
                "summary": history_text[:300] + "...",
                "facts": [],
            }

        # 写入 L2：会话摘要 / write L2: session summary
        self._store.upsert_session(SessionRecord(
            session_key=ctx.session_key,
            agent_id=ctx.agent_id,
            summary=result.get("summary", ""),
            started_at=started_at,
            ended_at=time.time(),
        ))

        # 写入 L3：事实和偏好 / write L3: facts and preferences
        for item in result.get("facts", []):
            content = item.get("content", "").strip()
            if not content:
                continue
            raw_type = item.get("type", "fact")
            mtype = MemoryType.PREFERENCE if raw_type == "preference" else MemoryType.FACT
            self._store.upsert(MemoryEntry(
                content=content,
                agent_id=ctx.agent_id,
                session_key=ctx.session_key,
                memory_type=mtype,
                importance=0.75 if mtype == MemoryType.PREFERENCE else 0.65,
            ))

        log.info(
            "记忆巩固完成: session=%s 摘要=%d字 事实=%d条",
            ctx.session_key,
            len(result.get("summary", "")),
            len(result.get("facts", [])),
        )

    async def _llm_consolidate(self, history_text: str, system_hint: str) -> dict:
        import json, re

        prompt = f"请整理以下对话：\n\n{history_text}"
        resp = await self._provider.complete(
            system=_CONSOLIDATION_SYSTEM,
            messages=[Message(role="user", content=prompt)],
            max_tokens=512,
        )
        text = resp.final_text.strip()

        # 容错解析（和 LLMRouter 保持一致的策略） / fault-tolerant parsing (same strategy as LLMRouter)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {"summary": text[:300], "facts": []}

    def _format_history(self, messages: list[Message]) -> str:
        """把对话历史转成文字，截断到 max_history_chars。 / Convert conversation history to text, truncated to max_history_chars."""
        import json
        lines: list[str] = []
        total = 0
        for msg in messages:
            if isinstance(msg.content, str):
                line = f"{msg.role.upper()}: {msg.content}"
            else:
                line = f"{msg.role.upper()}: {json.dumps(msg.content, ensure_ascii=False)[:300]}"
            total += len(line)
            if total > self._max_chars:
                lines.append("... （对话过长，已截断）")
                break
            lines.append(line)
        return "\n".join(lines)

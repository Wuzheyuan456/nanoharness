"""
DuckDuckGo 网络搜索工具，无需 API key / DuckDuckGo web search tool, no API key required.

使用 DuckDuckGo Instant Answer API（免费，JSON 格式） /
Uses DuckDuckGo Instant Answer API (free, JSON format).
对于知识性问题能返回摘要；实时性问题（股票、天气等）返回相关主题 /
Returns summaries for knowledge questions; for real-time queries returns related topics.
"""
from __future__ import annotations

from typing import Any

import httpx

from nanoharness.core.tool_executor import ToolResult, ToolResultStatus

WEB_SEARCH_DEF = {
    "name": "web_search",
    "description": (
        "使用 DuckDuckGo 搜索网络信息，返回摘要和相关结果。"
        "适用于需要实时信息、事实核实的场景。英文关键词搜索效果更好。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题，英文搜索效果通常更好",
            }
        },
        "required": ["query"],
    },
}

_DDG_API = "https://api.duckduckgo.com/"
_HEADERS = {"User-Agent": "NanoHarness/0.1 (+https://github.com/Wuzheyuan456/nanoharness)"}


async def web_search_fn(tool_input: dict[str, Any], _ctx: Any) -> ToolResult:
    query = (tool_input.get("query") or "").strip()
    if not query:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content="搜索词为空",
            error_code="empty_query",
        )

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(
                _DDG_API,
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                headers=_HEADERS,
            )
        data: dict[str, Any] = resp.json()
    except Exception as exc:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content=f"搜索请求失败: {exc}",
            error_code="network_error",
            next_action_hint="网络不可用时，请用已有知识回答用户。",
        )

    lines: list[str] = []

    # 直接答案（计算结果、数字等）/ Direct answer (calculations, facts)
    answer = (data.get("Answer") or "").strip()
    if answer:
        lines.append(f"直接答案: {answer}")

    # Wikipedia 等权威摘要 / Authoritative abstract (Wikipedia, etc.)
    abstract = (data.get("AbstractText") or "").strip()
    if abstract:
        lines.append(f"摘要: {abstract[:600]}")
        src = (data.get("AbstractSource") or "").strip()
        url = (data.get("AbstractURL") or "").strip()
        if src:
            lines.append(f"来源: {src}" + (f" — {url}" if url else ""))

    # 相关主题（取前 4 条）/ Related topics (top 4)
    topics = [t for t in data.get("RelatedTopics", []) if isinstance(t, dict) and t.get("Text")]
    for t in topics[:4]:
        text = (t.get("Text") or "").strip()
        if text:
            lines.append(f"• {text[:250]}")

    if not lines:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content=(
                f"DuckDuckGo 未找到 '{query}' 的直接结果。\n"
                "建议：换用更具体的英文关键词，或用已有知识回答。"
            ),
            error_code="no_results",
            next_action_hint="请换个搜索词重试，或用已有知识直接回答。",
        )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        content="\n".join(lines),
        next_action_hint="基于以上搜索结果回答用户问题，必要时说明信息来源。",
    )

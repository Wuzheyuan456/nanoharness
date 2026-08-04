"""
NanoHarness 内置工具集 / NanoHarness built-in tool suite.

开箱即用，无需配置 API key / Ready to use, no API key required.
  - calculator    : 安全的数学表达式求值（AST 白名单）
  - current_datetime: 当前日期时间（支持时区）
  - web_search    : DuckDuckGo 搜索（无需 API key）
"""
from __future__ import annotations

from nanoharness.tools.calculator import CALCULATOR_DEF, calculator_fn
from nanoharness.tools.datetime_tool import DATETIME_DEF, datetime_fn
from nanoharness.tools.web_search import WEB_SEARCH_DEF, web_search_fn


def get_builtin_tools() -> tuple[dict, list[dict]]:
    """
    返回 (tools_dict, tool_definitions)，直接传给 NanoCore。
    tools_dict: {name: callable}，callable 签名 (input_dict, ToolContext) -> ToolResult
    tool_definitions: Anthropic API 格式的工具描述列表
    """
    tools = {
        "calculator": calculator_fn,
        "current_datetime": datetime_fn,
        "web_search": web_search_fn,
    }
    definitions = [CALCULATOR_DEF, DATETIME_DEF, WEB_SEARCH_DEF]
    return tools, definitions


__all__ = ["get_builtin_tools", "CALCULATOR_DEF", "DATETIME_DEF", "WEB_SEARCH_DEF"]

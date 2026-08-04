"""
当前日期时间工具，支持时区 / Current date-time tool with timezone support.
"""
from __future__ import annotations

import zoneinfo
from datetime import datetime
from typing import Any

from nanoharness.core.tool_executor import ToolResult, ToolResultStatus

DATETIME_DEF = {
    "name": "current_datetime",
    "description": (
        "获取当前日期和时间。默认返回上海时间（Asia/Shanghai）。"
        "可指定 IANA 时区，如 UTC、America/New_York、Europe/London。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA 时区名，默认 Asia/Shanghai。示例：UTC、America/New_York",
            }
        },
        "required": [],
    },
}

_WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]


def datetime_fn(tool_input: dict[str, Any], _ctx: Any) -> ToolResult:
    tz_name = (tool_input.get("timezone") or "Asia/Shanghai").strip()
    try:
        tz = zoneinfo.ZoneInfo(tz_name)
    except zoneinfo.ZoneInfoNotFoundError:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content=f"未知时区: {tz_name}。请使用标准 IANA 名，如 Asia/Shanghai、UTC、America/New_York",
            error_code="unknown_timezone",
            next_action_hint="请改用 UTC 或 Asia/Shanghai。",
        )

    now = datetime.now(tz)
    weekday = _WEEKDAYS[now.weekday()]
    offset = now.strftime("%z")
    formatted_offset = f"UTC{offset[:3]}:{offset[3:]}" if len(offset) == 5 else "UTC"
    result = (
        f"当前时间（{tz_name}）: {now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"星期{weekday}，{formatted_offset}"
    )
    return ToolResult(status=ToolResultStatus.SUCCESS, content=result)

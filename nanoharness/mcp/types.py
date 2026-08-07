"""MCP server configuration and tool descriptor types. / MCP 服务器配置与工具描述类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class McpStdioConfig:
    """标准输入/输出传输的 MCP 服务器配置 / MCP server config using stdio transport."""
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None


@dataclass
class McpSseConfig:
    """SSE/HTTP 传输的 MCP 服务器配置 / MCP server config using SSE (HTTP) transport."""
    url: str
    headers: dict[str, str] = field(default_factory=dict)


# Union type — either stdio or SSE
McpServerConfig = McpStdioConfig | McpSseConfig


@dataclass
class McpToolInfo:
    """从 MCP 服务器发现的工具元数据 / Tool metadata discovered from an MCP server."""
    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Return namespaced name: ``{server}__{name}``."""
        return f"{self.server}__{self.name}"

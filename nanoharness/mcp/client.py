"""MCP client: connects to servers, discovers tools, wraps them as NanoHarness callables.
MCP 客户端：连接服务器、发现工具、封装为 NanoHarness 可调用。

Usage / 使用方式:
    config = McpConfig.load_for_nano()
    async with McpClient(config) as client:
        tools_dict, tool_defs = client.get_tools()
        # merge into NanoCore's tool dict + definition list
"""
from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from nanoharness.core.tool_executor import ToolResult, ToolResultStatus
from nanoharness.mcp.config import McpConfig
from nanoharness.mcp.types import McpSseConfig, McpStdioConfig, McpToolInfo

log = logging.getLogger(__name__)


class McpClient:
    """
    异步上下文管理器，管理全部 MCP 服务器连接 / Async context manager managing all MCP server connections.

    每个服务器的连接失败不影响其他服务器 / Per-server failures are non-fatal; other servers continue to function.
    """

    def __init__(self, config: McpConfig) -> None:
        self._config = config
        self._sessions: dict[str, Any] = {}   # server_name -> mcp.ClientSession
        self._tool_infos: list[McpToolInfo] = []
        self._stack = AsyncExitStack()

    # ── async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "McpClient":
        await self._stack.__aenter__()
        if not self._config.is_empty():
            await self._connect_all()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._stack.__aexit__(*args)

    # ── internal ──────────────────────────────────────────────────────────────

    async def _connect_all(self) -> None:
        try:
            from mcp import ClientSession                              # type: ignore[import]
            from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore[import]
        except ImportError:
            log.warning(
                "mcp package not installed — MCP servers unavailable. "
                "Run: pip install 'nanoharness[mcp]'  or  pip install mcp"
            )
            return

        for name, cfg in self._config.servers.items():
            try:
                session = await self._open_session(name, cfg, ClientSession, stdio_client, StdioServerParameters)
                self._sessions[name] = session
                result = await session.list_tools()
                for t in result.tools:
                    schema = t.inputSchema
                    if hasattr(schema, "model_dump"):
                        schema = schema.model_dump()
                    self._tool_infos.append(McpToolInfo(
                        server=name,
                        name=t.name,
                        description=t.description or "",
                        input_schema=schema or {},
                    ))
                log.info("MCP '%s' connected: %d tools", name, len(result.tools))
            except Exception as exc:
                log.warning("MCP server '%s' failed to connect: %s", name, exc)

    async def _open_session(
        self,
        name: str,
        cfg: McpStdioConfig | McpSseConfig,
        ClientSession: Any,
        stdio_client: Any,
        StdioServerParameters: Any,
    ) -> Any:
        if isinstance(cfg, McpStdioConfig):
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env=cfg.env or {},
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            # SSE / HTTP transport
            try:
                from mcp.client.sse import sse_client  # type: ignore[import]
            except ImportError:
                raise RuntimeError(
                    f"SSE transport not available for server '{name}'. "
                    "Ensure mcp>=1.0.0 is installed."
                )
            read, write = await self._stack.enter_async_context(
                sse_client(cfg.url, headers=cfg.headers)
            )
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    # ── public API ────────────────────────────────────────────────────────────

    async def call_tool(self, server: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具并返回字符串结果 / Call an MCP tool and return string output."""
        session = self._sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server '{server}' is not connected.")
        result = await session.call_tool(tool_name, arguments)
        parts: list[str] = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                raw = item.model_dump() if hasattr(item, "model_dump") else str(item)
                parts.append(json.dumps(raw, ensure_ascii=False))
        return "\n".join(parts) if parts else "(no output)"

    @property
    def tool_infos(self) -> list[McpToolInfo]:
        """All discovered MCP tools across connected servers."""
        return list(self._tool_infos)

    def get_tools(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        NanoHarness 工具字典 + 定义列表，可直接合并到 NanoCore / Return tools_dict and tool_definitions ready to merge into NanoCore.

        Tool names are namespaced as ``{server}__{tool_name}``.
        """
        tools_dict: dict[str, Any] = {}
        definitions: list[dict[str, Any]] = []

        for info in self._tool_infos:
            qualified = info.qualified_name
            # Default args capture current values to avoid closure-over-loop-variable bug
            _s, _t = info.server, info.name

            async def _mcp_fn(
                input_dict: dict[str, Any],
                _ctx: Any,
                _srv: str = _s,
                _tool: str = _t,
            ) -> ToolResult:
                try:
                    content = await self.call_tool(_srv, _tool, input_dict)
                    return ToolResult(status=ToolResultStatus.SUCCESS, content=content)
                except Exception as exc:
                    return ToolResult(
                        status=ToolResultStatus.FAILURE,
                        content=str(exc),
                        error_code="mcp_error",
                    )

            tools_dict[qualified] = _mcp_fn

            schema = info.input_schema or {"type": "object", "properties": {}}
            definitions.append({
                "name": qualified,
                "description": f"[{info.server}] {info.description}",
                "input_schema": schema,
            })

        return tools_dict, definitions

    def connected_servers(self) -> list[str]:
        """Names of successfully connected servers. / 成功连接的服务器名称列表。"""
        return list(self._sessions.keys())

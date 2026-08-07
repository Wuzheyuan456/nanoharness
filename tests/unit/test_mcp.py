"""Tests for MCP config parsing and client tool-name logic. / MCP 配置解析与客户端工具名测试。

No real MCP connections are made — tests cover pure-Python logic only.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nanoharness.mcp.config import McpConfig, _parse_servers
from nanoharness.mcp.types import McpSseConfig, McpStdioConfig, McpToolInfo


# ── McpConfig.load ────────────────────────────────────────────────────────────

def test_mcp_config_load_missing_file(tmp_path: Path):
    cfg = McpConfig.load(paths=[tmp_path / "nonexistent.json"])
    assert cfg.is_empty()
    assert cfg.servers == {}


def test_mcp_config_load_stdio(tmp_path: Path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({
        "mcpServers": {
            "fs": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }
        }
    }))
    cfg = McpConfig.load(paths=[p])
    assert "fs" in cfg.servers
    srv = cfg.servers["fs"]
    assert isinstance(srv, McpStdioConfig)
    assert srv.command == "npx"
    assert srv.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_mcp_config_load_sse(tmp_path: Path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({
        "mcpServers": {
            "remote": {
                "type": "sse",
                "url": "http://localhost:8000/mcp",
                "headers": {"x-api-key": "secret"},
            }
        }
    }))
    cfg = McpConfig.load(paths=[p])
    assert "remote" in cfg.servers
    srv = cfg.servers["remote"]
    assert isinstance(srv, McpSseConfig)
    assert srv.url == "http://localhost:8000/mcp"
    assert srv.headers == {"x-api-key": "secret"}


def test_mcp_config_url_implies_sse(tmp_path: Path):
    """A server with a ``url`` field but no ``type`` is treated as SSE. / 有 url 字段无 type 视为 SSE。"""
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({
        "mcpServers": {
            "api": {"url": "http://localhost:9000/mcp"}
        }
    }))
    cfg = McpConfig.load(paths=[p])
    assert isinstance(cfg.servers["api"], McpSseConfig)


def test_mcp_config_later_path_overrides(tmp_path: Path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps({"mcpServers": {"srv": {"command": "old", "args": []}}}))
    p2.write_text(json.dumps({"mcpServers": {"srv": {"command": "new", "args": ["-v"]}}}))
    cfg = McpConfig.load(paths=[p1, p2])
    assert isinstance(cfg.servers["srv"], McpStdioConfig)
    assert cfg.servers["srv"].command == "new"


def test_mcp_config_multiple_servers(tmp_path: Path):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({
        "mcpServers": {
            "s1": {"command": "npx", "args": []},
            "s2": {"url": "http://localhost:8000"},
        }
    }))
    cfg = McpConfig.load(paths=[p])
    assert len(cfg.servers) == 2
    assert isinstance(cfg.servers["s1"], McpStdioConfig)
    assert isinstance(cfg.servers["s2"], McpSseConfig)


def test_mcp_config_missing_command_skipped(tmp_path: Path):
    """A stdio entry without ``command`` must be silently skipped. / 无 command 的 stdio 条目静默跳过。"""
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"mcpServers": {"bad": {"args": ["-v"]}}}))
    cfg = McpConfig.load(paths=[p])
    assert "bad" not in cfg.servers


def test_mcp_config_malformed_file_skipped(tmp_path: Path):
    """A malformed JSON file must be skipped without raising. / 格式错误的 JSON 文件不抛出异常。"""
    p = tmp_path / "bad.json"
    p.write_text("not json at all!!!")
    cfg = McpConfig.load(paths=[p])
    assert cfg.is_empty()


# ── McpToolInfo ───────────────────────────────────────────────────────────────

def test_mcp_tool_info_qualified_name():
    info = McpToolInfo(server="filesystem", name="read_file", description="reads a file")
    assert info.qualified_name == "filesystem__read_file"


def test_mcp_tool_info_qualified_name_preserves_underscores():
    info = McpToolInfo(server="my_server", name="do_thing", description="")
    assert info.qualified_name == "my_server__do_thing"


# ── McpClient.get_tools ───────────────────────────────────────────────────────

def _make_client_with_tools(tool_infos: list[McpToolInfo]):
    """Build a McpClient instance pre-populated with tool_infos (no real connection). / 预填充 tool_infos 的 McpClient，无真实连接。"""
    from nanoharness.mcp.client import McpClient
    from nanoharness.mcp.config import McpConfig

    client = McpClient(McpConfig())
    client._tool_infos = tool_infos
    return client


def test_get_tools_returns_correct_names():
    infos = [
        McpToolInfo("fs", "read_file", "read a file", {}),
        McpToolInfo("fs", "write_file", "write a file", {}),
    ]
    client = _make_client_with_tools(infos)
    tools_dict, definitions = client.get_tools()

    assert set(tools_dict.keys()) == {"fs__read_file", "fs__write_file"}
    def_names = [d["name"] for d in definitions]
    assert def_names == ["fs__read_file", "fs__write_file"]


def test_get_tools_description_prefixed_with_server():
    infos = [McpToolInfo("db", "query", "Run SQL query", {})]
    client = _make_client_with_tools(infos)
    _, definitions = client.get_tools()
    assert definitions[0]["description"] == "[db] Run SQL query"


def test_get_tools_callables_are_different_per_tool():
    """Each tool must get its own closure, not share the same function. / 每个工具有独立闭包，不共享。"""
    infos = [
        McpToolInfo("s", "tool_a", "a", {}),
        McpToolInfo("s", "tool_b", "b", {}),
    ]
    client = _make_client_with_tools(infos)
    tools_dict, _ = client.get_tools()
    assert tools_dict["s__tool_a"] is not tools_dict["s__tool_b"]


def test_get_tools_default_schema_when_missing():
    infos = [McpToolInfo("srv", "ping", "ping tool", {})]
    client = _make_client_with_tools(infos)
    _, definitions = client.get_tools()
    assert definitions[0]["input_schema"] == {"type": "object", "properties": {}}


def test_get_tools_uses_provided_schema():
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    infos = [McpToolInfo("fs", "read", "read", schema)]
    client = _make_client_with_tools(infos)
    _, definitions = client.get_tools()
    assert definitions[0]["input_schema"] == schema


@pytest.mark.asyncio
async def test_mcp_fn_returns_tool_result_success():
    """The wrapped async callable must return ToolResult on success. / 封装的异步函数成功时返回 ToolResult。"""
    from nanoharness.core.tool_executor import ToolResultStatus

    infos = [McpToolInfo("srv", "echo", "echo", {})]
    client = _make_client_with_tools(infos)
    client.call_tool = AsyncMock(return_value="hello world")  # type: ignore[method-assign]

    tools_dict, _ = client.get_tools()
    fn = tools_dict["srv__echo"]
    result = await fn({"msg": "hi"}, None)

    assert result.status == ToolResultStatus.SUCCESS
    assert result.content == "hello world"


@pytest.mark.asyncio
async def test_mcp_fn_returns_tool_result_failure():
    """The wrapped async callable must return ToolResult on error. / 封装的异步函数异常时返回 failure ToolResult。"""
    from nanoharness.core.tool_executor import ToolResultStatus

    infos = [McpToolInfo("srv", "bad_tool", "breaks", {})]
    client = _make_client_with_tools(infos)
    client.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))  # type: ignore[method-assign]

    tools_dict, _ = client.get_tools()
    fn = tools_dict["srv__bad_tool"]
    result = await fn({}, None)

    assert result.status == ToolResultStatus.FAILURE
    assert result.error_code == "mcp_error"
    assert "connection lost" in result.content


# ── McpClient.call_tool ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_tool_raises_when_not_connected():
    from nanoharness.mcp.client import McpClient
    from nanoharness.mcp.config import McpConfig

    client = McpClient(McpConfig())
    async with client:
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("missing_server", "some_tool", {})


# ── McpClient no mcp package ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_client_graceful_when_mcp_not_installed(tmp_path: Path):
    """If mcp package absent, McpClient must connect without error and have 0 tools. / mcp 包未安装时，连接无异常且工具数为 0。"""
    import builtins
    real_import = builtins.__import__

    def _mock_import(name: str, *args, **kwargs):
        if name.startswith("mcp"):
            raise ImportError("mcp not installed")
        return real_import(name, *args, **kwargs)

    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"mcpServers": {"srv": {"command": "npx", "args": []}}}))
    cfg = McpConfig.load(paths=[p])

    from nanoharness.mcp.client import McpClient

    with patch("builtins.__import__", side_effect=_mock_import):
        async with McpClient(cfg) as client:
            assert client.tool_infos == []
            assert client.connected_servers() == []

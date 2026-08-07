"""Load MCP server config from JSON files. / 从 JSON 文件加载 MCP 服务器配置。

Default search paths (later files win on conflict / 后者覆盖前者):
  ~/.nanoharness/mcp.json   — user-global
  ./.mcp.json               — project-local

For nano the caller should prepend ~/.nano/mcp.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from nanoharness.mcp.types import McpServerConfig, McpSseConfig, McpStdioConfig

_NANO_DIR = Path.home() / ".nano"
_NH_DIR   = Path.home() / ".nanoharness"


@dataclass
class McpConfig:
    """Parsed MCP configuration holding all server definitions. / 包含全部服务器定义的解析后配置。"""
    servers: dict[str, McpServerConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, paths: list[Path] | None = None) -> "McpConfig":
        """从多个路径加载，后者覆盖同名服务器 / Load from multiple paths; later paths override earlier on name collision."""
        if paths is None:
            paths = _default_paths()
        merged: dict = {}
        for p in paths:
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    merged.update(data.get("mcpServers", {}))
                except Exception:
                    pass  # malformed file — skip silently
        return cls(servers=_parse_servers(merged))

    @classmethod
    def load_for_nano(cls) -> "McpConfig":
        """nano 专用加载顺序: ~/.nanoharness/mcp.json → ~/.nano/mcp.json → ./mcp.json / Nano-specific load order."""
        return cls.load([
            _NH_DIR / "mcp.json",
            _NANO_DIR / "mcp.json",
            Path.cwd() / "mcp.json",
        ])

    def is_empty(self) -> bool:
        return not self.servers


def _default_paths() -> list[Path]:
    return [_NH_DIR / "mcp.json", Path.cwd() / "mcp.json"]


def _parse_servers(raw: dict) -> dict[str, McpServerConfig]:
    result: dict[str, McpServerConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("type") == "sse" or "url" in cfg:
            result[name] = McpSseConfig(
                url=cfg["url"],
                headers=cfg.get("headers", {}),
            )
        else:
            command = cfg.get("command")
            if not command:
                continue
            result[name] = McpStdioConfig(
                command=command,
                args=cfg.get("args", []),
                env=cfg.get("env"),
                cwd=cfg.get("cwd"),
            )
    return result

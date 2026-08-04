from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillDef:
    """
    一个技能的定义：工具集 + 系统提示词补丁 + 路由能力标签 / Definition of a skill: tool set, system-prompt patch, and capability tags for routing.

    tools 语义 / tools semantics:
      []          — 空列表 = 继承全部内置工具（默认行为）/ empty = inherit all built-in tools (default)
      ["_none"]   — 显式无工具哨兵值 / explicit no-tools sentinel
      ["calc"]    — 仅列出的工具 / only the listed tools
    """

    name: str
    description: str = ""
    system_prompt_patch: str = ""
    tools: list[str] = field(default_factory=list)
    tier: str = "T1"
    capabilities: list[str] = field(default_factory=list)
    source_path: Path | None = None

    @classmethod
    def from_toml(cls, path: Path) -> "SkillDef":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            system_prompt_patch=data.get("system_prompt_patch", ""),
            tools=data.get("tools", []),
            tier=data.get("tier", "T1"),
            capabilities=data.get("capabilities", []),
            source_path=path,
        )

    def tool_summary(self) -> str:
        """人类可读的工具列表 / Human-readable tool list."""
        if self.tools == ["_none"]:
            return "（无）"
        if not self.tools:
            return "全部"
        return ", ".join(self.tools)

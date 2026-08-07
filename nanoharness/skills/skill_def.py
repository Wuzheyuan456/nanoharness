from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _parse_md_frontmatter(text: str) -> dict:
    """解析 --- 块内的简单 key: value 行（无需 PyYAML）/ Parse simple key: value lines inside --- block (no PyYAML needed)."""
    result: dict = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        # 列表格式：[a, b, c] / list format: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            result[key] = [v.strip().strip("\"'") for v in inner.split(",") if v.strip()]
        elif val:
            result[key] = val.strip("\"'")
    return result


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

    @classmethod
    def from_markdown(cls, path: Path) -> "SkillDef":
        """
        从 Markdown 文件解析技能定义 / Parse a skill definition from a Markdown file.

        支持两种格式 / Supports two formats:

        格式一（带 YAML frontmatter）/ Format 1 (with YAML frontmatter):
            ---
            name: researcher
            description: Web research agent
            tools: [web_search, current_datetime]
            tier: T2
            capabilities: [search]
            ---
            You are a thorough researcher...

        格式二（纯 Markdown）/ Format 2 (plain Markdown):
            # researcher
            Web research agent.
            You are a thorough researcher...
        """
        text = path.read_text(encoding="utf-8")
        meta: dict = {}
        body = text

        # 解析 frontmatter 块 / Parse frontmatter block
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end != -1:
                meta = _parse_md_frontmatter(text[4:end])
                body = text[end + 5:]   # 跳过 \n---\n / skip \n---\n

        # name: frontmatter > 文件名 / frontmatter > file stem
        name = str(meta["name"]) if "name" in meta else path.stem

        # description: frontmatter > body 里第一个非空非标题行 / frontmatter > first non-empty non-heading line in body
        description = str(meta["description"]) if "description" in meta else ""
        if not description:
            for line in body.splitlines():
                stripped = line.strip().lstrip("#").strip()
                if stripped:
                    description = stripped
                    break

        return cls(
            name=name,
            description=description,
            system_prompt_patch=body.strip(),
            tools=meta.get("tools", []),
            tier=str(meta.get("tier", "T1")),
            capabilities=meta.get("capabilities", []),
            source_path=path,
        )

    def tool_summary(self) -> str:
        """人类可读的工具列表 / Human-readable tool list."""
        if self.tools == ["_none"]:
            return "（无）"
        if not self.tools:
            return "全部"
        return ", ".join(self.tools)

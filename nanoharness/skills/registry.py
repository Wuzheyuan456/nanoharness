from __future__ import annotations

import logging
from pathlib import Path

from nanoharness.skills.skill_def import SkillDef

log = logging.getLogger(__name__)

_BUILTIN_DIR = Path(__file__).parent / "builtin"


class SkillRegistry:
    """
    懒加载技能注册表，从多个目录扫描 .toml 文件 / Lazy-loading skill registry that scans .toml files from multiple dirs.

    加载优先级（高优先覆盖低优先）/ Loading priority (later dirs override earlier):
      1. builtin/                   — 内置技能 / built-in skills
      2. ~/.nanoharness/skills/     — 用户全局自定义 / user-global custom skills
      3. ./skills/                  — 项目级覆盖 / project-local overrides
      4. extra_dirs（构造参数）      — 额外目录 / extra dirs passed at construction
    """

    def __init__(self, extra_dirs: list[Path] | None = None) -> None:
        self._dirs: list[Path] = [_BUILTIN_DIR]

        user_dir = Path.home() / ".nanoharness" / "skills"
        if user_dir.exists():
            self._dirs.append(user_dir)

        local_dir = Path.cwd() / "skills"
        if local_dir.exists() and local_dir.resolve() != _BUILTIN_DIR.resolve():
            self._dirs.append(local_dir)

        if extra_dirs:
            self._dirs.extend(extra_dirs)

        self._cache: dict[str, SkillDef] | None = None

    # ── 内部 / Internal ────────────────────────────────────────────────────────

    def _reload(self) -> None:
        cache: dict[str, SkillDef] = {}
        for d in self._dirs:
            if not d.exists():
                continue
            # .md 先扫，.toml 后扫：同名时 TOML 覆盖 Markdown（TOML 是显式结构化声明）
            # Scan .md first, then .toml: same-name TOML overrides Markdown (TOML is explicit structured)
            for path in sorted(d.glob("*.md")):
                try:
                    skill = SkillDef.from_markdown(path)
                    cache[skill.name] = skill
                except Exception as exc:
                    log.warning("Failed to load skill from %s: %s", path, exc)
            for path in sorted(d.glob("*.toml")):
                try:
                    skill = SkillDef.from_toml(path)
                    cache[skill.name] = skill   # 后覆盖前（优先级实现）/ later overrides earlier (priority)
                except Exception as exc:
                    log.warning("Failed to load skill from %s: %s", path, exc)
        self._cache = cache

    def _ensure(self) -> dict[str, SkillDef]:
        if self._cache is None:
            self._reload()
        return self._cache  # type: ignore[return-value]

    # ── 公共 API / Public API ─────────────────────────────────────────────────

    def lookup(self, name: str) -> SkillDef | None:
        """按名称查找技能；未找到返回 None / Look up a skill by name; returns None if not found."""
        return self._ensure().get(name)

    def find_by_capability(self, cap: str) -> list[SkillDef]:
        """返回包含指定能力标签的所有技能 / Return all skills that include the given capability tag."""
        return [s for s in self._ensure().values() if cap in s.capabilities]

    def list_all(self) -> list[SkillDef]:
        """按名称排序列出所有已加载的技能 / List all loaded skills sorted by name."""
        return sorted(self._ensure().values(), key=lambda s: s.name)

    def reload(self) -> None:
        """使缓存失效，下次访问时重新从磁盘扫描 / Invalidate cache; next access rescans from disk."""
        self._cache = None

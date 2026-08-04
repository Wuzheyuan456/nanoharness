"""
技能系统单测 / Skill system unit tests.
不依赖真实 LLM / No real LLM required.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanoharness.skills.loader import SkillLoader
from nanoharness.skills.registry import SkillRegistry, _BUILTIN_DIR
from nanoharness.skills.skill_def import SkillDef


# ─── SkillDef ──────────────────────────────────────────────────────────────────

def test_skill_def_from_toml_researcher():
    """SkillDef.from_toml 能正确解析 researcher.toml / SkillDef.from_toml parses researcher.toml correctly."""
    path = _BUILTIN_DIR / "researcher.toml"
    assert path.exists(), f"researcher.toml not found at {path}"

    skill = SkillDef.from_toml(path)
    assert skill.name == "researcher"
    assert "web_search" in skill.tools
    assert "search" in skill.capabilities
    assert skill.source_path == path


def test_skill_def_from_toml_base():
    """base.toml 解析后 tools == ['_none'] / base.toml parses to tools == ['_none']."""
    skill = SkillDef.from_toml(_BUILTIN_DIR / "base.toml")
    assert skill.name == "base"
    assert skill.tools == ["_none"]
    assert skill.tier == "T0"


def test_skill_def_tool_summary_none():
    """`_none` 哨兵时 tool_summary 返回（无）/ tool_summary returns （无） for _none sentinel."""
    skill = SkillDef(name="x", tools=["_none"])
    assert skill.tool_summary() == "（无）"


def test_skill_def_tool_summary_empty():
    """空列表时 tool_summary 返回全部 / empty list returns 全部."""
    skill = SkillDef(name="x", tools=[])
    assert skill.tool_summary() == "全部"


def test_skill_def_tool_summary_specific():
    """具体列表时 tool_summary 返回逗号分隔名 / specific list returns comma-separated names."""
    skill = SkillDef(name="x", tools=["calculator", "web_search"])
    assert skill.tool_summary() == "calculator, web_search"


# ─── SkillRegistry ────────────────────────────────────────────────────────────

def test_skill_registry_lists_four_builtins():
    """内置技能目录应含 base / math / researcher / full 四个 / Builtin dir contains base/math/researcher/full."""
    reg = SkillRegistry()
    names = {s.name for s in reg.list_all()}
    assert {"base", "math", "researcher", "full"} <= names


def test_skill_registry_lookup_known():
    """lookup 已知名称返回 SkillDef / lookup returns a SkillDef for a known name."""
    reg = SkillRegistry()
    sk = reg.lookup("math")
    assert sk is not None
    assert sk.name == "math"
    assert "calculator" in sk.tools


def test_skill_registry_lookup_unknown():
    """lookup 未知名称返回 None / lookup returns None for an unknown name."""
    reg = SkillRegistry()
    assert reg.lookup("nonexistent-skill-xyz") is None


def test_skill_registry_find_by_capability():
    """find_by_capability 返回含该标签的所有技能 / find_by_capability returns all skills with the given tag."""
    reg = SkillRegistry()
    results = reg.find_by_capability("math")
    names = {s.name for s in results}
    assert "math" in names


def test_skill_registry_reload_invalidates_cache(tmp_path: Path):
    """reload() 使缓存失效，下次访问时重新扫描 / reload() invalidates cache; next access rescans."""
    reg = SkillRegistry(extra_dirs=[tmp_path])
    _ = reg.list_all()   # 触发加载 / trigger load
    assert reg._cache is not None

    reg.reload()
    assert reg._cache is None

    _ = reg.list_all()   # 重新加载 / reload
    assert reg._cache is not None


def test_skill_registry_extra_dir_overrides(tmp_path: Path):
    """extra_dirs 中同名 skill 覆盖内置版本 / extra_dirs skill overrides builtin of same name."""
    custom_toml = tmp_path / "math.toml"
    custom_toml.write_text(
        'name = "math"\n'
        'description = "custom math"\n'
        'tools = ["calculator"]\n'
        'tier = "T2"\n'
        'capabilities = []\n'
    )
    reg = SkillRegistry(extra_dirs=[tmp_path])
    sk = reg.lookup("math")
    assert sk is not None
    assert sk.description == "custom math"
    assert sk.tier == "T2"


# ─── SkillLoader ──────────────────────────────────────────────────────────────

_ALL_TOOLS = {"calculator": lambda x: x, "web_search": lambda x: x, "current_datetime": lambda x: x}
_ALL_DEFS = [
    {"name": "calculator", "description": "calc"},
    {"name": "web_search", "description": "search"},
    {"name": "current_datetime", "description": "dt"},
]


def test_skill_loader_filter_none_sentinel():
    """tools == ['_none'] 时过滤结果为空 / tools ['_none'] returns empty dicts."""
    skill = SkillDef(name="base", tools=["_none"])
    t, d = SkillLoader.filter_tools(skill, _ALL_TOOLS, _ALL_DEFS)
    assert t == {}
    assert d == []


def test_skill_loader_filter_empty_inherits_all():
    """tools == [] 时继承全部工具 / empty tools list inherits all tools."""
    skill = SkillDef(name="full", tools=[])
    t, d = SkillLoader.filter_tools(skill, _ALL_TOOLS, _ALL_DEFS)
    assert set(t.keys()) == {"calculator", "web_search", "current_datetime"}
    assert len(d) == 3


def test_skill_loader_filter_specific():
    """tools = ['calculator'] 时仅保留 calculator / specific list keeps only listed tools."""
    skill = SkillDef(name="math", tools=["calculator"])
    t, d = SkillLoader.filter_tools(skill, _ALL_TOOLS, _ALL_DEFS)
    assert list(t.keys()) == ["calculator"]
    assert [x["name"] for x in d] == ["calculator"]


def test_skill_loader_filter_ignores_missing():
    """tools 列表中不存在的工具名被安全忽略 / unknown tool names in the list are safely ignored."""
    skill = SkillDef(name="x", tools=["calculator", "nonexistent_tool"])
    t, d = SkillLoader.filter_tools(skill, _ALL_TOOLS, _ALL_DEFS)
    assert set(t.keys()) == {"calculator"}
    assert [x["name"] for x in d] == ["calculator"]


def test_skill_loader_patch_system_with_patch():
    """有补丁时前置 [Skill: name] 段落 / with a patch the result prepends [Skill: name] block."""
    skill = SkillDef(name="researcher", system_prompt_patch="搜索优先。")
    result = SkillLoader.patch_system(skill, "基础提示。")
    assert result.startswith("[Skill: researcher]\n搜索优先。")
    assert "基础提示。" in result


def test_skill_loader_patch_system_empty():
    """补丁为空时返回原始 base / empty patch returns base unchanged."""
    skill = SkillDef(name="base", system_prompt_patch="")
    base = "原始系统提示。"
    assert SkillLoader.patch_system(skill, base) == base


def test_skill_loader_apply_hot_swap():
    """SkillLoader.apply 正确更新 nano.swap_tools 和 ctx.active_skill / apply correctly updates nano.swap_tools and ctx.active_skill."""
    skill = SkillDef(name="math", tools=["calculator"], system_prompt_patch="用计算器。")

    mock_ctx = MagicMock()
    mock_nano = MagicMock()
    mock_nano.ctx = mock_ctx

    SkillLoader.apply(mock_nano, skill, _ALL_TOOLS, _ALL_DEFS)

    mock_nano.swap_tools.assert_called_once()
    call_args = mock_nano.swap_tools.call_args[0]
    assert set(call_args[0].keys()) == {"calculator"}    # tools dict
    assert [x["name"] for x in call_args[1]] == ["calculator"]  # defs list
    assert mock_ctx.active_skill == "math"

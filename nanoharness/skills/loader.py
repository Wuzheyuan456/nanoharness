from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanoharness.skills.skill_def import SkillDef

if TYPE_CHECKING:
    from nanoharness.core.nano_core import NanoCore


class SkillLoader:
    """
    将 SkillDef 应用到工具集和系统提示词 / Applies a SkillDef to a tool set and system prompt.
    所有方法为静态方法，无状态 / All methods are static and stateless.
    """

    @staticmethod
    def filter_tools(
        skill: SkillDef,
        all_tools: dict[str, Any],
        all_defs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        根据 skill.tools 过滤工具字典和定义列表 / Filter the tools dict and defs list per skill.tools.

        tools == ["_none"]  →  ({}, [])          # 显式无工具 / explicit no-tools
        tools == []         →  (all_tools, all_defs)  # 继承全部 / inherit all
        tools == [name, …]  →  仅保留列出的工具 / keep only listed tools
        """
        if skill.tools == ["_none"]:
            return {}, []
        if not skill.tools:
            return all_tools, all_defs
        allowed = set(skill.tools)
        return (
            {k: v for k, v in all_tools.items() if k in allowed},
            [d for d in all_defs if d.get("name") in allowed],
        )

    @staticmethod
    def patch_system(skill: SkillDef, base: str) -> str:
        """
        将技能的系统提示词补丁前置到基础系统提示词 / Prepend the skill's system-prompt patch to the base.
        补丁为空则直接返回 base / Returns base unchanged if the patch is empty.
        """
        patch = skill.system_prompt_patch.strip()
        if not patch:
            return base
        return f"[Skill: {skill.name}]\n{patch}\n\n{base}"

    @staticmethod
    def apply(
        nano: "NanoCore",
        skill: SkillDef,
        all_tools: dict[str, Any],
        all_defs: list[dict[str, Any]],
    ) -> None:
        """
        热换技能到正在运行的 NanoCore（下一个 run_turn 生效）/ Hot-swap skill into a live NanoCore (effective on next run_turn).
        asyncio 单线程，+1/-1 不在 await 段，无竞态 / asyncio single-thread; +1/-1 are outside await sections, no race.
        """
        tools, defs = SkillLoader.filter_tools(skill, all_tools, all_defs)
        nano.swap_tools(tools, defs)
        nano.ctx.active_skill = skill.name

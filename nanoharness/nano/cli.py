"""
Nano — personal AI assistant built on NanoHarness.

Usage / 用法:
    nano                        # interactive chat  / 交互式对话
    nano -p "what time is it"   # one-shot prompt   / 单次查询
    nano --skill researcher     # activate a skill  / 激活技能
    nano init                   # create ~/.nano/config.toml
    nano skills                 # list available skills
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── ANSI colors ───────────────────────────────────────────────────────────────
_R      = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"


def _print_banner(config_name: str, tool_names: list[str], skill_name: str | None) -> None:
    tools_str = ", ".join(tool_names) if tool_names else "none"
    skill_tag = f"  [{skill_name}]" if skill_name else ""
    line = "─" * 52
    print(f"\n{_BOLD}┌{line}┐")
    print(f"│  {config_name}{skill_tag:<47}│")
    print(f"│  tools: {tools_str:<43}│")
    print(f"│  {_DIM}bye{_BOLD} to quit · {_DIM}/skill NAME{_BOLD} to switch · {_DIM}/skills{_BOLD} to list{' ' * 4}│")
    print(f"└{line}┘{_R}\n")


async def _run(args: argparse.Namespace, one_shot: bool = False) -> None:
    from nanoharness.core.context import AgentContext
    from nanoharness.core.event_store import DoneEvent, ErrorEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
    from nanoharness.core.nano_core import NanoCore
    from nanoharness.provider.anthropic import AnthropicProvider
    from nanoharness.router.decision_log import DecisionLog
    from nanoharness.router.llm_router import LLMRouter
    from nanoharness.router.tiers import Tier, TierRegistry
    from nanoharness.skills import SkillLoader, SkillRegistry
    from nanoharness.tools import get_builtin_tools
    from nanoharness.mcp.client import McpClient
    from nanoharness.mcp.config import McpConfig

    from nanoharness.nano.config import NanoConfig, SKILLS_DIR
    from nanoharness.nano.persona import DEFAULT_SYSTEM_PROMPT

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(f"{_RED}Error: ANTHROPIC_API_KEY is not set.{_R}")
        print(f"  {_DIM}export ANTHROPIC_API_KEY=sk-ant-...{_R}")
        sys.exit(1)

    cfg = NanoConfig.load()
    base_system = cfg.system_prompt or DEFAULT_SYSTEM_PROMPT

    # Skill registry — includes ~/.nano/skills/ as extra dir
    extra_dirs = [SKILLS_DIR] if SKILLS_DIR.exists() else []
    skill_registry = SkillRegistry(extra_dirs=extra_dirs)

    all_tools, all_tool_defs = get_builtin_tools()

    # MCP tool loading — connect once, reuse across all turns
    mcp_config = McpConfig.load_for_nano()
    mcp_client = McpClient(mcp_config)
    await mcp_client.__aenter__()
    if mcp_client.tool_infos:
        mcp_tools, mcp_defs = mcp_client.get_tools()
        all_tools = {**all_tools, **mcp_tools}
        all_tool_defs = all_tool_defs + mcp_defs
        if not one_shot:
            servers = ", ".join(mcp_client.connected_servers())
            print(f"{_DIM}[MCP: {len(mcp_client.tool_infos)} tools from {servers}]{_R}")

    tools, tool_defs = all_tools, all_tool_defs
    current_system = base_system
    active_skill: str | None = None

    # --skill arg or default_skill from config
    skill_name = getattr(args, "skill", None) or cfg.default_skill or None
    if skill_name:
        sk = skill_registry.lookup(skill_name)
        if sk is None:
            print(f"{_RED}Skill '{skill_name}' not found. Run 'nano skills' to see available skills.{_R}")
            sys.exit(1)
        tools, tool_defs = SkillLoader.filter_tools(sk, all_tools, all_tool_defs)
        current_system = SkillLoader.patch_system(sk, base_system)
        active_skill = sk.name

    # Routing setup
    registry = TierRegistry()
    forced_tier_str = getattr(args, "tier", None) or (None if cfg.tier == "auto" else cfg.tier)
    forced_tier = Tier[forced_tier_str] if forced_tier_str else None

    router: LLMRouter | None = None
    if not forced_tier:
        classify_provider = AnthropicProvider(model_id=registry.model_id(Tier.T0), api_key=api_key)
        router = LLMRouter(
            provider=classify_provider,
            registry=registry,
            decision_log=DecisionLog(str(Path.home() / ".nano" / "router.db")),
            timeout=30.0,
        )

    default_model = registry.model_id(forced_tier or Tier.T1)
    ctx = AgentContext(
        agent_id="nano",
        session_key="nano-session",
        system_prompt=current_system,
        model_id=default_model,
        active_skill=active_skill,
    )

    if not one_shot:
        _print_banner(cfg.name, list(tools.keys()), active_skill)

    turn_count = 0

    async def _run_turn(user_input: str) -> None:
        nonlocal turn_count
        turn_count += 1

        # Route to model tier
        if router:
            try:
                route = await router.classify(user_input, trace_id=f"nano-{turn_count}", session_key="nano-session")
                model_id = registry.model_id(route.tier)
                if not one_shot:
                    print(f"{_DIM}[{route.tier.value} · {model_id}]{_R}")
            except Exception:
                model_id = default_model
        else:
            tier = forced_tier or Tier.T1
            model_id = registry.model_id(tier)

        ctx.model_id = model_id
        provider = AnthropicProvider(model_id=model_id, api_key=api_key)
        nano = NanoCore(ctx=ctx, provider=provider, tools=tools, tool_definitions=tool_defs)

        if not one_shot:
            print(f"\n{_BOLD}{cfg.name}:{_R} ", end="", flush=True)

        after_tool = False
        try:
            async for event in nano.run_turn(user_input):
                if isinstance(event, TextDeltaEvent):
                    if after_tool and not one_shot:
                        print(f"\n{_BOLD}{cfg.name}:{_R} ", end="", flush=True)
                        after_tool = False
                    print(event.delta, end="", flush=True)

                elif isinstance(event, ToolCallEvent) and not one_shot:
                    print(f"\n{_YELLOW}  → {event.tool_name}({event.input_summary[:60]}){_R}", end="", flush=True)

                elif isinstance(event, ToolResultEvent) and not one_shot:
                    icon = f"{_GREEN}✓{_R}" if event.success else f"{_RED}✗{_R}"
                    preview = event.output_preview[:80].replace("\n", " ")
                    print(f" {icon} {_DIM}{preview}{_R}", flush=True)
                    after_tool = True

                elif isinstance(event, DoneEvent) and not one_shot:
                    elapsed = event.elapsed_ms / 1000
                    tokens = event.total_input_tokens + event.total_output_tokens
                    print(f"\n{_DIM}[{elapsed:.1f}s · {tokens} tokens]{_R}")

                elif isinstance(event, ErrorEvent):
                    print(f"\n{_RED}Error: {event.error_message}{_R}")

        except KeyboardInterrupt:
            if not one_shot:
                print(f"\n{_DIM}(interrupted){_R}")
        except Exception as exc:
            print(f"\n{_RED}Error: {exc}{_R}")

        if not one_shot:
            print()

    # ── One-shot mode ──────────────────────────────────────────────────────────
    if one_shot:
        try:
            await _run_turn(args.prompt)
        finally:
            await mcp_client.__aexit__(None, None, None)
        return

    # ── Interactive loop ───────────────────────────────────────────────────────
    try:
        while True:
            try:
                user_input = input(f"{_BOLD}You:{_R} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_DIM}Goodbye.{_R}")
                break

            if not user_input:
                continue
            if user_input.lower() in ("bye", "exit", "quit", "再见", "q"):
                print(f"{_DIM}Goodbye.{_R}")
                break

            # /skills — list available skills
            if user_input.strip() == "/skills":
                all_skills = skill_registry.list_all()
                print(f"\n{_BOLD}Available skills ({len(all_skills)}):{_R}")
                for s in all_skills:
                    marker = "  ◀ active" if s.name == ctx.active_skill else ""
                    print(f"  {_CYAN}{s.name:<14}{_R}{s.description}  {_DIM}[{s.tool_summary()}]{_R}{marker}")
                print()
                continue

            # /skill NAME — hot-swap skill
            if user_input.startswith("/skill "):
                sname = user_input[7:].strip()
                sk = skill_registry.lookup(sname)
                if sk is None:
                    print(f"{_RED}Skill '{sname}' not found.{_R}")
                else:
                    tools, tool_defs = SkillLoader.filter_tools(sk, all_tools, all_tool_defs)
                    current_system = SkillLoader.patch_system(sk, base_system)
                    ctx.system_prompt = current_system
                    ctx.active_skill = sk.name
                    active_skill = sk.name
                    print(f"{_CYAN}[{sk.name}] activated — tools: {sk.tool_summary()}{_R}")
                continue

            await _run_turn(user_input)
    finally:
        await mcp_client.__aexit__(None, None, None)


def _cmd_init(_args: argparse.Namespace) -> None:
    """Create ~/.nano/config.toml and ~/.nano/skills/ / 创建配置文件和技能目录"""
    from nanoharness.nano.config import NANO_DIR, CONFIG_PATH, SKILLS_DIR, CONFIG_TEMPLATE

    NANO_DIR.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(exist_ok=True)

    if CONFIG_PATH.exists():
        print(f"{_YELLOW}~/.nano/config.toml already exists — not overwritten.{_R}")
        print(f"  {_DIM}Edit it directly to change settings.{_R}")
    else:
        CONFIG_PATH.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        print(f"{_GREEN}✓{_R} Created {_BOLD}~/.nano/config.toml{_R}")
        print(f"{_GREEN}✓{_R} Created {_BOLD}~/.nano/skills/{_R}  (drop .toml or .md skill files here)")

    mcp_path = NANO_DIR / "mcp.json"
    if not mcp_path.exists():
        mcp_path.write_text('{"mcpServers": {}}\n', encoding="utf-8")
        print(f"{_GREEN}✓{_R} Created {_BOLD}~/.nano/mcp.json{_R}  (add MCP server entries to enable MCP tools)")

    print(f"\n  Next: set your API key and run {_BOLD}nano{_R}")
    print(f"  {_DIM}export ANTHROPIC_API_KEY=sk-ant-...{_R}")
    print(f"  {_DIM}pip install 'nanoharness[mcp]'  # optional: enable MCP servers{_R}")


def _cmd_skills(_args: argparse.Namespace) -> None:
    """List available skills / 列出可用技能"""
    from nanoharness.skills import SkillRegistry
    from nanoharness.nano.config import SKILLS_DIR

    extra_dirs = [SKILLS_DIR] if SKILLS_DIR.exists() else []
    registry = SkillRegistry(extra_dirs=extra_dirs)
    skills = registry.list_all()

    print(f"\n{_BOLD}Available skills ({len(skills)}){_R}\n")
    for s in skills:
        src = str(s.source_path).replace(str(Path.home()), "~") if s.source_path else "?"
        print(f"  {_CYAN}{_BOLD}{s.name}{_R}  {_DIM}{src}{_R}")
        print(f"    {s.description}")
        print(f"    tools: {s.tool_summary()}  tier: {s.tier}"
              + (f"  capabilities: {', '.join(s.capabilities)}" if s.capabilities else ""))
        print()
    print(f"  {_DIM}Usage: nano --skill <name>{_R}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nano",
        description="Nano — personal AI assistant built on NanoHarness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nano                          # start interactive chat\n"
            "  nano -p 'what time is it'     # one-shot query\n"
            "  nano --skill researcher       # start with a skill active\n"
            "  nano init                     # create ~/.nano/config.toml\n"
            "  nano skills                   # list available skills\n"
        ),
    )

    # Top-level flags (apply to interactive chat)
    parser.add_argument("-p", "--prompt", default=None, metavar="TEXT",
                        help="one-shot prompt — print response and exit")
    parser.add_argument("--skill", default=None, metavar="NAME",
                        help="activate a skill on startup")
    parser.add_argument("--tier", choices=["T0", "T1", "T2", "T3"], default=None,
                        help="force a model tier (skips LLM routing)")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help="create ~/.nano/config.toml and ~/.nano/skills/")
    sub.add_parser("skills", help="list available skills")

    args = parser.parse_args()

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "skills":
        _cmd_skills(args)
    elif args.prompt:
        asyncio.run(_run(args, one_shot=True))
    else:
        asyncio.run(_run(args, one_shot=False))


if __name__ == "__main__":
    main()

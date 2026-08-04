"""
NanoHarness CLI — 一行启动 ReAct 对话智能体 / One-line CLI to launch the ReAct agent.

用法 / Usage:
    nanoharness chat                  # 自动路由，交互式对话
    nanoharness chat --tier T2        # 强制 T2 档位
    nanoharness chat --no-tools       # 纯对话，不调工具
    nanoharness chat --skill researcher  # 用 researcher 技能启动
    nanoharness skills                # 列出所有可用技能
    nanoharness serve                 # 启动 OpenAI 兼容 HTTP 服务器
    nanoharness serve --port 8080     # 指定端口
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── ANSI 颜色 / ANSI colors ───────────────────────────────────────────────────
_R = "\033[0m"        # reset
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BLUE = "\033[34m"


_DEFAULT_SYSTEM = """\
你是一个通用智能助手，配备以下工具：
- calculator：数学表达式求值（支持 sqrt、log、sin/cos 等 math 函数）
- current_datetime：获取当前日期时间（支持时区）
- web_search：DuckDuckGo 网络搜索（无需 API key）

工具调用准则：
- 需要实时数据（天气、价格、新闻）时调用 web_search
- 需要计算时调用 calculator，不要心算
- 需要当前时间时调用 current_datetime
- 知识性问题（定义、原理）可直接回答，无需工具
"""


def _print_banner(tool_names: list[str], skill_name: str | None = None) -> None:
    tools_str = ", ".join(tool_names) if tool_names else "（无）"
    skill_str = f" [{skill_name}]" if skill_name else ""
    line = "═" * 54
    print(f"\n{_BOLD}╔{line}╗")
    print(f"║  NanoHarness ReAct Agent{skill_str:<28}║")
    print(f"║  工具: {tools_str:<46}║")
    print(f"║  输入 {_CYAN}quit{_BOLD} 退出，{_CYAN}/skill NAME{_BOLD} 切换技能{' ' * 20}║")
    print(f"╚{line}╝{_R}\n")


async def _run_chat(args: argparse.Namespace) -> None:
    from nanoharness.core.context import AgentContext
    from nanoharness.core.event_store import (
        DoneEvent,
        ErrorEvent,
        TextDeltaEvent,
        ToolCallEvent,
        ToolResultEvent,
    )
    from nanoharness.core.nano_core import NanoCore
    from nanoharness.provider.anthropic import AnthropicProvider
    from nanoharness.router.decision_log import DecisionLog
    from nanoharness.router.llm_router import LLMRouter
    from nanoharness.router.tiers import Tier, TierRegistry
    from nanoharness.tools import get_builtin_tools

    from nanoharness.skills import SkillLoader, SkillRegistry

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(f"{_RED}错误: 未设置 ANTHROPIC_API_KEY{_R}")
        print(f"  {_DIM}配置方法: export ANTHROPIC_API_KEY=sk-ant-...{_R}")
        sys.exit(1)

    all_tools, all_tool_defs = ({}, []) if args.no_tools else get_builtin_tools()
    skill_registry = SkillRegistry()
    base_system = args.system or _DEFAULT_SYSTEM

    # 启动时应用 --skill / Apply --skill at startup
    active_skill_name: str | None = None
    tools, tool_defs = all_tools, all_tool_defs
    current_system = base_system
    if getattr(args, "skill", None):
        sk = skill_registry.lookup(args.skill)
        if sk is None:
            print(f"{_RED}错误: 找不到技能 '{args.skill}'。用 'nanoharness skills' 查看可用列表。{_R}")
            sys.exit(1)
        tools, tool_defs = SkillLoader.filter_tools(sk, all_tools, all_tool_defs)
        current_system = SkillLoader.patch_system(sk, base_system)
        active_skill_name = sk.name
        print(f"{_CYAN}[Skill] {sk.name} 已激活 → 工具: {sk.tool_summary()}{_R}")

    _print_banner(list(tools.keys()), active_skill_name)

    registry = TierRegistry()
    forced_tier = Tier[args.tier] if args.tier else None

    # 路由器（不强制档位时启用）/ router (enabled when tier is not forced)
    router: LLMRouter | None = None
    if not forced_tier:
        classify_provider = AnthropicProvider(
            model_id=registry.model_id(Tier.T0),
            api_key=api_key,
        )
        router = LLMRouter(
            provider=classify_provider,
            registry=registry,
            decision_log=DecisionLog("router_decisions.db"),
            timeout=30.0,
        )

    default_model = args.model or registry.model_id(forced_tier or Tier.T1)
    ctx = AgentContext(
        agent_id="cli-agent",
        session_key="cli-session",
        system_prompt=current_system,
        model_id=default_model,
        active_skill=active_skill_name,
    )

    turn_count = 0
    while True:
        try:
            user_input = input(f"{_BOLD}You:{_R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}再见！{_R}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye", "再见", "q"):
            print(f"{_DIM}再见！{_R}")
            break

        # ── /skills 内联命令：列出可用技能 / Inline /skills command: list available skills ──
        if user_input.strip() == "/skills":
            skills = skill_registry.list_all()
            print(f"\n{_BOLD}可用技能 ({len(skills)} 个):{_R}")
            for s in skills:
                marker = " ◀ 当前" if s.name == ctx.active_skill else ""
                print(f"  {_CYAN}{s.name:<12}{_R} {s.description}  {_DIM}[{s.tool_summary()}]{_R}{marker}")
            print(f"\n  用法: /skill <name>\n")
            continue

        # ── /skill NAME 内联命令：热换技能 / Inline /skill NAME: hot-swap skill ──
        if user_input.startswith("/skill "):
            skill_name = user_input[7:].strip()
            sk = skill_registry.lookup(skill_name)
            if sk is None:
                print(f"{_RED}[Skill] 找不到 '{skill_name}'。输入 /skills 查看可用列表。{_R}")
            else:
                tools, tool_defs = SkillLoader.filter_tools(sk, all_tools, all_tool_defs)
                current_system = SkillLoader.patch_system(sk, base_system)
                ctx.system_prompt = current_system
                ctx.active_skill = sk.name
                print(f"{_CYAN}[Skill] → {sk.name}  工具: {sk.tool_summary()}{_R}")
            continue

        turn_count += 1

        # ── 路由分类 / Routing ──────────────────────────────────────────────
        if router:
            try:
                route = await router.classify(
                    user_input,
                    trace_id=f"cli-{turn_count}",
                    session_key="cli-session",
                )
                model_id = args.model or registry.model_id(route.tier)
                print(
                    f"{_CYAN}[Router] {route.tier.value} → {model_id} "
                    f"(conf={route.confidence:.2f}, {route.method}){_R}"
                )
            except Exception as exc:
                model_id = default_model
                print(f"{_DIM}[Router] 分类失败({exc})，使用默认模型 {model_id}{_R}")
        else:
            tier = forced_tier or Tier.T1
            model_id = args.model or registry.model_id(tier)
            print(f"{_CYAN}[Tier] 强制 {tier.value} → {model_id}{_R}")

        ctx.model_id = model_id

        # ── ReAct 推理 / ReAct reasoning ────────────────────────────────────
        provider = AnthropicProvider(model_id=model_id, api_key=api_key)
        nano = NanoCore(
            ctx=ctx,
            provider=provider,
            tools=tools,           # 可能已被 /skill 热换 / may have been hot-swapped by /skill
            tool_definitions=tool_defs,
        )

        print(f"\n{_BOLD}Assistant:{_R} ", end="", flush=True)
        after_tool = False   # 工具结果之后，下一个 TextDelta 需重打前缀 / after tool result, next TextDelta needs prefix

        try:
            async for event in nano.run_turn(user_input):
                if isinstance(event, TextDeltaEvent):
                    if after_tool:
                        print(f"\n{_BOLD}Assistant:{_R} ", end="", flush=True)
                        after_tool = False
                    print(event.delta, end="", flush=True)

                elif isinstance(event, ToolCallEvent):
                    print(
                        f"\n{_YELLOW}[Tool] {event.tool_name}"
                        f"({event.input_summary[:60]}){_R}",
                        end="",
                        flush=True,
                    )

                elif isinstance(event, ToolResultEvent):
                    icon = f"{_GREEN}✓{_R}" if event.success else f"{_RED}✗{_R}"
                    preview = event.output_preview[:80].replace("\n", " ")
                    print(f" {icon} {_DIM}{preview}{_R}", flush=True)
                    after_tool = True

                elif isinstance(event, DoneEvent):
                    elapsed = event.elapsed_ms / 1000
                    tokens = event.total_input_tokens + event.total_output_tokens
                    print(
                        f"\n{_DIM}[{elapsed:.1f}s, {tokens} tokens, "
                        f"{event.total_tool_calls} tool calls]{_R}"
                    )

                elif isinstance(event, ErrorEvent):
                    print(f"\n{_RED}[Error] {event.error_message}{_R}")

        except KeyboardInterrupt:
            print(f"\n{_DIM}（已中断）{_R}")
        except Exception as exc:
            print(f"\n{_RED}错误: {exc}{_R}")

        print()


def _run_skills(_args: argparse.Namespace) -> None:
    """列出所有可用技能 / List all available skills."""
    from nanoharness.skills import SkillRegistry

    skill_registry = SkillRegistry()
    skills = skill_registry.list_all()

    line = "═" * 54
    print(f"\n{_BOLD}╔{line}╗")
    print(f"║  NanoHarness Skills{' ' * 34}║")
    print(f"╚{line}╝{_R}\n")

    for s in skills:
        print(f"  {_CYAN}{_BOLD}{s.name}{_R}")
        print(f"    {s.description}")
        print(f"    {_DIM}工具: {s.tool_summary()}  tier: {s.tier}  capabilities: {', '.join(s.capabilities) or '—'}{_R}\n")

    print(f"  {_DIM}用法: nanoharness chat --skill <name>{_R}\n")


def _run_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print(f"{_RED}错误: 未安装 uvicorn。请运行: pip install uvicorn{_R}")
        sys.exit(1)

    from nanoharness.server import create_app

    app = create_app()
    print(f"\n{_BOLD}NanoHarness API Server{_R}")
    print(f"  地址:   http://{args.host}:{args.port}")
    print(f"  文档:   http://{args.host}:{args.port}/docs")
    print(f"  接口:   POST /v1/chat/completions  (OpenAI 兼容)")
    print(f"  接口:   GET  /v1/models")
    print(f"  接口:   GET  /health\n")
    print(f"  接入示例 (Python):")
    print(f"  {_DIM}from openai import OpenAI")
    print(f"  client = OpenAI(base_url='http://{args.host}:{args.port}/v1', api_key='any')")
    print(f"  client.chat.completions.create(model='nanoharness', messages=[...]){_R}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nanoharness",
        description="NanoHarness — 手写 ReAct 智能体引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  nanoharness chat                       # 自动路由对话\n"
            "  nanoharness chat --tier T2             # 强制 T2 档位\n"
            "  nanoharness chat --skill researcher    # 用技能启动\n"
            "  nanoharness skills                     # 列出所有技能\n"
            "  nanoharness serve                      # 启动 API 服务器\n"
            "  nanoharness serve --port 9000          # 自定义端口\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # chat 子命令
    chat_p = sub.add_parser("chat", help="交互式 ReAct 对话智能体")
    chat_p.add_argument(
        "--model", default=None, metavar="MODEL_ID",
        help="强制指定模型 ID（否则由路由器自动选择）",
    )
    chat_p.add_argument(
        "--tier", choices=["T0", "T1", "T2", "T3"], default=None,
        help="强制使用指定档位，跳过 LLM 路由",
    )
    chat_p.add_argument(
        "--no-tools", action="store_true",
        help="禁用内置工具（纯对话模式）",
    )
    chat_p.add_argument(
        "--system", default=None, metavar="PROMPT",
        help="自定义系统提示词",
    )
    chat_p.add_argument(
        "--skill", default=None, metavar="NAME",
        help="启动时激活指定技能（过滤工具 + 注入提示词补丁）",
    )

    # skills 子命令
    sub.add_parser("skills", help="列出所有可用技能（内置 + 用户自定义）")

    # serve 子命令
    serve_p = sub.add_parser("serve", help="启动 OpenAI 兼容 HTTP API 服务器")
    serve_p.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    serve_p.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")

    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_run_chat(args))
    elif args.command == "skills":
        _run_skills(args)
    elif args.command == "serve":
        _run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

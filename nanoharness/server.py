"""
NanoHarness FastAPI 服务器 — OpenAI 兼容 API / FastAPI server with OpenAI-compatible API.

特点 / Features:
  1. 完全兼容 OpenAI Python SDK，只需改 base_url / Drop-in for OpenAI SDK — change base_url only
  2. 自动路由：LLMRouter 按任务难度选 T0~T3 档位 / Auto-routing: LLMRouter picks T0~T3 by task difficulty
  3. 流式 SSE（server-sent events）+ 非流式两种模式 / Streaming SSE + non-streaming
  4. 内置工具（calculator / current_datetime / web_search）/ Built-in tools

接入示例 / Integration example:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
    resp = client.chat.completions.create(
        model="nanoharness",
        messages=[{"role": "user", "content": "sqrt(144) 等于多少？"}],
    )
    print(resp.choices[0].message.content)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanoharness.core.context import AgentContext
from nanoharness.core.context import Message as CoreMessage
from nanoharness.core.event_store import DoneEvent, ErrorEvent, TextDeltaEvent
from nanoharness.core.nano_core import NanoCore
from nanoharness.provider.anthropic import AnthropicProvider
from nanoharness.router.decision_log import DecisionLog
from nanoharness.router.llm_router import LLMRouter
from nanoharness.router.tiers import Tier, TierRegistry
from nanoharness.tools import get_builtin_tools

_DEFAULT_SYSTEM = """\
你是一个通用智能助手，配备以下工具：
- calculator：数学表达式求值（支持 sqrt、log、sin/cos 等 math 函数）
- current_datetime：获取当前日期时间（支持时区）
- web_search：DuckDuckGo 网络搜索（无需 API key）

工具调用准则：需要实时数据时调用 web_search；需要计算时调用 calculator；知识性问题可直接回答。
"""


# ── Pydantic 请求/响应模型（OpenAI 格式）/ Request/response models (OpenAI format) ─────────────

class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "nanoharness"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    # NanoHarness 扩展：可选强制档位 / NanoHarness extension: optional tier override
    tier: str | None = Field(None, description="强制档位 T0/T1/T2/T3，否则自动路由")
    use_tools: bool = Field(True, description="是否启用内置工具")
    # NanoHarness 扩展：激活指定技能（过滤工具 + 注入提示词补丁）/ activate a named skill (filter tools + inject prompt patch)
    skill: str | None = Field(None, description="技能名称，如 'researcher'；优先级高于 use_tools")


class ResponseMessage(BaseModel):
    role: str
    content: str


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage


# ── 应用工厂 / App factory ─────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="NanoHarness API",
        description=(
            "OpenAI 兼容的 LLM API，内置智能路由（T0~T3 档位）和工具调用（calculator / datetime / web_search）。\n\n"
            "**接入方式**：将 OpenAI SDK 的 `base_url` 改为本服务地址即可，`api_key` 传任意值。"
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    _wire_routes(app)
    return app


# ── 路由注册 / Route registration ─────────────────────────────────────────────

def _wire_routes(app: FastAPI) -> None:
    from nanoharness.skills import SkillLoader, SkillRegistry

    registry = TierRegistry()
    tools, tool_defs = get_builtin_tools()
    skill_registry = SkillRegistry()

    # ── 辅助：API key 提取 / Helper: API key extraction ──────────────────────

    def _api_key(request: Request) -> str:
        """优先用 Authorization Bearer，其次用环境变量 / Prefer Authorization Bearer, fallback to env var."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            candidate = auth[7:].strip()
            if candidate and candidate.lower() not in ("any", "none", ""):
                return candidate
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise HTTPException(
                status_code=401,
                detail="未配置 ANTHROPIC_API_KEY。请在环境变量中设置，或通过 Authorization Bearer 传入。",
            )
        return key

    # ── 辅助：路由选档 / Helper: tier selection ──────────────────────────────

    async def _select_tier(api_key: str, req: ChatCompletionRequest) -> Tier:
        if req.tier:
            try:
                return Tier[req.tier.upper()]
            except KeyError:
                raise HTTPException(status_code=400, detail=f"无效档位: {req.tier}，可选 T0/T1/T2/T3")

        # 用最后一条 user 消息路由 / route on last user message
        last_user = next(
            (m.content for m in reversed(req.messages) if m.role == "user" and m.content),
            "",
        )
        if not last_user:
            return Tier.T1

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
        result = await router.classify(last_user, trace_id=uuid.uuid4().hex[:8])
        return result.tier

    # ── 辅助：构建 AgentContext + NanoCore / Helper: build AgentContext + NanoCore ──────────

    def _build_nano(req: ChatCompletionRequest, api_key: str, tier: Tier) -> tuple[NanoCore, AgentContext, str]:
        """返回 (nano, ctx, last_user_content) / Returns (nano, ctx, last_user_content)."""
        model_id = registry.model_id(tier)

        system_prompt = next(
            (m.content for m in req.messages if m.role == "system" and m.content),
            _DEFAULT_SYSTEM,
        )

        # 分离系统消息和对话历史 / separate system message from conversation history
        history_msgs = [m for m in req.messages if m.role != "system"]

        # 找最后一条 user 消息 / find last user message
        last_user_idx = next(
            (i for i in range(len(history_msgs) - 1, -1, -1) if history_msgs[i].role == "user"),
            None,
        )
        if last_user_idx is None:
            raise HTTPException(status_code=400, detail="请求中没有 user 消息")

        last_user_content = history_msgs[last_user_idx].content or ""
        prior = history_msgs[:last_user_idx]

        ctx = AgentContext(
            agent_id=f"srv-{uuid.uuid4().hex[:8]}",
            session_key=uuid.uuid4().hex[:16],
            system_prompt=system_prompt,
            model_id=model_id,
            history=[
                CoreMessage(role=m.role, content=m.content or "")
                for m in prior
                if m.role in ("user", "assistant") and m.content
            ],
        )

        # skill 优先级高于 use_tools / skill takes priority over use_tools
        if req.skill:
            sk = skill_registry.lookup(req.skill)
            if sk is None:
                raise HTTPException(status_code=400, detail=f"找不到技能 '{req.skill}'")
            active_tools, active_defs = SkillLoader.filter_tools(sk, tools, tool_defs)
            system_prompt = SkillLoader.patch_system(sk, system_prompt)
            ctx.system_prompt = system_prompt
            ctx.active_skill = sk.name
        elif not req.use_tools:
            active_tools, active_defs = {}, []
        else:
            active_tools, active_defs = tools, tool_defs

        provider = AnthropicProvider(model_id=model_id, api_key=api_key)
        nano = NanoCore(
            ctx=ctx,
            provider=provider,
            tools=active_tools,
            tool_definitions=active_defs,
        )
        return nano, ctx, last_user_content

    # ── 路由：/health ──────────────────────────────────────────────────────────

    @app.get("/health", tags=["infra"])
    async def health() -> dict[str, str]:
        """服务健康检查 / Health check."""
        return {"status": "ok", "version": "0.1.0"}

    # ── 路由：/v1/models ──────────────────────────────────────────────────────

    @app.get("/v1/models", tags=["openai"])
    async def list_models() -> dict[str, Any]:
        """列出可用模型（OpenAI 兼容）/ List available models (OpenAI-compatible)."""
        model_list = [
            {
                "id": "nanoharness",
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "nanoharness",
                "description": "Auto-routing across T0~T3 tiers",
            }
        ]
        for tier in [Tier.T0, Tier.T1, Tier.T2, Tier.T3]:
            model_list.append({
                "id": f"nanoharness/{tier.value.lower()}",
                "object": "model",
                "created": 1_700_000_000,
                "owned_by": "nanoharness",
                "description": f"Force {tier.value} tier — {registry.model_id(tier)}",
            })
        return {"object": "list", "data": model_list}

    # ── 路由：/v1/chat/completions ─────────────────────────────────────────────

    @app.post("/v1/chat/completions", tags=["openai"])
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
        """
        OpenAI 兼容的聊天补全端点 / OpenAI-compatible chat completion endpoint.

        - `model`: 传 `"nanoharness"` 自动路由，或 `"nanoharness/t2"` 强制档位
        - `stream`: true 返回 SSE 流，false 返回完整响应
        - `tier`: 扩展字段，强制指定档位（T0/T1/T2/T3）
        """
        api_key = _api_key(request)

        # 支持通过 model 字段指定档位，如 "nanoharness/t2" / support tier via model field "nanoharness/t2"
        if "/" in req.model and req.tier is None:
            _, tier_hint = req.model.rsplit("/", 1)
            req = req.model_copy(update={"tier": tier_hint.upper()})

        tier = await _select_tier(api_key, req)
        nano, _ctx, last_user = _build_nano(req, api_key, tier)

        if req.stream:
            return StreamingResponse(
                _sse_stream(nano, last_user, req.model, tier),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return await _collect(nano, last_user, req.model, tier)

    # ── 流式响应生成器 / Streaming response generator ─────────────────────────

    async def _sse_stream(
        nano: NanoCore,
        user_msg: str,
        model_name: str,
        tier: Tier,
    ) -> AsyncIterator[str]:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # 首帧：角色 delta / first frame: role delta
        yield _sse(cid, created, model_name, {"role": "assistant", "tier": tier.value}, None)

        try:
            async for event in nano.run_turn(user_msg):
                if isinstance(event, TextDeltaEvent):
                    yield _sse(cid, created, model_name, {"content": event.delta}, None)
                elif isinstance(event, DoneEvent):
                    yield _sse(cid, created, model_name, {}, "stop")
                elif isinstance(event, ErrorEvent):
                    yield _sse(cid, created, model_name,
                                {"content": f"[Error] {event.error_message}"}, "stop")
        except Exception as exc:
            yield _sse(cid, created, model_name, {"content": f"[Server Error] {exc}"}, "stop")

        yield "data: [DONE]\n\n"

    # ── 非流式响应收集 / Non-streaming response collector ────────────────────

    async def _collect(
        nano: NanoCore,
        user_msg: str,
        model_name: str,
        tier: Tier,
    ) -> ChatCompletionResponse:
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        text_parts: list[str] = []
        in_tok = out_tok = tool_calls = 0

        async for event in nano.run_turn(user_msg):
            if isinstance(event, TextDeltaEvent):
                text_parts.append(event.delta)
            elif isinstance(event, DoneEvent):
                in_tok = event.total_input_tokens
                out_tok = event.total_output_tokens
                tool_calls = event.total_tool_calls
            elif isinstance(event, ErrorEvent):
                raise HTTPException(status_code=500, detail=f"Agent error: {event.error_message}")

        return ChatCompletionResponse(
            id=cid,
            created=created,
            model=model_name,
            choices=[
                Choice(
                    index=0,
                    message=ResponseMessage(role="assistant", content="".join(text_parts)),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
                total_tokens=in_tok + out_tok,
            ),
        )


# ── SSE 帧格式化 / SSE frame formatter ────────────────────────────────────────

def _sse(
    cid: str,
    created: int,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None,
) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

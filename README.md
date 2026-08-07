# NanoHarness

> **Lightweight multi-agent orchestration engine — built from scratch, zero framework dependencies.**
> 轻量级多智能体编排引擎，不依赖任何 Agent 框架，从零手写 ReAct 运行时。

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-225_passing-brightgreen?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-yellow)

NanoHarness implements a complete agent harness stack — hand-written ReAct state machine, LLM difficulty router, three-tier memory, multi-agent orchestration, multi-channel gateway, and observability — without depending on LangChain, LangGraph, or any agent framework. Every component is purpose-built and fully transparent.

NanoHarness 实现了完整的 Agent 基础设施栈——手写 ReAct 状态机、LLM 难度路由、三层记忆、多 Agent 编排、多通道网关与可观测性，不依赖 LangChain / LangGraph / AutoGen 等任何框架。

---

## Architecture 架构

```
NanoHarness
│
├── L1   Core Engine     core/           ReAct async-generator state machine
│                                        Semantic compaction · Event sourcing · Tool contracts
├── L1.5 Engine Layer    engine/         TurnRunner pipeline · Quality-gate hooks · Memory bridge
├── L2   Services
│         router/                        LLM difficulty routing (T0–T3) · Confidence gate · Anti-downgrade
│         memory/                        L1 in-process · L2 SQLite sessions · L3 FTS5 full-text index
│         agents/                        AgentCard registry · Supervisor · Debate mode · DAG scheduling
│         provider/                      LLM abstraction (Anthropic / OpenAI) · Retry · Failover
│         channels/                      Envelope abstraction · Lane isolation · Telegram / Discord
│         tools/                         Built-in tools: calculator · current_datetime · web_search
│         skills/                        TOML + Markdown skill definitions · Hot-swap · 3-tier priority
│         mcp/                           MCP client · stdio + SSE transport · tool namespace
└── L3   Access
          observability/                 OTel tracing bridge · Golden signals · Gradio dashboard
          nano/                          Personal assistant · Persona · ~/.nano/ config · one-shot mode
          cli.py                         nanoharness developer CLI
          server.py                      OpenAI-compatible HTTP API  (/v1/chat/completions)
          scripts/                       Benchmarks · Load tests
```

---

## Highlights 核心亮点

### LLM Difficulty Routing — 90% accuracy, 20.6% cost savings
### 智能难度路由

Four model tiers (T0–T3) selected by a single cheap classifier call. The routing pipeline chains two policy stages: **confidence gate** (auto-upgrade on low confidence) + **anti-downgrade** (30-min session cache prevents quality regression). A three-level fallback chain (LLM → keyword heuristic → default tier) ensures the classifier's own failure never blocks a request.

四档模型分级（T0–T3），由一次廉价分类 LLM 调用选档。路由策略串联两阶段：低置信度自动升档 + 30 分钟缓存防降档。三级降级链保证分类器超时不影响主流程。

```
Benchmark (benchmark_router.py · 40-sample test set)
  Accuracy          90.0%   (keyword-rule baseline: 65%)
  Cost savings      20.6%   (vs. single top-tier model)
  Penalty factor     3      (lower is better; baseline: 13)
```

---

### Semantic Context Compaction — turn-boundary safe, CJK-aware
### 语义贡献度压缩

Messages are scored by contribution weight (tool results **0.8** > user messages **0.6** > assistant text **0.4**) with position decay, then trimmed from least valuable first. A `retreat_to_turn_boundary` step ensures the cut never lands inside a `tool_use` / `tool_result` pair — splitting pairs causes API errors. Token estimation is CJK-aware (Chinese averages ~2 chars/token vs. ~4 for ASCII).

按语义贡献度评分后裁剪低价值内容，turn-boundary 保护不切断工具配对，token 估算对中文进行特殊处理（~2 字符/token）。

---

### Execution Flow Control — intervention before hard stop
### 执行流控制：干预优先于硬停

Beyond `max_iter` / `max_tool_calls`, NanoHarness adds two orthogonal stuck-detection layers:

- **Call fingerprint** (`tool_name + args_hash`, counted across the whole turn): catches exact repetition and A→B→A→B oscillation.
- **Per-tool budget** (`max_calls_per_tool`): catches same-tool retries with varying arguments that fingerprinting misses.

On trigger, the engine **does not raise** — it injects a recovery hint, hides the offending tool from `tool_definitions`, and lets the model attempt a graceful finish. Only on a second budget hit does it strip all tools for one final answer attempt; hard-stop is the last resort.

Termination cause is reported in `DoneEvent.stop_reason` (6 variants) + `outcome` (3 classes) — observable and testable, not buried in `final_text`.

在 max_iter / max_tool_calls 之外，额外两道卡死检测：调用指纹（抓重复与振荡）+ per-tool 调用预算（抓换参数钻空子）。触发时不直接停止，而是注入恢复指令、动态禁用该工具，给模型一次换方法收尾的机会。两阶段优雅收尾后才硬停。

---

### Behavioral Fingerprint Tests
### 行为指纹测试框架

Test suites assert *behavioral paths*, not output text — LLM output is non-deterministic and text assertions break on model updates.

`BehaviorFingerprint` distinguishes:
- `tools_called` — tools the LLM *requested* (from `ToolCallStart` events)
- `tools_executed` — tools that *actually ran* (from `ToolCallResult` events)

Security assertions take the form `must_not_execute_tools ∩ tools_executed = ∅`. Even if a prompt-injection tricks the LLM into requesting a dangerous tool, the execution layer can block it — and the test verifies that boundary holds.

测试断言行为路径而非输出文字。区分"LLM 请求的工具"与"执行层实际运行的工具"，安全断言验证执行层拦截能力，而非 LLM 输出。

---

### Skill System — TOML + Markdown, hot-swap at runtime
### 技能系统

Skills are plain files (`*.toml` or `*.md`) declaring a tool whitelist, a system-prompt patch, and capability tags for orchestrator routing. Three-tier priority loading: `builtin/ → ~/.nanoharness/skills/ → ./skills/` (later overrides earlier). TOML takes precedence over Markdown on name collision.

Runtime hot-swap: `/skill researcher` switches the active skill mid-conversation; the next turn picks up the new tool set and prompt patch immediately. No restart required.

技能文件（TOML 或 Markdown）声明工具白名单、提示词补丁与能力标签。三层优先级加载，运行时 `/skill NAME` 热换，下一个 turn 立即生效。

---

### Multi-Agent Orchestration — true isolation, DAG scheduling
### 多 Agent 编排

Supervisor mode: decompose task → route by capability tags → `asyncio.gather` parallel execution → synthesize. Debate mode: two Reviewers run with entirely separate `session_key + AgentContext` — they cannot see each other's history. Judge identifies disagreement rather than averaging.

Task dependency DAG (`SubtaskSpec.depends_on`): tasks with no dependencies run in parallel; tasks with dependencies wait on `asyncio.Event`. DFS cycle detection prevents deadlock.

`ContextVar _ORCHESTRATION_DEPTH` prevents infinite recursion when an agent tool calls back into the orchestrator.

Supervisor 模式：拆解→路由→并行→综合。辩论模式：两个 Reviewer 物理隔离 session，无法看到彼此历史。任务依赖 DAG 实现拓扑调度，ContextVar 防嵌套递归。

---

## Nano — Personal Assistant 个人助手

Nano is a personal AI assistant built on top of NanoHarness. It uses the full engine stack (routing, tools, skills, memory) and comes ready to use out of the box.

Nano 是构建在 NanoHarness 之上的个人 AI 助手，开箱即用。

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

nano init                       # create ~/.nano/config.toml and ~/.nano/skills/
nano                            # start interactive chat
nano -p "what time is it"       # one-shot query, prints response and exits
nano --skill researcher         # start with researcher skill active
nano skills                     # list available skills
```

**Customise** — edit `~/.nano/config.toml`:

```toml
[assistant]
name = "Nano"
# system_prompt = "You are a ..."   # override persona

[routing]
tier = "auto"   # or T0 / T1 / T2 / T3

[defaults]
# skill = "researcher"   # activate a skill on every startup
```

**Personal skills** — drop a `.toml` or `.md` file in `~/.nano/skills/`:

```markdown
---
name: my-analyst
description: Data analysis specialist
tools: [calculator, web_search]
---
You are a rigorous data analyst. Show your work step by step.
```

**MCP tools** — create `~/.nano/mcp.json` to connect any [Model Context Protocol](https://modelcontextprotocol.io) server:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/docs"]
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    },
    "my-api": {
      "type": "sse",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

MCP tools are discovered at startup and appear alongside builtins. Tool names are namespaced as `{server}__{tool_name}`. Install MCP support with `pip install 'nanoharness[mcp]'`.

MCP 工具在启动时自动发现，与内置工具并列提供。工具名格式：`{server名}__{工具名}`。

---

## Quick Start 快速开始

```bash
# Install / 安装
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

# Interactive chat / 交互式对话
nanoharness chat                      # auto-routing T0–T3, built-in tools
nanoharness chat --tier T2            # force a specific tier
nanoharness chat --no-tools           # pure conversation mode
nanoharness chat --skill researcher   # activate a skill (tool filter + prompt patch)
nanoharness skills                    # list available skills

# OpenAI-compatible API server / OpenAI 兼容服务
nanoharness serve --port 8080         # http://localhost:8080/docs

# Connect any OpenAI client / 接入示例
# from openai import OpenAI
# client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
# client.chat.completions.create(model="nanoharness", messages=[...])

# Benchmarks (--mock mode, no API key needed) / 量化 benchmark
python scripts/benchmark_router.py          # routing accuracy + cost savings
python scripts/benchmark_compaction.py      # compaction token reduction

# Tests / 测试
python -m pytest -v                         # 225 tests

# Load test / 压测
python scripts/load_test_gateway.py --concurrency 50 --rounds 3

# Observability dashboard / 可观测面板  (http://localhost:7860)
python -m nanoharness.observability.dashboard
```

**Writing a custom skill** — drop a file in `./skills/`:

```toml
# skills/my_analyst.toml
name = "analyst"
description = "Data analysis assistant"
tools = ["calculator", "web_search"]
tier = "T2"
system_prompt_patch = "You are a rigorous data analyst. Always show your work."
capabilities = ["analysis"]
```

Or Markdown with frontmatter (compatible with Claude Code / OpenHarness skill format):

```markdown
---
name: analyst
description: Data analysis assistant
tools: [calculator, web_search]
tier: T2
---
You are a rigorous data analyst. Always show your work.
```

---

## Benchmarks 量化数据

| Metric | Source | Result |
|--------|--------|--------|
| Routing accuracy | `benchmark_router.py` · 40-sample set | **90.0%** (keyword-rule baseline: 65%) |
| Cost savings | `benchmark_router.py` | **20.6%** vs. single top-tier model |
| Penalty factor | `benchmark_router.py` | **3** (lower is better; baseline: 13) |
| Gateway P99 latency | `load_test_gateway.py` · 50 concurrent | **≈32 ms** (lane layer, LLM latency excluded) |

---

## Tech Stack 技术栈

| Layer | Choice | Reason |
|-------|--------|--------|
| Runtime | Python 3.12 / asyncio | IO-bound workloads; single-process sufficient |
| LLM | Anthropic SDK | Streaming + extended thinking (T3) |
| Storage | SQLite + FTS5 trigram | Zero extra infra; Chinese full-text search built-in |
| Channels | aiogram 3.x / discord.py | Native async IM adapters |
| Observability | Self-implemented + OTel bridge | Lightweight; no heavy SDK required |
| Dashboard | Gradio | Minimal frontend code |
| Tests | pytest + behavioral fingerprints | Behavior-path assertions, not text matching |

---

## Layer Dependency Rules 层间依赖

```
core.*      ←  provider.* only — no imports from engine / memory / channels / agents
engine.*    ←  core.* — no imports from channels / memory / agents
provider.*  ←  core.context (Message type) only
router.*    ←  provider.* — no imports from channels
memory.*    ←  core.* — no imports from channels / agents
agents.*    ←  core.* + engine.*
channels.*  ←  all layers (outermost)
```

---

## GitHub

https://github.com/Wuzheyuan456/nanoharness

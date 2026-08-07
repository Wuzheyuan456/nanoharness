<p align="center">
  <a href="README.md"><strong>English</strong></a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

# NanoHarness

> **Lightweight multi-agent orchestration engine — built from scratch, zero framework dependencies.**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/tests-253_passing-brightgreen?logo=pytest&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
</p>

NanoHarness implements a complete agent harness stack — hand-written ReAct state machine, LLM difficulty router, three-tier memory, multi-agent orchestration, multi-channel gateway, and observability — without depending on LangChain, LangGraph, or any agent framework. Every component is purpose-built and fully transparent.

---

## Architecture

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
│         channels/                      Envelope abstraction · Lane isolation · Telegram / Discord / Feishu
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

## Highlights

### LLM Difficulty Routing — 90% accuracy, 20.6% cost savings

Four model tiers (T0–T3) selected by a single cheap classifier call. The routing pipeline chains two policy stages: **confidence gate** (auto-upgrade on low confidence) + **anti-downgrade** (30-min session cache prevents quality regression). A three-level fallback chain (LLM → keyword heuristic → default tier) ensures the classifier's own failure never blocks a request.

```
Benchmark (benchmark_router.py · 40-sample test set)
  Accuracy          90.0%   (keyword-rule baseline: 65%)
  Cost savings      20.6%   (vs. single top-tier model)
  Penalty factor     3      (lower is better; baseline: 13)
```

---

### Semantic Context Compaction — turn-boundary safe, CJK-aware

Messages are scored by contribution weight (tool results **0.8** > user messages **0.6** > assistant text **0.4**) with position decay, then trimmed from least valuable first. A `retreat_to_turn_boundary` step ensures the cut never lands inside a `tool_use` / `tool_result` pair — splitting pairs causes API errors. Token estimation is CJK-aware (Chinese averages ~2 chars/token vs. ~4 for ASCII).

---

### Execution Flow Control — intervention before hard stop

Beyond `max_iter` / `max_tool_calls`, NanoHarness adds two orthogonal stuck-detection layers:

- **Call fingerprint** (`tool_name + args_hash`, counted across the whole turn): catches exact repetition and A→B→A→B oscillation.
- **Per-tool budget** (`max_calls_per_tool`): catches same-tool retries with varying arguments that fingerprinting misses.

On trigger, the engine **does not raise** — it injects a recovery hint, hides the offending tool from `tool_definitions`, and lets the model attempt a graceful finish. Only on a second budget hit does it strip all tools for one final answer attempt; hard-stop is the last resort.

Termination cause is reported in `DoneEvent.stop_reason` (6 variants) + `outcome` (3 classes) — observable and testable, not buried in `final_text`.

---

### Behavioral Fingerprint Tests

Test suites assert *behavioral paths*, not output text — LLM output is non-deterministic and text assertions break on model updates.

`BehaviorFingerprint` distinguishes:
- `tools_called` — tools the LLM *requested* (from `ToolCallStart` events)
- `tools_executed` — tools that *actually ran* (from `ToolCallResult` events)

Security assertions take the form `must_not_execute_tools ∩ tools_executed = ∅`. Even if a prompt-injection tricks the LLM into requesting a dangerous tool, the execution layer can block it — and the test verifies that boundary holds.

---

### Skill System — TOML + Markdown, hot-swap at runtime

Skills are plain files (`*.toml` or `*.md`) declaring a tool whitelist, a system-prompt patch, and capability tags for orchestrator routing. Three-tier priority loading: `builtin/ → ~/.nanoharness/skills/ → ./skills/` (later overrides earlier). TOML takes precedence over Markdown on name collision.

Runtime hot-swap: `/skill researcher` switches the active skill mid-conversation; the next turn picks up the new tool set and prompt patch immediately. No restart required.

---

### MCP Client — connect any external tool server, zero-restart discovery

Drop `~/.nano/mcp.json` to connect any [Model Context Protocol](https://modelcontextprotocol.io) server (filesystem, databases, APIs). Nano discovers tools at startup and adds them alongside builtins — no restart needed when you add a new server.

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/docs"] },
    "fetch":      { "command": "uvx", "args": ["mcp-server-fetch"] },
    "my-api":     { "type": "sse",  "url": "http://localhost:8000/mcp" }
  }
}
```

`AsyncExitStack` manages N server connections as one lifecycle — all connections close cleanly on session exit. Per-server failure isolation: an unreachable server is logged and skipped; the others start normally. Tool names are namespaced `{server}__{tool}` to prevent collisions. Graceful degradation: if the `mcp` package is absent, Nano starts with a one-line warning.

---

### Multi-Agent Orchestration — true isolation, DAG scheduling

Supervisor mode: decompose task → route by capability tags → `asyncio.gather` parallel execution → synthesize. Debate mode: two Reviewers run with entirely separate `session_key + AgentContext` — they cannot see each other's history. Judge identifies disagreement rather than averaging.

Task dependency DAG (`SubtaskSpec.depends_on`): tasks with no dependencies run in parallel; tasks with dependencies wait on `asyncio.Event`. DFS cycle detection prevents deadlock.

`ContextVar _ORCHESTRATION_DEPTH` prevents infinite recursion when an agent tool calls back into the orchestrator.

---

## Nano — Personal Assistant

Nano is a personal AI assistant built on top of NanoHarness. It uses the full engine stack (routing, tools, skills, MCP) and comes ready to use out of the box.

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

nano init                       # create ~/.nano/config.toml, ~/.nano/skills/, ~/.nano/mcp.json
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

**MCP tools** — add servers to `~/.nano/mcp.json` (install support with `pip install 'nanoharness[mcp]'`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/docs"]
    }
  }
}
```

---

## Quick Start

```bash
# Install
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

# Interactive chat
nanoharness chat                      # auto-routing T0–T3, built-in tools
nanoharness chat --tier T2            # force a specific tier
nanoharness chat --no-tools           # pure conversation mode
nanoharness chat --skill researcher   # activate a skill (tool filter + prompt patch)
nanoharness skills                    # list available skills

# OpenAI-compatible API server
nanoharness serve --port 8080         # docs at http://localhost:8080/docs

# Benchmarks (--mock mode, no API key needed)
python scripts/benchmark_router.py          # routing accuracy + cost savings
python scripts/benchmark_compaction.py      # compaction token reduction

# Tests
python -m pytest -v                         # 253 tests

# Load test
python scripts/load_test_gateway.py --concurrency 50 --rounds 3

# Observability dashboard  (http://localhost:7860)
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

## Benchmarks

| Metric | Source | Result |
|--------|--------|--------|
| Routing accuracy | `benchmark_router.py` · 40-sample set | **90.0%** (keyword-rule baseline: 65%) |
| Cost savings | `benchmark_router.py` | **20.6%** vs. single top-tier model |
| Penalty factor | `benchmark_router.py` | **3** (lower is better; baseline: 13) |
| Gateway P99 latency | `load_test_gateway.py` · 50 concurrent | **≈32 ms** (lane layer, LLM latency excluded) |

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Runtime | Python 3.12 / asyncio | IO-bound workloads; single-process sufficient |
| LLM | Anthropic SDK | Streaming + extended thinking (T3) |
| Storage | SQLite + FTS5 trigram | Zero extra infra; full-text search built-in |
| Channels | aiogram 3.x / discord.py / httpx (Feishu) | Native async IM adapters |
| Observability | Self-implemented + OTel bridge | Lightweight; no heavy SDK required |
| Dashboard | Gradio | Minimal frontend code |
| Tests | pytest + behavioral fingerprints | Behavior-path assertions, not text matching |

---

## Layer Dependency Rules

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

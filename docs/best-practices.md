# NanoHarness Developer Best Practices

NanoHarness is a lightweight multi-agent orchestration engine built without LangChain or LangGraph. Tools are plain Python functions returning typed `ToolResult` objects; skills are TOML or Markdown config files; the runtime is a single async generator. This guide covers the four things you touch most often: writing tools, writing skills, testing agent behavior, and wiring in MCP servers.

---

## Writing a Tool

A tool is a Python function that receives `tool_input: dict` and a context object, and returns a `ToolResult`. Never raise exceptions out of a tool function — the LLM receives no error context and cannot recover. Return `ToolResult` with `status=FAILURE` instead.

### Full example

```python
# tools/summarizer.py
from typing import Any
from core.tool_executor import ToolResult, ToolResultStatus

def summarize(tool_input: dict, _ctx: Any) -> ToolResult:
    text = tool_input.get("text", "").strip()
    max_words = tool_input.get("max_words", 100)

    if not text:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content="No text provided.",
            error_code="empty_input",
            next_action_hint="Ask the user to provide the text they want summarized.",
        )

    try:
        words = text.split()
        summary = " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            content=summary,
            next_action_hint="Present this summary to the user.",
        )
    except Exception as exc:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content=str(exc),
            error_code="summarize_failed",
            next_action_hint="Inform the user the summarization failed and ask them to try again.",
        )


SUMMARIZE_DEF = {
    "name": "summarize",
    "description": "Truncate text to a maximum word count.",
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to summarize."},
            "max_words": {"type": "integer", "description": "Maximum words to return (default 100)."},
        },
        "required": ["text"],
    },
}
```

### Async tools

Return a coroutine from the function body. `ToolExecutor` detects it via `asyncio.iscoroutine` and awaits it automatically.

```python
async def fetch_url(tool_input: dict, _ctx: Any) -> ToolResult:
    url = tool_input.get("url", "").strip()
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            body = await resp.text()
    return ToolResult(status=ToolResultStatus.SUCCESS, content=body)
```

### Modifying context inside a tool

`ToolContext` is a frozen dataclass. Direct attribute assignment raises `FrozenInstanceError`. Use `dataclasses.replace`:

```python
from dataclasses import replace

def set_locale(tool_input: dict, ctx: ToolContext) -> ToolResult:
    new_ctx = replace(ctx, locale=tool_input.get("locale", "en"))
    # use new_ctx downstream; do not mutate ctx
    return ToolResult(status=ToolResultStatus.SUCCESS, content="Locale updated.")
```

---

## Writing a Skill

A skill binds a name, a set of tools, a tier, capability tags, and an optional system prompt patch. Use TOML for simple skills; Markdown frontmatter for skills that need longer prompt patches inline.

### TOML skill

```toml
# skills/researcher.toml
name = "researcher"
description = "Searches the web and synthesizes findings."
tools = ["web_search", "current_datetime"]
tier = "T2"
capabilities = ["search", "research", "fact-check"]
system_prompt_patch = """
You are a rigorous researcher. Cite sources. Never fabricate data.
When the user asks for current information, always call current_datetime first.
"""
```

### Markdown skill

```markdown
---
name: analyst
tools: [calculator, web_search]
tier: T2
capabilities: [analysis, math, research]
---

You are a quantitative analyst. Use the calculator for all numeric operations
rather than computing in your head. Show your work step by step.
```

### Key fields

| Field | Required | Notes |
|---|---|---|
| `name` | yes | Unique skill identifier used for routing |
| `tools` | yes | List of tool names this skill is allowed to call |
| `tier` | yes | Supervisor uses tier for priority ordering (`T1` > `T2` > `T3`) |
| `capabilities` | yes | Tags the Supervisor matches against when routing by capability; **leave empty and the skill is invisible to capability-based routing** |
| `description` | recommended | Used in Supervisor prompts to describe the skill |
| `system_prompt_patch` | optional | Appended to the base system prompt when this skill is active |

---

## Testing Your Agent

Behavioral tests use `BehaviorFingerprint` to record which tools were actually called during a turn, then assert constraints against that fingerprint. This catches regressions without mocking LLM responses.

### Basic pattern

```python
# tests/behavioral/test_summarizer.py
import pytest
from tests.behavioral.fingerprint import BehaviorFingerprint, BehaviorConstraint, run_and_fingerprint

@pytest.mark.asyncio
async def test_summarizer_is_called():
    fp: BehaviorFingerprint = await run_and_fingerprint(
        prompt="Summarize this paragraph for me: ...",
        provider="anthropic",
        tools=["summarize"],
        tool_defs=[SUMMARIZE_DEF],
    )

    BehaviorConstraint(
        must_execute_tools={"summarize"},
        must_not_execute_tools={"web_search"},
        call_count_min=1,
        call_count_max=2,
    ).assert_satisfied(fp)
```

### Constraint fields

| Field | Type | Meaning |
|---|---|---|
| `must_execute_tools` | `set[str]` | Every tool in this set must appear in the fingerprint |
| `must_not_execute_tools` | `set[str]` | None of these tools may appear |
| `call_count_min` | `int` | Minimum total tool calls across the turn |
| `call_count_max` | `int` | Maximum total tool calls across the turn |

### Running the turn manually

`NanoCore.run_turn()` is an **async generator**, not a coroutine. Use `async for`, never `await`:

```python
# correct
async for event in core.run_turn("What is 2+2?"):
    handle(event)

# wrong — raises TypeError
result = await core.run_turn("What is 2+2?")
```

### Blocking tools at runtime

Pass `denied_tools` to prevent specific tools from being called during a turn without modifying the skill:

```python
async for event in core.run_turn(msg, denied_tools={"web_search", "file_write"}):
    ...
```

---

## Common Pitfalls

| Pitfall | What happens | Fix |
|---|---|---|
| Raising an exception instead of returning `ToolResult(status=FAILURE, ...)` | The exception propagates up; the LLM receives no error content and cannot adjust its plan | Wrap all fallible logic in `try/except` and return a `FAILURE` result with `error_code` and `next_action_hint` |
| Assigning directly to a `ToolContext` field | `FrozenInstanceError` at runtime because `ToolContext` is a frozen dataclass | Use `dataclasses.replace(ctx, field=new_value)` |
| Awaiting `run_turn()` with `await` | `TypeError: object async_generator can't be used in 'await' expression` | Use `async for event in core.run_turn(msg)` |
| Leaving `next_action_hint` empty on `FAILURE` results | The LLM may blindly retry the same failing tool with the same inputs | Always set `next_action_hint` on failures; tell the model what to do instead |
| Empty `capabilities` list on a skill | The Supervisor cannot route to this skill by capability matching; it becomes a dead skill | Add at least one capability tag that describes what the skill does |
| Using `{server}_{tool}` (single underscore) for MCP tool names | Tool lookup fails silently; the LLM sees the tool as unavailable | MCP namespacing uses double underscore: `{server}__{tool}` |

---

## MCP Integration

NanoHarness connects to MCP servers via `McpClient`, which manages an `AsyncExitStack` for clean shutdown. Tool names from MCP are automatically namespaced as `{server}__{tool}` (double underscore). You cannot rename or override this convention.

### Config example

```python
# config/mcp_servers.py
MCP_SERVERS = [
    {
        "name": "filesystem",          # becomes filesystem__read_file, filesystem__write_file, etc.
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"],
    },
    {
        "name": "postgres",
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
    },
]
```

### Using MCP tools in a skill

Reference the namespaced name directly in the `tools` list:

```toml
name = "data-analyst"
tools = ["filesystem__read_file", "postgres__query", "calculator"]
tier = "T2"
capabilities = ["data", "sql", "analysis"]
```

### Startup and shutdown

`McpClient` uses `AsyncExitStack` internally. Always start and stop it within an async context; avoid forcing shutdown mid-turn or connections leak:

```python
async with McpClient(MCP_SERVERS) as client:
    tools, tool_defs = await client.get_tools()
    async for event in core.run_turn(msg):
        ...
# all MCP subprocesses are cleanly terminated here
```

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md"><strong>简体中文</strong></a>
</p>

# NanoHarness

> **轻量级多智能体编排引擎——从零手写，不依赖任何框架。**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/测试-253_通过-brightgreen?logo=pytest&logoColor=white" alt="Tests">
  <img src="https://img.shields.io/badge/协议-MIT-yellow" alt="License">
</p>

NanoHarness 实现了完整的 Agent 基础设施栈——手写 ReAct 状态机、LLM 难度路由、三层记忆、多 Agent 编排、多通道网关与可观测性，不依赖 LangChain / LangGraph / AutoGen 等任何框架。每个组件都是专门构建的，完全透明可审查。

---

## 架构

```
NanoHarness
│
├── L1   核心引擎     core/           ReAct 异步生成器状态机
│                                     语义压缩 · 事件溯源 · 工具契约
├── L1.5 引擎层       engine/         TurnRunner 流水线 · 质量门控 Hook · 记忆桥接
├── L2   服务层
│         router/                     LLM 难度路由（T0–T3）· 置信度门 · 防降档
│         memory/                     L1 进程内 · L2 SQLite 会话 · L3 FTS5 全文索引
│         agents/                     AgentCard 注册表 · Supervisor · 辩论模式 · DAG 调度
│         provider/                   LLM 抽象（Anthropic / OpenAI）· 重试 · 故障转移
│         channels/                   信封抽象 · 车道隔离 · Telegram / Discord / 飞书
│         tools/                      内置工具：calculator · current_datetime · web_search
│         skills/                     TOML + Markdown 技能定义 · 热换 · 三级优先级
│         mcp/                        MCP 客户端 · stdio + SSE 传输 · 工具命名空间
└── L3   接入层
          observability/              OTel 追踪桥接 · 黄金信号 · Gradio 面板
          nano/                       个人助手 · Persona · ~/.nano/ 配置 · 一次性模式
          cli.py                      nanoharness 开发者 CLI
          server.py                   OpenAI 兼容 HTTP API（/v1/chat/completions）
          scripts/                    Benchmark · 压测脚本
```

---

## 核心亮点

### LLM 难度路由——准确率 90%，成本节省 20.6%

四档模型分级（T0–T3），由一次廉价分类 LLM 调用选档。路由策略串联两阶段：**置信度门**（低置信度自动升档）+ **防降档**（30 分钟会话缓存防止质量回退）。三级降级链（LLM → 关键词启发式 → 默认档位）确保分类器自身失败时不阻塞请求。

```
Benchmark（benchmark_router.py · 40 条测试集）
  准确率         90.0%   （关键词规则基线：65%）
  成本节省       20.6%   （vs. 全用高档模型）
  惩罚系数        3      （越低越好；基线：13）
```

---

### 语义贡献度压缩——turn-boundary 安全，CJK 感知

消息按贡献权重评分（工具结果 **0.8** > 用户消息 **0.6** > 助手文本 **0.4**）加位置衰减，从价值最低处开始裁剪。`retreat_to_turn_boundary` 步骤确保裁剪点永不落在 `tool_use` / `tool_result` 配对中间——切断配对会导致 API 报错。Token 估算对 CJK 敏感（中文约 2 字符/token，ASCII 约 4 字符/token）。

---

### 执行流控制——干预优先于硬停

在 `max_iter` / `max_tool_calls` 之外，NanoHarness 额外增加两道正交的卡死检测：

- **调用指纹**（`tool_name + args_hash`，统计整个 turn）：捕获精确重复和 A→B→A→B 振荡。
- **per-tool 调用预算**（`max_calls_per_tool`）：捕获换参数钻空子的同工具重复调用。

触发时引擎**不抛异常**——注入恢复指令，从 `tool_definitions` 中隐藏该工具，让模型有机会优雅收尾。第二次预算触发才剥离所有工具做最后一次答题；硬停是最后手段。

终止原因在 `DoneEvent.stop_reason`（6 种）+ `outcome`（3 类）中上报——可观测、可测试，不埋进 `final_text`。

---

### 行为指纹测试框架

测试套件断言*行为路径*，而非输出文字——LLM 输出是不确定的，文字断言在模型更新后就会失效。

`BehaviorFingerprint` 区分两层：
- `tools_called` — LLM *请求*的工具（来自 `ToolCallStart` 事件）
- `tools_executed` — *实际运行*的工具（来自 `ToolCallResult` 事件）

安全断言形如 `must_not_execute_tools ∩ tools_executed = ∅`。即使 prompt 注入诱导 LLM 请求危险工具，执行层也可以拦截——测试验证的是这道边界，而非 LLM 的输出内容。

---

### 技能系统——TOML + Markdown，运行时热换

技能文件（`*.toml` 或 `*.md`）声明工具白名单、System Prompt 补丁和能力标签供编排器路由。三级优先级加载：`builtin/ → ~/.nanoharness/skills/ → ./skills/`（后者覆盖前者）。同名冲突时 TOML 优先于 Markdown。

运行时热换：`/skill researcher` 在会话中途切换技能，下一个 turn 立即生效新工具集和提示词补丁，无需重启。

---

### MCP 客户端——连接任意外部工具服务器，零重启发现

在 `~/.nano/mcp.json` 中配置任意 [Model Context Protocol](https://modelcontextprotocol.io) 服务器（文件系统、数据库、API 等），Nano 启动时自动发现并加入内置工具列表。

```json
{
  "mcpServers": {
    "filesystem": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/docs"] },
    "fetch":      { "command": "uvx", "args": ["mcp-server-fetch"] },
    "my-api":     { "type": "sse",  "url": "http://localhost:8000/mcp" }
  }
}
```

`AsyncExitStack` 将 N 个 server 连接作为单一生命周期管理——会话退出时全部干净关闭。per-server 失败隔离：不可达的 server 记录日志后跳过，其他 server 正常启动。工具名格式 `{server}__{tool}` 防命名碰撞。若 `mcp` 包未安装，Nano 一行警告后正常启动。

---

### 多 Agent 编排——真隔离，DAG 调度

Supervisor 模式：拆解任务 → 按能力标签路由 → `asyncio.gather` 并行执行 → 综合结果。辩论模式：两个 Reviewer 使用完全独立的 `session_key + AgentContext` 运行，彼此无法看到对方历史。Judge 识别分歧而非取均值。

任务依赖 DAG（`SubtaskSpec.depends_on`）：无依赖的任务并行运行，有依赖的任务等待 `asyncio.Event`。DFS 环检测防止死锁。

`ContextVar _ORCHESTRATION_DEPTH` 防止 agent 工具回调编排器时的无限递归。

---

## Nano——个人助手

Nano 是构建在 NanoHarness 之上的个人 AI 助手，使用完整引擎栈（路由、工具、技能、MCP），开箱即用。

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

nano init                       # 创建 ~/.nano/config.toml、~/.nano/skills/、~/.nano/mcp.json
nano                            # 启动交互式对话
nano -p "现在几点"              # 一次性查询，打印结果后退出
nano --skill researcher         # 启动时激活 researcher 技能
nano skills                     # 列出所有可用技能
```

**自定义配置** — 编辑 `~/.nano/config.toml`：

```toml
[assistant]
name = "Nano"
# system_prompt = "你是一个..."   # 覆盖 persona

[routing]
tier = "auto"   # 或 T0 / T1 / T2 / T3

[defaults]
# skill = "researcher"   # 每次启动时激活某个技能
```

**个人技能** — 在 `~/.nano/skills/` 中放一个 `.toml` 或 `.md` 文件：

```markdown
---
name: my-analyst
description: 数据分析专家
tools: [calculator, web_search]
---
你是一个严谨的数据分析师。每一步都要展示推导过程。
```

**MCP 工具** — 在 `~/.nano/mcp.json` 中添加服务器（先安装：`pip install 'nanoharness[mcp]'`）：

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

## 快速开始

```bash
# 安装
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

# 交互式对话
nanoharness chat                      # 自动路由 T0–T3，内置工具
nanoharness chat --tier T2            # 强制指定档位
nanoharness chat --no-tools           # 纯对话模式
nanoharness chat --skill researcher   # 激活技能（工具过滤 + 提示词补丁）
nanoharness skills                    # 列出可用技能

# OpenAI 兼容 API 服务
nanoharness serve --port 8080         # 文档见 http://localhost:8080/docs

# Benchmark（--mock 模式，无需 API Key）
python scripts/benchmark_router.py          # 路由准确率 + 成本节省
python scripts/benchmark_compaction.py      # 压缩 token 减少量

# 测试
python -m pytest -v                         # 253 个测试

# 压测
python scripts/load_test_gateway.py --concurrency 50 --rounds 3

# 可观测面板（http://localhost:7860）
python -m nanoharness.observability.dashboard
```

**编写自定义技能** — 在 `./skills/` 放一个文件：

```toml
# skills/my_analyst.toml
name = "analyst"
description = "数据分析助手"
tools = ["calculator", "web_search"]
tier = "T2"
system_prompt_patch = "你是一个严谨的数据分析师，每步都要展示推导过程。"
capabilities = ["analysis"]
```

或使用 Markdown frontmatter 格式（兼容 Claude Code / OpenHarness 技能格式）：

```markdown
---
name: analyst
description: 数据分析助手
tools: [calculator, web_search]
tier: T2
---
你是一个严谨的数据分析师，每步都要展示推导过程。
```

---

## 量化数据

| 指标 | 数据来源 | 结果 |
|------|---------|------|
| 路由准确率 | `benchmark_router.py` · 40 条测试集 | **90.0%**（关键词规则基线：65%）|
| 成本节省 | `benchmark_router.py` | **20.6%** vs. 全用高档模型 |
| 惩罚系数 | `benchmark_router.py` | **3**（越低越好；基线：13）|
| Gateway P99 延迟 | `load_test_gateway.py` · 50 并发 | **≈32 ms**（车道层，不含 LLM 延迟）|

---

## 技术栈

| 层 | 选型 | 理由 |
|----|------|------|
| 运行时 | Python 3.12 / asyncio | IO 密集型工作负载，单进程够用 |
| LLM | Anthropic SDK | 流式输出 + extended thinking（T3）|
| 存储 | SQLite + FTS5 trigram | 零额外基础设施；内置全文搜索 |
| 通道 | aiogram 3.x / discord.py / httpx（飞书）| 原生异步 IM 适配器 |
| 可观测性 | 自实现 + OTel 桥接 | 轻量；不需要重量级 SDK |
| 面板 | Gradio | 最少前端代码 |
| 测试 | pytest + 行为指纹 | 断言行为路径，不断言文字输出 |

---

## 层间依赖规则

```
core.*      ←  仅 provider.*——不 import engine / memory / channels / agents
engine.*    ←  core.*——不 import channels / memory / agents
provider.*  ←  仅 core.context（Message 类型）
router.*    ←  provider.*——不 import channels
memory.*    ←  core.*——不 import channels / agents
agents.*    ←  core.* + engine.*
channels.*  ←  所有层（最外层）
```

---

## GitHub

https://github.com/Wuzheyuan456/nanoharness

# NanoHarness — Claude 上下文说明

## 项目定位

秋招代表作项目。**轻量级多智能体编排引擎**，手写 ReAct 运行时，不依赖 LangChain / LangGraph / DeepAgent 等框架。

核心卖点：面试官问任何实现细节都能答上来，因为每一行都是自己写的。

参考项目：`/Users/zheyuan.wu/project/opensquilla-main/`（生产级 Python Agent harness，Phase 0 阶段分析过）

---

## 三层架构

```
L1 内核层  nanoharness/core/          — ReAct 状态机引擎（面试核心竞争力）
L1.5 引擎层 nanoharness/engine/        — TurnRunner、分阶段流水线、Hooks 质量门
L2 服务层  nanoharness/router/        — LLM 难度路由（T0~T3 档位）
           nanoharness/memory/        — SQLite + sqlite-vec 记忆系统
           nanoharness/agents/        — Agent Card 注册、Supervisor、Debate 模式
           nanoharness/provider/      — LLM Provider 抽象（Anthropic / OpenAI）
L3 接入层  nanoharness/channels/      — Telegram / Discord 多通道网关
           nanoharness/observability/ — OTel 追踪 + Gradio 可视化面板
```

---

## 开发环境

```bash
# 激活 conda 环境
conda activate nanoharness        # Python 3.12

# 跑单测
python -m pytest tests/unit/ -v

# 跑所有测试
python -m pytest -v
```

Python 路径：`/Users/zheyuan.wu/miniconda3/envs/nanoharness/bin/python`

---

## 当前进度

### Phase 1 ✅ 已完成（核心引擎）

| 文件 | 状态 | 说明 |
|------|------|------|
| `core/context.py` | ✅ | AgentState / ToolContext(frozen) / AgentContext / TurnContext |
| `core/nano_core.py` | ✅ | 手写 ReAct while 循环，非递归 async generator，~320 行 |
| `core/event_store.py` | ✅ | 8 种 Event 类型，append-only，trace_id 索引 |
| `core/compaction.py` | ✅ | 语义重要度评分 + turn-boundary 保护 + CompactionConfig |
| `core/tool_executor.py` | ✅ | ToolRegistry + DeadLoopDetector + 指数退避重试 |
| `provider/base.py` | ✅ | LLMProvider Protocol + ProviderErrorType 分类 |
| `provider/anthropic.py` | ✅ | 流式 + extended thinking (T3) |
| `engine/hooks/types.py` | ✅ | TurnHook / ToolHook / CompactionHook Protocol |
| `engine/hooks/defaults.py` | ✅ | 默认空实现 + safe_call() |
| `tests/unit/test_core.py` | ✅ | 4 个行为指纹测试，全部通过 |

### Phase 2 ⬜ 待开发（路由 + TurnRunner）

- `engine/turn_runner.py` — per-session asyncio.Lock + ContextVar 重入检测
- `router/tiers.py` — T0/T1/T2/T3 模型档位定义
- `router/llm_router.py` — LLM-based 难度分类器（用一次 T0 级别的 LLM call 打标签）
- `router/decision_log.py` — 路由决策持久化 SQLite

### Phase 3 ⬜ 记忆系统

- `memory/store.py` — SQLite + sqlite-vec 存储层
- `memory/retrieval.py` — BM25 + 向量检索 + RRF 重排
- `memory/compaction_hooks.py` / `consolidation.py` / `manager.py`

### Phase 4~7 ⬜ 多 Agent / 通道 / 可观测性 / 测试完善

---

## 关键设计决策（不要改动）

### 1. NanoCore 是 async generator，不是普通 coroutine
`run_turn()` 用 `async for event in core.run_turn(msg)` 消费，不能 `await`。
provider 的 `stream()` 同理，不要加 `await`。

### 2. ToolContext 必须保持 frozen
并发安全依赖不可变性。需要修改字段时用 `replace(ctx, budget_limit=5)`，不要改成普通 dataclass。

### 3. AgentEvent 子类的字段顺序
`AgentEvent` 基类把 `kind` 放在有默认值的位置（在 `trace_id/session_key/agent_id` 之后）。
子类覆盖 `kind` 时必须也给默认值，否则 Python 3.12 dataclass 继承会报 `non-default follows default`。

### 4. CompactionEngine 与主模型解耦
压缩用 `CompactionConfig.compaction_model`（默认 haiku），主对话模型由 `AgentContext.model_id` 决定。两者独立配置，不要合并。

### 5. Provider 错误归一化
`CONTEXT_TOO_LONG` 不是真错误，是触发压缩的信号，在 `_call_provider` 里捕获后调 `_do_compact()` 再重试，对 ReAct 循环透明。

### 6. 测试风格：行为指纹，不断言文字
测试断言状态转换路径、工具调用次数、事件类型。不用 `assertEqual(done.final_text, "...")` 这类脆弱断言。

---

## 代码规范

- 语言：注释、commit message、文档全部用**中文**
- 格式：`ruff` 检查，line-length=100，target Python 3.12
- 类型：启用 `from __future__ import annotations`，Protocol 优先于 ABC
- 不加多余注释：只注释"为什么"，不注释"是什么"
- 不超前设计：当前 Phase 需要什么写什么，不预留未来扩展接口

---

## 层间依赖规则

```
core.*     ← 只依赖 provider.*，不 import engine/memory/channels/agents
engine.*   ← 可 import core.*，不 import channels/memory/agents
provider.* ← 只依赖 core.context（Message 类型），不 import 其他层
router.*   ← 可 import provider.*，不 import channels
memory.*   ← 可 import core.*，不 import channels/agents
agents.*   ← 可 import core.* + engine.*，通过 agent_factory 注入避免循环
channels.* ← 可 import 所有层，是最外层
```

---

## 重要文件索引

| 文件 | 作用 |
|------|------|
| `/Users/zheyuan.wu/project/项目规划.md` | 完整开发计划，7 个 Phase，里程碑和简历写法 |
| `/Users/zheyuan.wu/project/phase0_opensquilla分析.md` | opensquilla 代码分析 + 10 个关键设计决策 |
| `/Users/zheyuan.wu/project/phase1_面试总结.md` | Phase 1 面试话术，10 个技术点 + 快速问答卡 |
| `pyproject.toml` | 依赖声明，conda 环境名 `nanoharness` |
| `tests/unit/test_core.py` | 4 个核心行为指纹测试 |

---

## GitHub

仓库：https://github.com/Wuzheyuan456/nanoharness

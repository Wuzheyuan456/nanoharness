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
| `core/context.py` | ✅ | AgentState / StopReason / TurnOutcome / ToolContext(frozen) / AgentContext / TurnContext |
| `core/nano_core.py` | ✅ | 手写 ReAct while 循环，非递归 async generator，~320 行 |
| `core/event_store.py` | ✅ | 9 种 Event 类型（含 InterventionEvent），DoneEvent 带 stop_reason/outcome |
| `core/compaction.py` | ✅ | 语义重要度评分 + turn-boundary 保护 + CompactionConfig |
| `core/tool_executor.py` | ✅ | ToolRegistry + StuckDetector(per-签名计数) + ToolResult 契约 + 指数退避重试 |
| `provider/base.py` | ✅ | LLMProvider Protocol + ProviderErrorType 分类 |
| `provider/anthropic.py` | ✅ | 流式 + extended thinking (T3) |
| `engine/hooks/types.py` | ✅ | TurnHook / ToolHook / CompactionHook Protocol |
| `engine/hooks/defaults.py` | ✅ | 默认空实现 + safe_call() |
| `tests/unit/test_core.py` | ✅ | 7 个测试（含工具返回契约），全部通过 |

### Phase 2 ✅ 已完成（路由 + TurnRunner + Provider 故障转移）

| 文件 | 状态 | 说明 |
|------|------|------|
| `router/tiers.py` | ✅ | T0~T3 档位、TierConfig、PromptPolicy、TierRegistry（运行时覆盖） |
| `router/llm_router.py` | ✅ | 一次 T0 LLM call 分类，降级链（LLM→超时→规则→fallback），JSON 容错；持有 `_session_tier_cache`，支持 `RoutingPolicy` |
| `router/decision_log.py` | ✅ | 路由决策 SQLite 持久化 + cost_savings_report() |
| `router/policy.py` | ✅ | `RoutingPolicy`：confidence gate（低置信度升档）+ anti-downgrade（30min KV cache 保护窗口）|
| `provider/selector.py` | ✅ | `ProviderSelector`：指数退避重试（jitter）+ fallback provider；实现 `LLMProvider` Protocol，TurnRunner 无感知 |
| `engine/turn_runner.py` | ✅ | per-session Lock 串行化 + ContextVar 重入检测 + Hook 触发；路由优先于 hook_ctx 创建（model_id 可传入 hook）|
| `engine/hooks/defaults.py` | ✅ | `InputSanitizationHook`（超长输入 WARNING）+ `TurnMetricsHook`（Phase 2↔7 桥接，写 latency/tokens/error）|
| `tests/unit/test_router.py` | ✅ | 16 个测试 + 4 个路由策略测试（共 20），含串并行验证、confidence gate、anti-downgrade |
| `tests/unit/test_provider.py` | ✅ | 5 个 ProviderSelector 测试：重试、failover、AUTH_INVALID 立即 re-raise、CONTEXT_TOO_LONG 透传 |
| `tests/unit/test_hooks.py` | ✅ | 4 个 hook 测试：InputSanitizationHook 超长/正常输入、TurnMetricsHook latency/error |

**设计决策**：`engine/stages/` 6 个空 stub 文件已删除（`TurnRunner._execute` 从未使用它们，是过度抽象的死代码，同 Phase 1 `DeadLoopDetector` 问题，果断删除，不超前设计）。

### Phase 3 ✅ 已完成（三层记忆系统）

| 文件 | 状态 | 说明 |
|------|------|------|
| `memory/store.py` | ✅ | SQLite + FTS5 trigram 全文索引，L2 sessions + L3 memories 双表 |
| `memory/retrieval.py` | ✅ | BM25×0.3 + 时间衰减×0.3 + importance×0.4 三路融合排序 |
| `memory/compaction_hooks.py` | ✅ | 压缩前保护 tool_result 和长消息写入 L3 |
| `memory/consolidation.py` | ✅ | Dream 机制：session 结束后异步 LLM 提炼摘要+事实 |
| `memory/manager.py` | ✅ | Facade 门面：prefetch / on_compact / flush 三个方法 |
| `tests/unit/test_memory.py` | ✅ | 33 个测试，全部通过 |

### Phase 4 ✅ 已完成（多 Agent 编排）

| 文件 | 状态 | 说明 |
|------|------|------|
| `agents/registry.py` | ✅ | AgentCard + AgentRegistry，能力标签路由，对齐 A2A 协议 |
| `agents/orchestrator.py` | ✅ | Supervisor 模式：拆解/路由/并行/综合，ContextVar 深度保护 |
| `agents/debate.py` | ✅ | 辩论模式：并行 Reviewer → Judge，独立 session 视角 |
| `tests/unit/test_agents.py` | ✅ | 28 个测试，全部通过 |

### Phase 5 ✅ 已完成（行为指纹测试框架）

| 文件 | 状态 | 说明 |
|------|------|------|
| `tests/behavioral/fingerprint.py` | ✅ | BehaviorFingerprint（called/executed 区分）+ BehaviorConstraint |
| `tests/behavioral/test_intent_routing.py` | ✅ | 8 个测试：LLMRouter 分类、降级、启发式、行为约束 |
| `tests/behavioral/test_safety.py` | ✅ | 8 个测试：未注册工具拦截、约束违规检测、prompt 注入 |

### Phase 6 ✅ 已完成（多通道消息网关）

| 文件 | 状态 | 说明 |
|------|------|------|
| `channels/base.py` | ✅ | InboundEnvelope/OutboundEnvelope 信封 + BaseChannel Protocol |
| `channels/lane_queue.py` | ✅ | 车道隔离：per-session 串行、跨 session 并行、ContextVar 重入检测 |
| `channels/router.py` | ✅ | BindingRule 声明式路由 + make_session_key（DM/群聊隔离） |
| `channels/gateway.py` | ✅ | Gateway 流水线：去重→安全→路由→车道分发，SafetyPolicy |
| `channels/telegram.py` | ✅ | aiogram 适配器，parse_message + 4096 分块发送 |
| `channels/discord.py` | ✅ | discord.py 适配器，parse_message + 2000 分块发送 |
| `tests/unit/test_channels.py` | ✅ | 29 个测试，全部通过 |

### Phase 7 ✅ 已完成（可观测性 + 量化数据 + 压测）

| 文件 | 状态 | 说明 |
|------|------|------|
| `observability/tracing.py` | ✅ | 轻量 OTel：Span 树 + ContextVar 父子继承 + build_tree 回放 |
| `observability/metrics.py` | ✅ | 四大黄金信号，自实现 Counter/Histogram/Gauge，render_prometheus |
| `observability/dashboard.py` | ✅ | Gradio 三 Tab：路由决策/链路回放/黄金信号 |
| `scripts/benchmark_router.py` | ✅ | LLMRouter 准确率 + cost 节省%，支持 mock/真实切换 |
| `scripts/benchmark_compaction.py` | ✅ | 压缩前后 token 对比，mock 模式降本 93.8% |
| `scripts/load_test_gateway.py` | ✅ | Gateway 50 并发压测 P50/P95/P99 + 车道隔离验证 |
| `tests/unit/test_observability.py` | ✅ | 14 个测试，全部通过 |
| `README.md` | ✅ | 架构图 + 启动 + 设计决策 + 量化数据 |

### Phase 8 ✅ 已完成（执行流深度控制）

动机：Phase 1 只有 `max_iter` / `max_tool_calls` 两道硬熔断（与 LangGraph 的 `recursion_limit` 同档），且 `DeadLoopDetector` 是死代码（`_execute_tool` 绕过了它）。本阶段补齐 `Agent 执行流的深度控制力.md` 的 4 个类别各一个机制，让终止原因可观测、卡死可干预、失败可优雅退出。刻意停在通用 harness 边界，不抄 opensquilla 的 coding-agent 专属检测器。

| 机制 | 文件 | 说明 |
|------|------|------|
| A. StuckDetector | `core/tool_executor.py` + `core/nano_core.py` | per-签名计数（catch 重复 + 振荡 A-B-A-B），接通主路径（修死代码），触发时跳过执行 + 注入恢复消息 + 禁用工具，不 raise |
| B. StopReason + TurnOutcome | `core/context.py` + `core/event_store.py` | 6 种 stop_reason + 3 种 outcome，DoneEvent 带字段，`classify_outcome()` 纯函数映射 |
| C. 两阶段优雅收尾 | `core/nano_core.py` | 撞 max_iter/max_tool_calls 首次：注入"别调工具直接答"指令 + 剥离工具再做一次；二次才硬停（opensquilla 招牌，LangGraph 没有）|
| D. 工具返回契约 | `core/tool_executor.py` + `core/nano_core.py` | `ToolResult(status, next_action_hint, error_code)`，裸 str 自动包装，序列化带 `[status:]`/`[next_action:]` 强指引 |
| E. 动态工具禁用 | `core/context.py` + `core/nano_core.py` | `TurnContext.denied_tools`，卡死/预算触发后从 tool_definitions 隐藏，给 Agent 退路 |
| F. per-tool 调用预算 | `core/nano_core.py` | 单工具调用次数超 `max_calls_per_tool` 触发，catch 同工具不同参数的钻空子（签名去重抓不到）|
| 测试 | `tests/behavioral/test_control.py` + `tests/unit/test_core.py` | 7 行为指纹测试 + 3 契约单测，全套 150 通过 |

> 诚实取舍：原计划的"字节级 provider 请求去重"在 NanoCore 语义下是死代码（每轮 append history，连续相同请求不可能），已用 per-tool 预算替换。

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

### 7. 执行流控制：干预优先于硬停
卡死检测 / per-tool 预算触发时**不 raise、不直接 force_done**，而是跳过本次执行 + 注入恢复消息 + 禁用该工具（从 tool_definitions 隐藏），让模型有机会换方法收尾。只有两阶段收尾二次撞线才硬停。终止原因走 `DoneEvent.stop_reason` + `outcome`，不塞进 `final_text`。`ToolContext` 保持 frozen，控制状态全放 `TurnContext`（可变）。

---

## 代码规范

- 语言：注释和 docstring 用**中英双语**（格式 `# 中文 / English`，中文在前英文括注）；commit message、文档正文用中文
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
| `/Users/zheyuan.wu/project/phase8_面试总结.md` | Phase 8 面试话术，执行流深度控制 7 技术点 + 三家对比 + 快速问答卡 |
| `pyproject.toml` | 依赖声明，conda 环境名 `nanoharness` |
| `tests/unit/test_core.py` | 7 个核心测试（含工具返回契约）|
| `tests/behavioral/test_control.py` | 7 个执行流控制行为指纹测试 |

---

## GitHub

仓库：https://github.com/Wuzheyuan456/nanoharness

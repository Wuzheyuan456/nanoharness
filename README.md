# NanoHarness

> 自研轻量级多智能体编排引擎与可观测网关。**不依赖任何 Agent 框架**（LangChain / LangGraph / AutoGen），从零手写 ReAct 运行时，并在其上架设智能模型路由、多 Agent 编排与多通道消息网关。

核心卖点：每一行代码都能在面试中解释清楚——因为都是自己写的。

---

## 架构总览

```
NanoHarness
│
├── L1  核心引擎层   core/             — 手写 ReAct 状态机、语义贡献度压缩、工具熔断、事件溯源
├── L1.5 编排层      engine/           — TurnRunner 分阶段流水线 + Hooks 质量门控
├── L2  服务层
│        router/                      — LLM-based 难度路由（T0~T3），成本可降级
│        memory/                      — 三层记忆：L1 内存 / L2 SQLite sessions / L3 FTS5 全文索引
│        agents/                      — AgentCard 注册（对齐 A2A）、Supervisor、辩论模式
│        provider/                    — LLM 抽象（Anthropic / OpenAI）
│        channels/                    — 多通道网关：信封抽象 + 车道隔离 + Telegram/Discord
└── L3  接入层
         observability/               — OTel 轻量追踪 + 四大黄金信号 + Gradio 面板
         scripts/                     — benchmark 量化数据 + Gateway 压测
```

---

## 七个 Phase

| Phase | 模块                          | 状态  | 一句话                                                              |
| ----- | --------------------------- | --- | ---------------------------------------------------------------- |
| 1     | `core/`                     | ✅   | 手写 ReAct async generator，语义贡献度压缩，事件溯源                            |
| 2     | `engine/` `router/`         | ✅   | TurnRunner 分阶段 + per-session 锁重入检测 + LLM 难度路由                    |
| 3     | `memory/`                   | ✅   | 三层记忆 + FTS5 trigram 中文检索 + 时间衰减 + Dream 巩固                       |
| 4     | `agents/`                   | ✅   | AgentCard/A2A + Supervisor 四步编排 + 辩论模式独立视角                       |
| 5     | `tests/behavioral/`         | ✅   | 行为指纹测试框架：断言行为约束，不断言输出文本                                          |
| 6     | `channels/`                 | ✅   | 多通道网关：信封抽象 + 车道隔离 + 声明式路由 + TG/Discord                           |
| 7     | `observability/` `scripts/` | ✅   | OTel 追踪 + 四大黄金信号 + Gradio 面板 + benchmark/压测                      |
| 8     | `core/`                     | ✅   | 执行流深度控制：StuckDetector + 两阶段收尾 + 工具契约 + 终止分类 + 动态禁用 + per-tool 预算 |

---

## 快速启动

```bash
# 环境（conda）
conda activate nanoharness        # Python 3.12

# 跑测试（150 个，含行为指纹测试）
python -m pytest -v

# 量化数据 benchmark（需 ANTHROPIC_API_KEY 跑真实数据，--mock 验证流程）
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/benchmark_router.py          # LLM Router 成本节省%
python scripts/benchmark_compaction.py       # 上下文压缩降本%

# Gateway 压测
python scripts/load_test_gateway.py --concurrency 50 --rounds 3

# 可视化面板（浏览器访问 http://localhost:7860）
python -m nanoharness.observability.dashboard
```

---

## 核心设计决策（面试弹药）

### 1. NanoCore 是 async generator，不是普通 coroutine
`run_turn()` 用 `async for event in core.run_turn(msg)` 消费。流式输出、事件溯源、外部观察者模式全靠这一层。

### 2. 语义贡献度压缩，不是朴素截断
每条 message 打分（工具结果 0.8 > 用户消息 0.6 > 助手文本 0.4，叠加位置衰减），turn-boundary 保护不切断 tool_use/tool_result 配对。

### 3. LLM-based 难度路由，不用本地 ONNX
一次 Haiku call 分类 T0~T3，超时降级到规则启发式，再降级到 fallback。相对全用 T1，成本节省 Y%（见 benchmark）。

### 4. ContextVar 一套原语复用三次
- TurnRunner `_LOCK_OWNER`：per-session turn 串行 + 重入检测
- Orchestrator `_ORCHESTRATION_DEPTH`：嵌套深度防无限递归
- LaneQueue `_LANE_OWNER`：车道锁重入检测

### 5. 辩论模式真独立视角
两个 Reviewer 用完全独立的 session_key + AgentContext，asyncio.gather 并行——物理上不可能看到对方历史。Judge 识别分歧不取平均。

### 6. 行为指纹测试框架
`BehaviorFingerprint` 区分 `tools_called`（LLM 请求）和 `tools_executed`（成功执行）。安全断言检查 `must_not_execute_tools ∩ tools_executed = ∅`——攻击者骗 LLM 请求危险工具，执行层拦截。

### 7. 信封抽象 + 车道隔离
`InboundEnvelope` 统一 Telegram/Discord 差异，Gateway 只认信封。`LaneQueue` per-session 串行、跨 session 并行，群聊默认要求 @机器人。

### 8. 四大黄金信号 + 零依赖 metrics
自实现 Counter/Histogram/Gauge，`render_prometheus()` 输出标准 exposition format。延迟分桶算 P99 而非平均——平均会掩盖长尾。

### 9. Provider 故障转移 — 把死代码 `retryable` 接通
`ProviderSelector` 实现 `LLMProvider` Protocol，对 TurnRunner 完全透明。RATE_LIMITED/SERVER_ERROR/TIMEOUT 指数退避重试（+25% jitter 防 thundering herd），耗尽切 fallback provider；AUTH_INVALID 和 CONTEXT_TOO_LONG 立即 re-raise（前者密钥无效，后者是压缩信号必须透传）。`retryable` 字段 Phase 1 就定义了，Phase 2 才接通——不超前设计。路由策略层（`router/policy.py`）在分类后串联两阶段：confidence gate（低置信度升档）+ anti-downgrade（30min KV cache 保护窗口防降档）。

### 10. 执行流深度控制 — 干预优先于硬停
不止 `max_iter`/`max_tool_calls` 两道硬熔断。卡死检测用 per-签名整轮计数（catch 重复 + A-B-A-B 振荡），触发时**不 raise**——跳过执行 + 注入恢复消息 + 动态禁用该工具，给 Agent 退路。撞预算走两阶段优雅收尾（注入"别调工具直接答"+剥离工具再做一次，二次才硬停），这是 opensquilla 的招牌机制、LangGraph 没有。终止原因走 `DoneEvent.stop_reason`（6 种）+ `outcome`（3 分类）可观测可测，不塞进 final_text。工具返回 `ToolResult(status, next_action_hint)` 契约，模糊返回是死循环根因。诚实取舍：没抄 opensquilla 的字节级请求去重（NanoCore 每轮 append history，连续相同请求不可能，是死代码），换成 per-tool 调用预算 catch 同工具不同参数的钻空子。详见 `phase8_面试总结.md`。

---

## 技术栈

| 维度 | 选型 | 理由 |
|------|------|------|
| 运行时 | Python 3.12 / asyncio | IO 密集型，单进程足够 |
| LLM 接口 | Anthropic SDK | Claude 流式 + extended thinking |
| 存储 | SQLite + FTS5 trigram | 零依赖，中文全文检索 |
| 通道 | aiogram 3.x / discord.py | 异步原生 IM 适配 |
| 可观测 | 自实现 + 可桥接 OTel | 轻量可解释，不引重 SDK |
| 面板 | Gradio | 几十行省前端工时 |
| 测试 | pytest + pytest-asyncio | 行为指纹自研范式 |

---

## 层间依赖规则

```
core.*      ← 只依赖 provider.*，不 import engine/memory/channels/agents
engine.*    ← 可 import core.*，不 import channels/memory/agents
provider.*  ← 只依赖 core.context（Message 类型）
router.*    ← 可 import provider.*，不 import channels
memory.*    ← 可 import core.*，不 import channels/agents
agents.*    ← 可 import core.* + engine.*
channels.*  ← 可 import 所有层，是最外层
```

---

## 面试总结文档

每个 Phase 配一份面试话术 MD（问题/坏方案/我的方案/代码锚点/快速问答卡）：

- `phase1_面试总结.md` — ReAct 状态机、压缩、事件溯源
- `phase2_面试总结.md` — TurnRunner、路由、降级链
- `phase3_面试总结.md` — 三层记忆、FTS5 trigram、Dream
- `phase4_面试总结.md` — AgentCard、Supervisor、辩论模式
- `phase5_面试总结.md` — 行为指纹、安全边界、约束规格
- `phase6_面试总结.md` — 信封、车道隔离、安全门控
- `phase7_面试总结.md` — OTel 追踪、四大黄金信号、benchmark/压测
- `phase8_面试总结.md` — 执行流深度控制：卡死检测/两阶段收尾/工具契约/终止分类

---

## GitHub

仓库：https://github.com/Wuzheyuan456/nanoharness

# AI Agent 学习进度追踪

## 学习目标
- [x] 掌握 AI Agent 核心概念
- [ ] 熟练使用主流框架 (LangChain, AutoGen, CrewAI)
- [ ] 完成 3 个实战项目
- [ ] 准备 AI Agent 相关面试

## 当前阶段
**阶段**: LangChain 深入学习 → 下一步：RAG
**开始日期**: 2025-12-23

## 学习记录

### Day 1-2 (12/23-24)
- [x] Agent 基础概念
- [x] LangChain 入门
- [x] 完成第一个简单 Agent (ReAct)

### Day 3 (12/25)
- [x] Workflow vs Agentic 区别
- [x] 主流架构对比 (ReAct / Plan-Execute / Reflexion / LATS)
- [x] 架构组合设计 (Plan-Execute + Reflexion)
- [x] 分布式 Agent 设计 (Map-Reduce + 分层 Reflexion)
- [x] **形成完整的 Agent 设计方法论（"实习生标准"）**

### Day 4 (12/26)
- [x] LangChain 基础系统学习
  - [x] ChatOpenAI / Prompt Template / LCEL
  - [x] Output Parser（结构化输出）
  - [x] Chain 组合 / Streaming
- [x] Tool 和 Agent 深入
  - [x] @tool 装饰器原理（元数据提取）
  - [x] ReAct 循环机制
  - [x] **理解 LLM 只看 Tool 描述，不看代码**
- [x] Memory 机制
  - [x] 手动管理对话历史
  - [x] ChatMessageHistory / RunnableWithMessageHistory
  - [x] Agent + MemorySaver
  - [x] **session_id 隔离原理**

## 技能清单

| 技能 | 状态 | 备注 |
|------|------|------|
| Prompt Engineering | 🟢 已掌握 | |
| Agent 架构设计 | 🟢 已掌握 | 见 notes/agent-architecture-design.md |
| LangChain 基础 | 🟢 已掌握 | LCEL/Tool/Memory |
| LangChain Agent | 🟢 已掌握 | ReAct/Tool 选择机制 |
| AutoGen | ⚪ 未开始 | |
| CrewAI | ⚪ 未开始 | |
| RAG | ⚪ 未开始 | 下一步 |
| Multi-Agent | 🟡 进行中 | 理论已掌握，待实战 |

## 里程碑

- [x] 🎯 完成入门项目 (01-simple-agent)
- [ ] 🎯 完成 RAG 项目 (02-rag-agent)
- [ ] 🎯 完成多智能体项目 (03-multi-agent)
- [ ] 🎯 通过模拟面试

## 核心笔记索引

| 主题 | 文件 | 内容 |
|------|------|------|
| Agent 架构设计 | `notes/agent-architecture-design.md` | 完整方法论 + 面试话术 |

## 代码示例索引

| 示例 | 文件 | 知识点 |
|------|------|--------|
| LangChain 基础 | `code-snippets/langchain/01_langchain_basics.py` | LCEL/Parser/Streaming |
| Tool 和 Agent | `code-snippets/langchain/02_tools_and_agents.py` | @tool/ReAct 循环 |
| Tool 元数据 | `code-snippets/langchain/03_tool_prompt_inspection.py` | LLM 如何看 Tool |
| Memory | `code-snippets/langchain/04_memory.py` | 对话历史/session_id |
| Workflow vs Agentic | `code-snippets/langchain/workflow_vs_agentic.py` | 两种模式对比 |
| 架构对比 | `code-snippets/langchain/agent_architectures.py` | 三种架构实现 |
| 组合架构 | `code-snippets/langchain/code_review_agent.py` | Plan-Execute + Reflexion |
| 分布式 Agent | `code-snippets/langchain/distributed_review_agent.py` | Map-Reduce + 分层反思 |

## 下一步计划

1. **RAG（检索增强生成）**
   - 向量数据库基础
   - Embedding 模型
   - 检索 + 生成流程
   
2. **LangGraph**
   - 复杂工作流编排
   - 状态管理

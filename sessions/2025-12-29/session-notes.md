# Session: 2025-12-29

## 主题：项目 1 完善 - 修复 Bug + 添加测试 + 优化检索

### 今日学习目标
- 修复 Gradio Chatbot 消息格式兼容性问题
- 添加单元测试
- 添加 Rerank 优化检索效果

---

## 一、Bug 修复

### 问题描述
```
gradio.exceptions.Error: "Data incompatible with messages format. 
Each message should be a dictionary with 'role' and 'content' keys..."
```

### 原因分析
Gradio 6.x 版本的 Chatbot 组件默认使用新的消息格式（字典格式），而代码使用的是旧的元组格式。

### 解决方案
```python
# 旧格式（不兼容）
history.append((message, response))

# 新格式（兼容）
history.append({"role": "user", "content": message})
history.append({"role": "assistant", "content": response})
```

---

## 二、单元测试

### 测试文件结构
```
tests/
├── __init__.py
├── test_document_loader.py   # DocumentLoader 测试
├── test_vector_store.py      # VectorStore 测试
└── test_qa_engine.py         # QAEngine 测试
```

### 测试覆盖
```
┌─────────────────────────────────────────────────────────────┐
│  测试覆盖                                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DocumentLoader (13 tests)                                  │
│  ─────────────────────────────────────────────────────────  │
│  ✅ 基本文本加载                                            │
│  ✅ 文本切分（chunk_size/overlap）                         │
│  ✅ 元数据保留                                              │
│  ✅ 空文本处理                                              │
│  ✅ 支持格式列表                                            │
│  ✅ 文件不存在处理                                          │
│  ✅ 不支持格式处理                                          │
│  ✅ TXT/Markdown 文件加载                                  │
│  ✅ Unicode 内容处理                                        │
│                                                             │
│  VectorStore (Mock + Integration)                          │
│  ─────────────────────────────────────────────────────────  │
│  ✅ 初始化测试                                              │
│  ✅ 接口完整性测试                                          │
│  ✅ 空状态统计                                              │
│  🔧 集成测试（需要 API Key）                               │
│                                                             │
│  QAEngine (Mock + Integration)                             │
│  ─────────────────────────────────────────────────────────  │
│  ✅ 初始化测试                                              │
│  ✅ 返回 QAResponse                                         │
│  ✅ 来源包含测试                                            │
│  ✅ 对话历史更新                                            │
│  ✅ 清空历史                                                │
│  🔧 集成测试（需要 API Key）                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 运行测试
```bash
cd projects/01-doc-qa-system
pytest tests/ -v --tb=short
```

---

## 三、Rerank 优化检索效果

### 什么是 Rerank？
```
┌─────────────────────────────────────────────────────────────┐
│  检索流程对比                                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  普通检索：                                                 │
│  Query → 向量搜索(top-k=5) → 返回结果                      │
│                                                             │
│  带 Rerank：                                                │
│  Query → 向量搜索(top-k=20) → Rerank → 返回(top-n=5)       │
│                                                             │
│  优势：                                                     │
│  - 向量搜索：快速但粗糙（召回）                            │
│  - Rerank：慢但精确（精排）                                │
│  - 组合使用效果更好                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 实现的 Reranker 类型

1. **SimpleReranker**（简单重排序）
   - 基于关键词匹配
   - 考虑原始排名
   - 考虑关键词位置
   - 不需要额外 API 调用

2. **LLMReranker**（LLM 重排序）
   - 使用 LLM 评估相关性
   - 更精确但更慢
   - 需要 API 调用

3. **HybridReranker**（混合重排序）
   - 先用简单方法快速筛选
   - 对 top 候选使用 LLM 精排
   - 平衡速度和精度

### 使用方式
```python
# 启用 Rerank
qa_engine = QAEngine(
    llm=llm,
    retriever=retriever,
    use_rerank=True,           # 启用重排序
    rerank_type="simple",      # simple/llm/hybrid
)
```

### 测试结果
```
Query: "什么是 AI Agent"

重排序前（向量搜索顺序）：
[0] AI Agent 是能够自主感知、决策、行动的智能系统
[1] Python 是一种流行的编程语言
[2] RAG 是检索增强生成...
[3] LangChain 是用于构建 AI Agent 的框架

重排序后：
[0→0] 分数: 0.760 | AI Agent 是能够自主感知、决策、行动的智能系统
[3→1] 分数: 0.670 | LangChain 是用于构建 AI Agent 的框架
[1→2] 分数: 0.490 | Python 是一种流行的编程语言

✅ 相关文档排名提升
```

---

## 四、项目当前状态

### 文件结构
```
projects/01-doc-qa-system/
├── src/
│   ├── __init__.py
│   ├── document_loader.py   # 文档加载器
│   ├── vector_store.py      # 向量存储
│   ├── qa_engine.py         # 问答引擎（支持 Rerank）
│   ├── reranker.py          # 重排序模块 ✨ 新增
│   └── app.py               # Gradio Web UI（已修复）
├── tests/                   # 单元测试 ✨ 新增
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_vector_store.py
│   └── test_qa_engine.py
├── docs/
└── requirements.txt
```

### 功能清单
| 功能 | 状态 |
|------|------|
| 文档加载（PDF/MD/TXT/DOCX） | ✅ 完成 |
| 向量存储（FAISS） | ✅ 完成 |
| 问答引擎（RAG） | ✅ 完成 |
| Web UI（Gradio） | ✅ 修复 |
| 多轮对话 | ✅ 完成 |
| 引用来源展示 | ✅ 完成 |
| 单元测试 | ✅ 新增 |
| Rerank 优化 | ✅ 新增 |

---

## 五、学习收获

1. **Gradio 版本兼容性**
   - 不同版本 API 可能不同
   - 需要关注组件的消息格式

2. **测试策略**
   - Mock 测试：不依赖外部服务
   - 集成测试：验证真实 API 调用
   - 分开标记，按需运行

3. **Rerank 原理**
   - 两阶段检索：召回 + 精排
   - 简单方法也能有效提升效果
   - 可根据场景选择不同策略

---

## 六、下一步计划

1. **项目 1 进一步优化**
   - [ ] 添加文档预览功能
   - [ ] 持久化向量库
   - [ ] Docker 容器化

2. **开始项目 2：代码助手 Agent**
   - Multi-Agent 架构
   - 代码审查 + Bug 检测

---

*Session 时长: ~1.5小时*
*学习状态: 高效*

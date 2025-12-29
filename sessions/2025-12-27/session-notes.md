# Session: 2025-12-27

## 主题：项目实战 - 智能文档问答系统

### 今日学习目标
- 开始项目 1：智能文档问答系统
- 搭建项目架构
- 实现核心模块

---

## 一、项目概述

### 项目目标
构建一个完整的文档问答系统，支持：
- 上传文档（PDF/Markdown/TXT/DOCX）
- 自动构建向量知识库
- 智能问答（带引用来源）
- Web UI 界面

### 技术栈
```
┌─────────────────────────────────────────────────────────────┐
│  技术架构                                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  前端：Gradio（快速构建 Web UI）                           │
│                                                             │
│  后端：                                                     │
│  ├── LangChain（RAG 框架）                                 │
│  ├── FAISS（向量数据库）                                   │
│  └── SiliconFlow（Embedding: bge-m3）                      │
│                                                             │
│  对话：iFlow API（qwen3-coder-plus）                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、项目结构

```
projects/01-doc-qa-system/
├── src/
│   ├── __init__.py
│   ├── document_loader.py   # 文档加载器
│   ├── vector_store.py      # 向量存储
│   ├── qa_engine.py         # 问答引擎
│   └── app.py               # Gradio Web UI
├── docs/                    # 测试文档
├── tests/                   # 测试用例
└── requirements.txt         # 依赖
```

---

## 三、核心模块实现

### 1. DocumentLoader（文档加载器）

**功能**：
- 支持多种格式（PDF/MD/TXT/DOCX）
- 自动切分文档
- 统一的加载接口

**关键代码**：
```python
class DocumentLoader:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
    
    def load_file(self, file_path: str) -> List[Document]:
        # 根据文件类型选择加载方法
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        elif suffix == ".md":
            return self._load_markdown(file_path)
        # ...
```

**设计要点**：
- 使用 RecursiveCharacterTextSplitter 智能切分
- separators 按优先级切分（段落 > 句子 > 字符）
- chunk_overlap 保持上下文连贯

### 2. VectorStore（向量存储）

**功能**：
- 创建/更新向量数据库
- 相似度搜索
- 持久化存储

**关键代码**：
```python
class VectorStore:
    def create_from_documents(self, documents: List[Document]):
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
    
    def similarity_search(self, query: str, k: int = 4):
        return self.vectorstore.similarity_search(query, k=k)
    
    def as_retriever(self, search_kwargs=None):
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
```

**设计要点**：
- 封装 FAISS 操作
- 支持持久化（save_local/load_local）
- 提供 retriever 接口供 RAG Chain 使用

### 3. QAEngine（问答引擎）

**功能**：
- RAG 问答链
- 对话历史管理
- 引用来源展示

**关键代码**：
```python
class QAEngine:
    def __init__(self, llm, retriever):
        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.chat_history,
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    
    def ask(self, question: str) -> QAResponse:
        sources = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        return QAResponse(answer=answer, sources=sources, query=question)
```

**设计要点**：
- 使用 LCEL 构建链
- 保存对话历史实现多轮对话
- 返回结构化响应（答案 + 来源）

### 4. Gradio Web UI

**功能**：
- 文件上传
- 文本直接输入
- 对话界面
- 状态显示

---

## 四、运行测试

### 模块测试结果

```bash
# 文档加载器
✅ 文档加载器测试通过: 1 块

# 向量存储
✅ 向量数据库创建完成
✅ 搜索测试通过

# 问答引擎
问: 公司年假有多少天？
答: 根据提供的文档，没有找到相关信息。

问: 如何请假？
答: 请假流程：提前3天在OA系统提交申请，由直属领导审批

问: 报销需要什么材料？
答: 报销流程：填写报销单，附上发票，提交财务审核
```

### Web UI 启动

```bash
cd ai-agent-learning
source .venv/bin/activate
python projects/01-doc-qa-system/src/app.py

# 访问 http://localhost:7860
```

---

## 五、学习收获

1. **模块化设计**：将功能拆分为独立模块（Loader/Store/Engine）
2. **接口统一**：不同格式文档使用统一的加载接口
3. **LCEL 实践**：用 LCEL 构建 RAG Chain
4. **Gradio 快速原型**：快速构建可交互的 Web UI

---

## 六、项目状态

| 功能 | 状态 |
|------|------|
| 文档加载（PDF/MD/TXT） | ✅ 完成 |
| 向量存储（FAISS） | ✅ 完成 |
| 问答引擎（RAG） | ✅ 完成 |
| Web UI（Gradio） | ✅ 完成 |
| 多轮对话 | ✅ 完成 |
| 引用来源展示 | ✅ 完成 |
| 持久化存储 | 🟡 基础实现 |
| 单元测试 | ⏳ 待完成 |

---

## 七、下一步计划

1. **功能增强**
   - [ ] 添加更多文档格式支持
   - [ ] 优化检索效果（Rerank）
   - [ ] 添加文档预览功能

2. **工程优化**
   - [ ] 添加单元测试
   - [ ] 错误处理完善
   - [ ] 日志记录

3. **部署准备**
   - [ ] Docker 容器化
   - [ ] 配置文件外置

---

*Session 时长: ~2小时*
*学习状态: 高效*

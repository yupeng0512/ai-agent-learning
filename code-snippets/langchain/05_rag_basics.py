"""
RAG（检索增强生成）基础教程

核心概念：
1. 为什么需要 RAG？LLM 不知道私有数据、知识有截止日期
2. RAG 流程：文档 → 切分 → Embedding → 存储 → 检索 → 生成
3. 核心组件：Embedding 模型 + 向量数据库 + Retriever

API 配置说明：
- iFlow：用于对话（Chat），不支持 Embedding
- SiliconFlow：用于 Embedding，支持 bge-m3 等中文模型

运行：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/05_rag_basics.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API 配置：多平台支持
# ============================================================

# iFlow - 用于对话（Chat）
IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "qwen3-coder-plus")

# SiliconFlow - 用于 Embedding
# 注册地址：https://siliconflow.cn （注册送 14 元）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")


# ============================================================
# Demo 1: RAG 的核心问题 - LLM 不知道私有数据
# ============================================================

def demo_1_llm_limitation():
    """演示 LLM 的知识局限性"""
    print("\n" + "=" * 60)
    print("Demo 1: LLM 的知识局限性")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 问一个关于"私有数据"的问题
    question = "我们公司的请假流程是什么？"
    response = llm.invoke(question)
    
    print(f"问题: {question}")
    print(f"回答: {response.content}")
    print("\n❌ 问题：LLM 不知道你公司的具体流程，只能给通用建议")
    print("   这就是为什么需要 RAG —— 让 LLM 能查询你的私有文档")


# ============================================================
# Demo 2: RAG 的核心流程
# ============================================================

def demo_2_rag_concept():
    """RAG 概念讲解"""
    print("\n" + "=" * 60)
    print("Demo 2: RAG 核心流程")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│  RAG = Retrieval-Augmented Generation（检索增强生成）       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【离线阶段】文档预处理                                      │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  文档   │ → │  切分   │ → │Embedding│ → │ 向量库  │  │
│  │ PDF/TXT │    │  Chunk  │    │  向量化 │    │  存储   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                                             │
│  【在线阶段】查询回答                                        │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  问题   │ → │  检索   │ → │  拼接   │ → │   LLM   │  │
│  │  Query  │    │ 相似文档 │    │ Prompt │    │  回答   │  │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  核心组件：                                                  │
│  1. Embedding 模型：把文本转成向量（数字数组）              │
│  2. 向量数据库：存储和检索向量（Chroma/FAISS/Pinecone）    │
│  3. Retriever：根据问题检索相关文档                         │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================
# Demo 3: Embedding 是什么
# ============================================================

def get_embeddings():
    """
    获取 Embedding 模型
    
    优先使用 SiliconFlow API（bge-m3，中文效果好）
    如果没有配置 API Key，回退到本地模型
    """
    if SILICONFLOW_API_KEY and SILICONFLOW_API_KEY != "your_siliconflow_api_key_here":
        # 使用 SiliconFlow API（推荐）
        print("[Embedding] 使用 SiliconFlow API (bge-m3)")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=SILICONFLOW_EMBEDDING_MODEL,
            openai_api_key=SILICONFLOW_API_KEY,
            openai_api_base=SILICONFLOW_BASE_URL,
        )
    else:
        # 回退到本地模型
        print("[Embedding] 使用本地模型 (sentence-transformers)")
        print("  提示: 配置 SILICONFLOW_API_KEY 可使用更好的 bge-m3 模型")
        from sentence_transformers import SentenceTransformer
        
        class LocalEmbeddings:
            """本地 Embedding 封装，兼容 LangChain 接口"""
            def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
                self.model = SentenceTransformer(model_name)
            
            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_numpy=True).tolist()
            
            def embed_query(self, text):
                return self.model.encode(text, convert_to_numpy=True).tolist()
        
        return LocalEmbeddings()


def demo_3_embedding():
    """演示 Embedding 的概念"""
    print("\n" + "=" * 60)
    print("Demo 3: Embedding（文本向量化）")
    print("=" * 60)
    
    embeddings = get_embeddings()
    
    # 示例文本
    texts = [
        "我喜欢吃苹果",
        "我喜欢吃香蕉", 
        "今天天气很好",
    ]
    
    # 转成向量
    vectors = embeddings.embed_documents(texts)
    
    print("Embedding 把文本转成向量（数字数组）：\n")
    for text, vec in zip(texts, vectors):
        print(f"文本: {text}")
        print(f"向量维度: {len(vec)}")
        print(f"向量前5个值: {[round(v, 4) for v in vec[:5]]}")
        print()
    
    # 计算相似度
    import numpy as np
    
    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    sim_01 = cosine_similarity(vectors[0], vectors[1])
    sim_02 = cosine_similarity(vectors[0], vectors[2])
    
    print(f"相似度计算：")
    print(f"  '{texts[0]}' vs '{texts[1]}': {sim_01:.4f}")
    print(f"  '{texts[0]}' vs '{texts[2]}': {sim_02:.4f}")
    print(f"\n✅ 语义相近的文本，向量也相近（相似度更高）")


# ============================================================
# Demo 4: 简单的 RAG 实现
# ============================================================

def demo_4_simple_rag():
    """最简单的 RAG 实现"""
    print("\n" + "=" * 60)
    print("Demo 4: 简单 RAG 实现")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    # 1. 准备"私有文档"（模拟公司内部文档）
    documents = [
        Document(
            page_content="公司请假流程：员工需要提前3天在OA系统提交请假申请，由直属领导审批。病假需要提供医院证明。",
            metadata={"source": "员工手册", "chapter": "请假制度"}
        ),
        Document(
            page_content="报销流程：员工在费用发生后30天内，通过财务系统提交报销申请，附上发票和审批单。",
            metadata={"source": "员工手册", "chapter": "报销制度"}
        ),
        Document(
            page_content="入职流程：新员工需要在入职当天到HR处办理入职手续，领取工牌和电脑。",
            metadata={"source": "员工手册", "chapter": "入职指南"}
        ),
        Document(
            page_content="年假规定：工作满1年的员工享有5天年假，满10年享有10天年假。年假需提前申请。",
            metadata={"source": "员工手册", "chapter": "请假制度"}
        ),
    ]
    
    print(f"1. 准备文档: {len(documents)} 个")
    
    # 2. 创建 Embedding 模型（使用本地模型）
    embeddings = get_embeddings()
    
    # 3. 创建向量数据库（FAISS 是本地向量库，不需要额外服务）
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"2. 创建向量数据库: FAISS")
    
    # 4. 创建 Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # 返回最相关的2个文档
    print(f"3. 创建 Retriever: 返回 top-2 文档")
    
    # 5. 创建 LLM
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 6. 创建 RAG Prompt
    rag_prompt = ChatPromptTemplate.from_template("""
根据以下文档回答问题。如果文档中没有相关信息，请说"文档中没有相关信息"。

文档内容：
{context}

问题：{question}

回答：""")
    
    # 7. 测试 RAG
    question = "公司请假流程是什么？需要提前几天？"
    
    # 检索相关文档
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"\n4. 用户问题: {question}")
    print(f"\n5. 检索到的文档:")
    for i, doc in enumerate(relevant_docs):
        print(f"   [{i+1}] {doc.page_content[:50]}...")
    
    # 生成回答
    chain = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"\n6. RAG 回答: {answer}")
    print("\n✅ RAG 让 LLM 能够回答关于私有文档的问题！")


# ============================================================
# Demo 5: 使用 LangChain 的 RAG Chain
# ============================================================

def demo_5_langchain_rag_chain():
    """使用 LangChain 封装好的 RAG Chain"""
    print("\n" + "=" * 60)
    print("Demo 5: LangChain RAG Chain")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    # 准备文档
    documents = [
        Document(page_content="LangChain 是一个用于开发 LLM 应用的框架，支持 Chain、Agent、RAG 等功能。"),
        Document(page_content="RAG 是检索增强生成，通过检索外部知识来增强 LLM 的回答能力。"),
        Document(page_content="Agent 是能够自主决策和使用工具的 AI 系统，核心是 ReAct 循环。"),
        Document(page_content="LCEL 是 LangChain Expression Language，使用 | 管道符连接组件。"),
    ]
    
    # 创建向量库和 Retriever（使用本地 Embedding）
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 创建 LLM
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # RAG Prompt
    prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：

上下文：
{context}

问题：{question}
""")
    
    # 使用 LCEL 构建 RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 测试
    questions = [
        "什么是 RAG？",
        "LangChain 能做什么？",
        "Agent 的核心是什么？",
    ]
    
    print("使用 LCEL 构建的 RAG Chain：\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


# ============================================================
# Demo 6: 文档切分（Chunking）
# ============================================================

def demo_6_text_splitting():
    """演示文档切分的重要性"""
    print("\n" + "=" * 60)
    print("Demo 6: 文档切分（Text Splitting）")
    print("=" * 60)
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # 模拟一个长文档
    long_document = """
# AI Agent 开发指南

## 第一章：什么是 AI Agent

AI Agent 是一种能够自主感知环境、做出决策并采取行动的智能系统。与传统的 AI 模型不同，Agent 具有自主性和目标导向性。

Agent 的核心组件包括：
1. 感知模块：获取环境信息
2. 决策模块：基于 LLM 进行推理
3. 行动模块：执行具体操作
4. 记忆模块：存储历史信息

## 第二章：Agent 架构

常见的 Agent 架构有：

### ReAct 架构
ReAct（Reasoning + Acting）是最基础的 Agent 架构。它的核心是一个循环：
- Thought：思考当前应该做什么
- Action：选择并执行工具
- Observation：观察执行结果
- 重复直到任务完成

### Plan-Execute 架构
先制定完整计划，再逐步执行。适合复杂任务。

### Reflexion 架构
在执行后进行反思，从错误中学习。

## 第三章：工具使用

Agent 通过工具与外部世界交互。常见工具包括：
- 搜索引擎
- 代码执行器
- 数据库查询
- API 调用

工具的定义需要清晰的描述，因为 LLM 是根据描述来选择工具的。
"""
    
    print(f"原始文档长度: {len(long_document)} 字符\n")
    
    # 方式 1：按字符数切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 每个 chunk 最大 200 字符
        chunk_overlap=50,    # chunk 之间重叠 50 字符
        separators=["\n\n", "\n", "。", "，", " "],  # 优先在这些位置切分
    )
    
    chunks = splitter.split_text(long_document)
    
    print(f"切分后: {len(chunks)} 个 chunks\n")
    print("前 3 个 chunks：")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({len(chunk)} 字符) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    print("""
\n┌─────────────────────────────────────────────────────────────┐
│  为什么要切分？                                              │
├─────────────────────────────────────────────────────────────┤
│  1. Embedding 模型有长度限制                                 │
│  2. 检索更精准（小 chunk 更容易匹配具体问题）               │
│  3. 减少 Token 消耗（只传相关部分给 LLM）                   │
│                                                             │
│  切分策略：                                                  │
│  - chunk_size: 每个块的大小                                 │
│  - chunk_overlap: 块之间的重叠（保持上下文连续性）          │
│  - separators: 优先在段落/句子边界切分                      │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================
# Demo 7: 为什么用 Embedding 而不是关键词搜索？
# ============================================================

def demo_7_why_embedding():
    """解释 Embedding vs 关键词搜索的核心区别"""
    print("\n" + "=" * 60)
    print("Demo 7: 为什么用 Embedding 而不是关键词搜索？")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│  场景：用户搜索 "怎么请假"                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【关键词搜索】匹配字面                                      │
│  ─────────────────────────────────────────────────────────  │
│  文档1: "请假流程：提前3天在OA提交申请"     ❌ 没有"怎么"   │
│  文档2: "年假规定：满1年享有5天年假"        ❌ 没有"怎么请假"│
│  文档3: "怎么申请报销？填写报销单"          ❌ 有"怎么"但不相关│
│                                                             │
│  结果：可能找不到，或者找到不相关的                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【Embedding 向量搜索】匹配语义                              │
│  ─────────────────────────────────────────────────────────  │
│  "怎么请假" 的向量 ≈ "请假流程" 的向量                      │
│                                                             │
│  文档1: "请假流程：提前3天在OA提交申请"     ✅ 语义相近！    │
│  文档2: "年假规定：满1年享有5天年假"        ⚠️ 有点相关     │
│  文档3: "怎么申请报销？填写报销单"          ❌ 语义不同      │
│                                                             │
│  结果：找到真正相关的文档                                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  核心区别                                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  关键词搜索：用户说的词 必须和 文档里的词 一模一样          │
│  Embedding： 用户的意思 和 文档的意思 相近就行              │
│                                                             │
│  举例：                                                     │
│  ┌─────────────────────┬──────────────┬──────────────┐     │
│  │ 对比                │ 关键词搜索   │ Embedding    │     │
│  ├─────────────────────┼──────────────┼──────────────┤     │
│  │ "怎么请假"vs"请假流程"│ ❌ 不匹配   │ ✅ 匹配      │     │
│  │ "苹果手机"vs"iPhone" │ ❌ 不匹配   │ ✅ 匹配      │     │
│  │ "我很开心"vs"我很高兴"│ ❌ 不匹配   │ ✅ 匹配      │     │
│  └─────────────────────┴──────────────┴──────────────┘     │
│                                                             │
│  Embedding 能做到是因为模型在海量文本上训练，学会了语义     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================
# Demo 8: RAG 优化技巧
# ============================================================

def demo_8_rag_optimization():
    """RAG 优化技巧总结"""
    print("\n" + "=" * 60)
    print("Demo 8: RAG 优化技巧")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│  RAG 优化方向                                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【检索优化】                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 混合检索：向量检索 + 关键词检索（BM25）          │   │
│  │    - 向量：语义相似                                  │   │
│  │    - 关键词：精确匹配                                │   │
│  │                                                     │   │
│  │ 2. 重排序（Rerank）：用更精确的模型对结果重排       │   │
│  │    retriever → reranker → top-k                    │   │
│  │                                                     │   │
│  │ 3. 查询改写：把用户问题改写成更适合检索的形式       │   │
│  │    "怎么请假" → "请假流程 申请方式 审批"           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  【切分优化】                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 语义切分：按段落/章节切分，而不是固定长度        │   │
│  │ 2. 父子 Chunk：小 chunk 检索，返回大 chunk 上下文   │   │
│  │ 3. 元数据增强：给 chunk 添加标题、来源等信息        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  【生成优化】                                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 引用来源：让 LLM 标注答案来自哪个文档            │   │
│  │ 2. 置信度：让 LLM 评估答案的可靠程度                │   │
│  │ 3. 多轮对话：结合 Memory 实现追问                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  面试常问：RAG 效果不好怎么优化？                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  回答框架：                                                  │
│  1. 先定位问题在哪个环节（检索不准 or 生成不好）           │
│  2. 检索不准 → 优化切分、混合检索、重排序                  │
│  3. 生成不好 → 优化 Prompt、增加上下文、调整温度           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print("=" * 60)
    print("API 配置状态")
    print("=" * 60)
    print(f"对话模型 (iFlow): {IFLOW_MODEL}")
    if SILICONFLOW_API_KEY and SILICONFLOW_API_KEY != "your_siliconflow_api_key_here":
        print(f"Embedding (SiliconFlow): {SILICONFLOW_EMBEDDING_MODEL}")
    else:
        print("Embedding: 本地模型 (未配置 SILICONFLOW_API_KEY)")
        print("  提示: 去 https://siliconflow.cn 注册可获得免费额度")
    
    demo_1_llm_limitation()
    demo_2_rag_concept()
    demo_3_embedding()
    demo_4_simple_rag()
    demo_5_langchain_rag_chain()
    demo_6_text_splitting()
    demo_7_why_embedding()
    demo_8_rag_optimization()
    
    print("\n" + "=" * 60)
    print("RAG 基础教程完成！")
    print("=" * 60)
    print("""
核心要点：

1. RAG = 检索 + 生成，让 LLM 能回答私有数据问题
2. 流程：文档 → 切分 → Embedding → 向量库 → 检索 → 生成
3. 核心组件：Embedding 模型 + 向量数据库 + Retriever
4. 优化方向：检索优化、切分优化、生成优化

面试要点：
- RAG 解决什么问题？（知识截止、私有数据、幻觉）
- RAG 效果不好怎么办？（定位环节 → 针对性优化）

下一步：Agent + RAG 结合，构建知识库问答 Agent
""")

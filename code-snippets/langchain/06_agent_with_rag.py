"""
Agent + RAG 结合教程

核心概念：
1. 单纯 RAG：用户问 → 检索 → 回答（一次性，被动）
2. Agent + RAG：Agent 自己决定什么时候需要查知识库（主动）
3. 多工具协作：RAG 只是 Agent 的工具之一

应用场景：
- 智能客服：查知识库 + 创建工单 + 转人工
- 企业助手：查文档 + 发邮件 + 预约会议
- 代码助手：查文档 + 执行代码 + 搜索网络

运行：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/06_agent_with_rag.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# API 配置
# ============================================================

# iFlow - 用于对话
IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "qwen3-coder-plus")

# SiliconFlow - 用于 Embedding
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")


def get_embeddings():
    """获取 Embedding 模型"""
    if SILICONFLOW_API_KEY and SILICONFLOW_API_KEY != "your_siliconflow_api_key_here":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=SILICONFLOW_EMBEDDING_MODEL,
            openai_api_key=SILICONFLOW_API_KEY,
            openai_api_base=SILICONFLOW_BASE_URL,
        )
    else:
        # 回退到本地模型
        from sentence_transformers import SentenceTransformer
        
        class LocalEmbeddings:
            def __init__(self):
                self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            
            def embed_documents(self, texts):
                return self.model.encode(texts, convert_to_numpy=True).tolist()
            
            def embed_query(self, text):
                return self.model.encode(text, convert_to_numpy=True).tolist()
        
        return LocalEmbeddings()


def get_llm():
    """获取 LLM"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )


# ============================================================
# Demo 1: 单纯 RAG vs Agent + RAG 的区别
# ============================================================

def demo_1_difference():
    """对比单纯 RAG 和 Agent + RAG"""
    print("\n" + "=" * 60)
    print("Demo 1: 单纯 RAG vs Agent + RAG")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│  单纯 RAG（被动检索）                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户: "公司请假流程是什么？"                               │
│         ↓                                                   │
│  系统: 检索知识库 → 返回答案                                │
│                                                             │
│  特点：                                                     │
│  - 每次都检索（不管需不需要）                               │
│  - 只能回答知识库里的问题                                   │
│  - 无法执行其他操作                                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Agent + RAG（主动决策）                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户: "帮我查一下请假流程，然后帮我请3天假"                │
│         ↓                                                   │
│  Agent 思考: 需要两步                                       │
│    1. 先用 RAG 工具查询请假流程                             │
│    2. 再用请假工具提交申请                                  │
│         ↓                                                   │
│  Agent 执行: 调用工具 → 返回结果                            │
│                                                             │
│  特点：                                                     │
│  - 自己决定是否需要检索                                     │
│  - 可以组合多个工具完成复杂任务                             │
│  - 更智能、更灵活                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================
# Demo 2: 创建知识库（向量数据库）
# ============================================================

def create_knowledge_base():
    """创建公司知识库"""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    
    # 模拟公司内部文档
    documents = [
        # 请假相关
        Document(
            page_content="请假流程：员工需要提前3天在OA系统提交请假申请，由直属领导审批。病假需要提供医院证明。",
            metadata={"category": "请假", "source": "员工手册"}
        ),
        Document(
            page_content="年假规定：工作满1年享有5天年假，满5年享有10天，满10年享有15天。年假可分次使用，但需提前申请。",
            metadata={"category": "请假", "source": "员工手册"}
        ),
        Document(
            page_content="病假规定：病假需要提供正规医院的诊断证明。3天以内由部门领导审批，3天以上需HR审批。",
            metadata={"category": "请假", "source": "员工手册"}
        ),
        
        # 报销相关
        Document(
            page_content="报销流程：费用发生后30天内，在财务系统提交报销申请，附上发票原件和审批单。500元以下部门经理审批，500元以上需总监审批。",
            metadata={"category": "报销", "source": "财务制度"}
        ),
        Document(
            page_content="差旅报销标准：飞机经济舱、高铁二等座。住宿标准：一线城市500元/晚，二线城市350元/晚。",
            metadata={"category": "报销", "source": "财务制度"}
        ),
        
        # IT相关
        Document(
            page_content="VPN使用：下载公司VPN客户端，使用工号登录。首次使用需要IT部门开通权限，联系IT热线：8888。",
            metadata={"category": "IT", "source": "IT指南"}
        ),
        Document(
            page_content="电脑故障处理：先尝试重启。如果问题持续，联系IT热线8888或提交IT工单。紧急问题可直接找IT部门。",
            metadata={"category": "IT", "source": "IT指南"}
        ),
        
        # 会议室相关
        Document(
            page_content="会议室预约：在OA系统的会议室预约模块进行预约。大会议室（20人以上）需要提前1天预约。",
            metadata={"category": "行政", "source": "行政指南"}
        ),
    ]
    
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def demo_2_knowledge_base():
    """演示创建知识库"""
    print("\n" + "=" * 60)
    print("Demo 2: 创建公司知识库")
    print("=" * 60)
    
    vectorstore = create_knowledge_base()
    
    print("已创建知识库，包含以下文档：")
    print("  - 请假相关：请假流程、年假规定、病假规定")
    print("  - 报销相关：报销流程、差旅标准")
    print("  - IT相关：VPN使用、电脑故障")
    print("  - 行政相关：会议室预约")
    
    # 测试检索
    print("\n测试检索：")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    test_queries = ["怎么请假", "出差住宿标准多少", "电脑坏了怎么办"]
    for query in test_queries:
        docs = retriever.invoke(query)
        print(f"\n  Q: {query}")
        print(f"  A: {docs[0].page_content[:60]}...")
    
    return vectorstore


# ============================================================
# Demo 3: 把 RAG 封装成 Tool
# ============================================================

def demo_3_rag_as_tool():
    """把 RAG 封装成 Agent 可用的工具"""
    print("\n" + "=" * 60)
    print("Demo 3: 把 RAG 封装成 Tool")
    print("=" * 60)
    
    from langchain_core.tools import tool
    
    # 创建知识库
    vectorstore = create_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 方式1：简单封装 - 直接返回检索结果
    @tool
    def search_company_docs(query: str) -> str:
        """搜索公司内部文档，包括请假制度、报销流程、IT指南等。
        当用户询问公司相关政策、流程、规定时使用此工具。
        
        Args:
            query: 搜索关键词，如"请假流程"、"报销标准"等
        """
        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关文档"
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"[{i+1}] {doc.page_content}")
        return "\n\n".join(results)
    
    print("创建了 RAG 工具: search_company_docs")
    print(f"  描述: {search_company_docs.description[:50]}...")
    
    # 测试工具
    print("\n测试工具调用：")
    result = search_company_docs.invoke("年假有多少天")
    print(f"  输入: '年假有多少天'")
    print(f"  输出: {result[:100]}...")
    
    return search_company_docs, vectorstore


# ============================================================
# Demo 4: Agent + RAG 实战
# ============================================================

def demo_4_agent_with_rag():
    """Agent 使用 RAG 工具"""
    print("\n" + "=" * 60)
    print("Demo 4: Agent + RAG 实战")
    print("=" * 60)
    
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.prebuilt import create_react_agent
    
    # 创建知识库和 RAG 工具
    vectorstore = create_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    @tool
    def search_company_docs(query: str) -> str:
        """搜索公司内部文档，包括请假制度、报销流程、IT指南、行政规定等。
        当用户询问公司政策、流程、规定时使用此工具。"""
        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关文档"
        return "\n\n".join([f"[来源:{doc.metadata.get('source', '未知')}] {doc.page_content}" for doc in docs])
    
    # 创建其他工具（模拟）
    @tool
    def submit_leave_request(days: int, reason: str) -> str:
        """提交请假申请。
        
        Args:
            days: 请假天数
            reason: 请假原因
        """
        return f"✅ 请假申请已提交：{days}天，原因：{reason}。等待领导审批。"
    
    @tool
    def book_meeting_room(room: str, time: str) -> str:
        """预约会议室。
        
        Args:
            room: 会议室名称，如"大会议室"、"小会议室A"
            time: 预约时间，如"明天下午2点"
        """
        return f"✅ 会议室预约成功：{room}，时间：{time}"
    
    @tool
    def get_current_time() -> str:
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 创建 Agent
    llm = get_llm()
    tools = [search_company_docs, submit_leave_request, book_meeting_room, get_current_time]
    
    system_prompt = """你是公司智能助手，可以帮助员工：
1. 查询公司制度和流程（使用 search_company_docs 工具）
2. 提交请假申请（使用 submit_leave_request 工具）
3. 预约会议室（使用 book_meeting_room 工具）
4. 查询当前时间（使用 get_current_time 工具）

请根据用户需求，选择合适的工具来完成任务。如果需要多个步骤，请逐步完成。
回答时请简洁明了，直接给出结果。"""
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    
    # 测试场景
    test_cases = [
        # 场景1：简单查询（只需要 RAG）
        "公司年假是怎么规定的？",
        
        # 场景2：复合任务（RAG + 其他工具）
        "我想请2天假去旅游，请先告诉我请假流程，然后帮我提交申请",
        
        # 场景3：不需要 RAG 的任务
        "现在几点了？",
    ]
    
    print("Agent 工具列表：")
    for t in tools:
        print(f"  - {t.name}: {t.description[:40]}...")
    
    for i, query in enumerate(test_cases):
        print(f"\n{'─' * 50}")
        print(f"场景 {i+1}: {query}")
        print("─" * 50)
        
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        
        # 显示 Agent 的思考过程
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  🔧 调用工具: {tc['name']}")
                    print(f"     参数: {tc['args']}")
            elif msg.type == "tool":
                print(f"  📋 工具返回: {msg.content[:80]}...")
            elif msg.type == "ai" and msg.content:
                print(f"\n  🤖 Agent 回答: {msg.content}")


# ============================================================
# Demo 5: 带 Memory 的 RAG Agent
# ============================================================

def demo_5_rag_agent_with_memory():
    """带记忆的 RAG Agent，支持多轮对话"""
    print("\n" + "=" * 60)
    print("Demo 5: 带 Memory 的 RAG Agent")
    print("=" * 60)
    
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    
    # 创建知识库
    vectorstore = create_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    @tool
    def search_company_docs(query: str) -> str:
        """搜索公司内部文档，包括请假、报销、IT、行政等制度。"""
        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关文档"
        return "\n\n".join([doc.page_content for doc in docs])
    
    # 创建带 Memory 的 Agent
    llm = get_llm()
    memory = MemorySaver()
    
    agent = create_react_agent(
        llm, 
        [search_company_docs],
        prompt="你是公司智能助手，帮助员工查询公司制度。请简洁回答。",
        checkpointer=memory
    )
    
    # 模拟多轮对话
    config = {"configurable": {"thread_id": "user_001"}}
    
    conversation = [
        "请假需要提前几天申请？",
        "那病假呢？需要什么材料？",  # 追问，Agent 需要记住上下文
        "好的，500元以上的报销谁审批？",  # 切换话题
    ]
    
    print("多轮对话演示：\n")
    for query in conversation:
        print(f"👤 用户: {query}")
        result = agent.invoke({"messages": [HumanMessage(content=query)]}, config)
        
        # 获取最后的 AI 回答
        ai_response = result["messages"][-1].content
        print(f"🤖 助手: {ai_response}\n")


# ============================================================
# Demo 6: 高级技巧 - 自定义 RAG Chain 作为工具
# ============================================================

def demo_6_advanced_rag_tool():
    """更高级的 RAG 工具：带引用来源"""
    print("\n" + "=" * 60)
    print("Demo 6: 高级 RAG 工具（带引用来源）")
    print("=" * 60)
    
    from langchain_core.tools import tool
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    
    # 创建知识库
    vectorstore = create_knowledge_base()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    
    # 创建带引用的 RAG Chain
    rag_prompt = ChatPromptTemplate.from_template("""
根据以下文档回答问题。请在回答末尾标注信息来源。

文档内容：
{context}

问题：{question}

要求：
1. 只根据文档内容回答，不要编造
2. 如果文档中没有相关信息，请说"文档中未找到相关信息"
3. 在回答末尾用【来源：xxx】标注信息来源
""")
    
    @tool
    def search_with_citation(query: str) -> str:
        """搜索公司文档并返回带引用来源的答案。
        比普通搜索更可靠，会标注信息来源。"""
        # 检索
        docs = retriever.invoke(query)
        if not docs:
            return "文档中未找到相关信息"
        
        # 构建上下文（带来源标注）
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "未知")
            context_parts.append(f"[来源:{source}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)
        
        # 用 LLM 生成答案
        chain = rag_prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        return answer
    
    # 创建 Agent
    agent = create_react_agent(
        llm, 
        [search_with_citation],
        prompt="你是公司助手。使用 search_with_citation 工具查询公司制度，该工具会返回带引用来源的准确答案。"
    )
    
    # 测试
    test_queries = [
        "出差住宿标准是多少？",
        "VPN怎么使用？",
    ]
    
    print("带引用来源的 RAG 回答：\n")
    for query in test_queries:
        print(f"Q: {query}")
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        answer = result["messages"][-1].content
        print(f"A: {answer}\n")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print("=" * 60)
    print("Agent + RAG 结合教程")
    print("=" * 60)
    print(f"对话模型: {IFLOW_MODEL}")
    print(f"Embedding: {SILICONFLOW_EMBEDDING_MODEL if SILICONFLOW_API_KEY else '本地模型'}")
    
    demo_1_difference()
    demo_2_knowledge_base()
    demo_3_rag_as_tool()
    demo_4_agent_with_rag()
    demo_5_rag_agent_with_memory()
    demo_6_advanced_rag_tool()
    
    print("\n" + "=" * 60)
    print("Agent + RAG 教程完成！")
    print("=" * 60)
    print("""
核心要点：

1. Agent + RAG vs 单纯 RAG
   - 单纯 RAG：每次都检索，只能回答知识库问题
   - Agent + RAG：自主决定是否检索，可组合多工具

2. RAG 作为 Tool 的封装方式
   - 简单封装：直接返回检索结果
   - 高级封装：RAG Chain + 引用来源

3. 实际应用模式
   - 智能客服：查知识库 + 创建工单
   - 企业助手：查文档 + 执行操作
   - 多轮对话：Memory + RAG

面试要点：
- Agent 和 RAG 如何结合？（RAG 作为 Tool）
- 什么时候用 Agent + RAG？（需要多工具协作、自主决策时）

下一步：学习更复杂的 Agent 架构（Plan-Execute、Multi-Agent）
""")

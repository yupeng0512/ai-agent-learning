"""
问答引擎模块

功能：
- RAG 问答链
- 对话历史管理
- 引用来源展示
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage


@dataclass
class QAResponse:
    """问答响应"""
    answer: str
    sources: List[Document]
    query: str


class QAEngine:
    """问答引擎"""
    
    def __init__(
        self,
        llm,
        retriever,
        system_prompt: Optional[str] = None,
    ):
        """
        初始化问答引擎
        
        Args:
            llm: 语言模型
            retriever: 检索器
            system_prompt: 系统提示词（可选）
        """
        self.llm = llm
        self.retriever = retriever
        self.chat_history: List = []
        
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = """你是一个专业的文档问答助手。请根据提供的上下文信息回答用户的问题。

回答要求：
1. 只基于提供的上下文信息回答，不要编造
2. 如果上下文中没有相关信息，请明确说明"根据提供的文档，没有找到相关信息"
3. 回答要简洁、准确、有条理
4. 如果可能，引用具体的来源

上下文信息：
{context}"""
        
        self.system_prompt = system_prompt
        self._build_chain()
    
    def _build_chain(self):
        """构建 RAG 链"""
        # 问答提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        
        # 格式化检索到的文档
        def format_docs(docs: List[Document]) -> str:
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知来源")
                formatted.append(f"[{i}] 来源: {source}\n{doc.page_content}")
            return "\n\n".join(formatted)
        
        # 构建链
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.chat_history,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str, return_sources: bool = True) -> QAResponse:
        """
        提问
        
        Args:
            question: 问题
            return_sources: 是否返回来源文档
            
        Returns:
            QAResponse 对象
        """
        # 获取相关文档
        sources = self.retriever.invoke(question) if return_sources else []
        
        # 生成回答
        answer = self.chain.invoke(question)
        
        # 更新对话历史
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        # 限制历史长度（保留最近 10 轮对话）
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        return QAResponse(
            answer=answer,
            sources=sources,
            query=question,
        )
    
    def ask_simple(self, question: str) -> str:
        """
        简单提问（只返回答案）
        
        Args:
            question: 问题
            
        Returns:
            答案字符串
        """
        return self.ask(question, return_sources=False).answer
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        print("✅ 对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Returns:
            对话历史列表
        """
        history = []
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history


class QAEngineWithAgent:
    """带 Agent 能力的问答引擎（可扩展工具）"""
    
    def __init__(
        self,
        llm,
        retriever,
        tools: Optional[List] = None,
    ):
        """
        初始化带 Agent 的问答引擎
        
        Args:
            llm: 语言模型
            retriever: 检索器
            tools: 额外的工具列表
        """
        from langchain_core.tools import tool
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain import hub
        
        self.llm = llm
        self.retriever = retriever
        self.chat_history = []
        
        # 创建知识库搜索工具
        @tool
        def search_knowledge_base(query: str) -> str:
            """搜索知识库获取相关信息。当需要查询文档内容时使用此工具。"""
            docs = retriever.invoke(query)
            if not docs:
                return "未找到相关信息"
            
            results = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知")
                results.append(f"[{i}] {doc.page_content[:200]}... (来源: {source})")
            return "\n\n".join(results)
        
        # 组合工具
        all_tools = [search_knowledge_base]
        if tools:
            all_tools.extend(tools)
        
        # 创建 Agent
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, all_tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
        )
    
    def ask(self, question: str) -> str:
        """
        提问
        
        Args:
            question: 问题
            
        Returns:
            答案
        """
        result = self.agent_executor.invoke({"input": question})
        return result.get("output", "无法回答")


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv(dotenv_path="../../.env")
    
    # 获取 API 配置
    IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
    IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL")
    IFLOW_MODEL = os.getenv("IFLOW_MODEL", "qwen3-coder-plus")
    
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL")
    
    if not IFLOW_API_KEY or not SILICONFLOW_API_KEY:
        print("❌ 请配置 API Key")
        exit(1)
    
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    
    # 创建 LLM
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
        temperature=0,
    )
    
    # 创建 Embedding
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
    )
    
    # 测试文档
    test_docs = [
        Document(page_content="公司年假规定：工作满1年享有5天年假，满5年享有10天年假", metadata={"source": "员工手册"}),
        Document(page_content="请假流程：提前3天在OA系统提交申请，由直属领导审批", metadata={"source": "员工手册"}),
        Document(page_content="报销流程：填写报销单，附上发票，提交财务审核", metadata={"source": "财务制度"}),
    ]
    
    # 创建向量数据库
    vectorstore = FAISS.from_documents(test_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 创建问答引擎
    qa = QAEngine(llm, retriever)
    
    # 测试问答
    print("\n--- 测试问答 ---")
    
    questions = [
        "公司年假有多少天？",
        "如何请假？",
        "报销需要什么材料？",
    ]
    
    for q in questions:
        print(f"\n问: {q}")
        response = qa.ask(q)
        print(f"答: {response.answer}")
        print(f"来源: {[doc.metadata.get('source') for doc in response.sources]}")

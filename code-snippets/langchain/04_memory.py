"""
LangChain Memory 教程 - 让 Agent 记住对话历史

核心问题：LLM 本身是无状态的，每次调用都是独立的
解决方案：用 Memory 管理对话历史，每次调用时带上历史

运行：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/04_memory.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "TBStars2-200B-A13B")


# ============================================================
# Demo 1: 没有 Memory 的问题
# ============================================================

def demo_1_without_memory():
    """演示没有 Memory 时 LLM 记不住上下文"""
    print("\n" + "=" * 60)
    print("Demo 1: 没有 Memory 的问题")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 第一轮对话
    response1 = llm.invoke("我叫小明，请记住我的名字")
    print(f"用户: 我叫小明，请记住我的名字")
    print(f"AI: {response1.content}\n")
    
    # 第二轮对话 - LLM 不记得了！
    response2 = llm.invoke("我叫什么名字？")
    print(f"用户: 我叫什么名字？")
    print(f"AI: {response2.content}")
    
    print("\n❌ 问题：LLM 是无状态的，每次调用都是独立的，不记得之前说过什么")


# ============================================================
# Demo 2: 手动管理对话历史
# ============================================================

def demo_2_manual_history():
    """手动管理对话历史 - 最基础的方式"""
    print("\n" + "=" * 60)
    print("Demo 2: 手动管理对话历史")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 手动维护消息历史
    messages = [
        SystemMessage(content="你是一个友好的助手，请记住用户告诉你的信息。")
    ]
    
    # 第一轮
    messages.append(HumanMessage(content="我叫小明，我喜欢编程"))
    response1 = llm.invoke(messages)
    messages.append(AIMessage(content=response1.content))
    
    print(f"用户: 我叫小明，我喜欢编程")
    print(f"AI: {response1.content}\n")
    
    # 第二轮 - 带上历史
    messages.append(HumanMessage(content="我叫什么？我喜欢什么？"))
    response2 = llm.invoke(messages)
    messages.append(AIMessage(content=response2.content))
    
    print(f"用户: 我叫什么？我喜欢什么？")
    print(f"AI: {response2.content}")
    
    print(f"\n✅ 成功！通过传递完整消息历史，LLM 能记住上下文")
    print(f"   当前消息数: {len(messages)}")


# ============================================================
# Demo 3: 使用 ChatMessageHistory
# ============================================================

def demo_3_chat_message_history():
    """使用 LangChain 的 ChatMessageHistory 管理历史"""
    print("\n" + "=" * 60)
    print("Demo 3: ChatMessageHistory")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 创建消息历史存储
    history = InMemoryChatMessageHistory()
    
    def chat(user_input: str) -> str:
        """封装的聊天函数"""
        # 添加用户消息
        history.add_user_message(user_input)
        
        # 调用 LLM（带上历史）
        response = llm.invoke(history.messages)
        
        # 保存 AI 回复
        history.add_ai_message(response.content)
        
        return response.content
    
    # 多轮对话
    print(f"用户: 我是一名 Python 开发者")
    print(f"AI: {chat('我是一名 Python 开发者')}\n")
    
    print(f"用户: 我想学习 AI Agent 开发")
    print(f"AI: {chat('我想学习 AI Agent 开发')}\n")
    
    print(f"用户: 根据我的背景，给我一个学习建议")
    print(f"AI: {chat('根据我的背景，给我一个学习建议')}")
    
    print(f"\n消息历史: {len(history.messages)} 条")


# ============================================================
# Demo 4: 使用 RunnableWithMessageHistory（推荐方式）
# ============================================================

def demo_4_runnable_with_history():
    """使用 RunnableWithMessageHistory - LangChain 推荐方式"""
    print("\n" + "=" * 60)
    print("Demo 4: RunnableWithMessageHistory（推荐）")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 创建带历史占位符的 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个编程导师，帮助用户学习编程。"),
        MessagesPlaceholder(variable_name="history"),  # 历史消息插入点
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    # 存储多个会话的历史
    store = {}
    
    def get_session_history(session_id: str):
        """根据 session_id 获取对应的历史"""
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    # 包装成带历史的 Runnable
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # 会话 1
    config1 = {"configurable": {"session_id": "user_001"}}
    
    print("【会话 1: user_001】")
    response = chain_with_history.invoke(
        {"input": "我想学 LangChain"},
        config=config1
    )
    print(f"用户: 我想学 LangChain")
    print(f"AI: {response.content}\n")
    
    response = chain_with_history.invoke(
        {"input": "有什么学习资源推荐？"},
        config=config1
    )
    print(f"用户: 有什么学习资源推荐？")
    print(f"AI: {response.content}\n")
    
    # 会话 2 - 不同用户，独立历史
    config2 = {"configurable": {"session_id": "user_002"}}
    
    print("【会话 2: user_002】")
    response = chain_with_history.invoke(
        {"input": "Python 怎么入门？"},
        config=config2
    )
    print(f"用户: Python 怎么入门？")
    print(f"AI: {response.content}")
    
    print(f"\n✅ 不同 session_id 有独立的对话历史")
    print(f"   user_001 历史: {len(store['user_001'].messages)} 条")
    print(f"   user_002 历史: {len(store['user_002'].messages)} 条")


# ============================================================
# Demo 5: 带 Memory 的 Agent
# ============================================================

def demo_5_agent_with_memory():
    """Agent + Memory：让 Agent 记住对话"""
    print("\n" + "=" * 60)
    print("Demo 5: Agent + Memory")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式。当用户需要进行数学计算时使用。
        
        Args:
            expression: 数学表达式，如 "2 + 3 * 4"
        """
        try:
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except:
            return "计算出错，请检查表达式"
    
    # 创建带记忆的 Agent
    memory = MemorySaver()
    agent = create_react_agent(llm, [calculate], checkpointer=memory)
    
    # 配置会话 ID
    config = {"configurable": {"thread_id": "calc_session_1"}}
    
    # 第一轮
    print("用户: 帮我算一下 15 * 8")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "帮我算一下 15 * 8"}]},
        config=config
    )
    print(f"AI: {result['messages'][-1].content}\n")
    
    # 第二轮 - Agent 记得之前的计算
    print("用户: 把刚才的结果乘以 2")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "把刚才的结果乘以 2"}]},
        config=config
    )
    print(f"AI: {result['messages'][-1].content}")
    
    print("\n✅ Agent 能记住之前的对话和计算结果")


# ============================================================
# Demo 6: Memory 类型对比
# ============================================================

def demo_6_memory_types():
    """不同 Memory 类型的对比说明"""
    print("\n" + "=" * 60)
    print("Demo 6: Memory 类型对比")
    print("=" * 60)
    
    print("""
┌─────────────────────────────────────────────────────────────┐
│  Memory 类型对比                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. InMemoryChatMessageHistory                              │
│     └─ 存储：内存                                           │
│     └─ 特点：简单，重启丢失                                 │
│     └─ 场景：开发测试                                       │
│                                                             │
│  2. FileChatMessageHistory                                  │
│     └─ 存储：本地文件                                       │
│     └─ 特点：持久化，单机                                   │
│     └─ 场景：简单应用                                       │
│                                                             │
│  3. RedisChatMessageHistory                                 │
│     └─ 存储：Redis                                          │
│     └─ 特点：分布式，高性能                                 │
│     └─ 场景：生产环境                                       │
│                                                             │
│  4. SQLChatMessageHistory                                   │
│     └─ 存储：SQL 数据库                                     │
│     └─ 特点：持久化，可查询                                 │
│     └─ 场景：需要历史分析                                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Memory 优化策略                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  问题：对话太长 → Token 超限 / 成本高                       │
│                                                             │
│  解决方案：                                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. 窗口截断：只保留最近 N 轮对话                     │   │
│  │    history = history[-10:]  # 保留最近 10 条        │   │
│  │                                                     │   │
│  │ 2. Token 限制：按 Token 数截断                      │   │
│  │    while count_tokens(history) > 4000:             │   │
│  │        history.pop(0)                               │   │
│  │                                                     │   │
│  │ 3. 摘要压缩：用 LLM 总结旧对话                      │   │
│  │    summary = llm("总结以下对话...")                 │   │
│  │    history = [summary] + recent_messages           │   │
│  │                                                     │   │
│  │ 4. 向量检索：相关历史按需加载                       │   │
│  │    relevant = vector_store.search(current_query)   │   │
│  └─────────────────────────────────────────────────────┘   │
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
    
    print(f"使用模型: {IFLOW_MODEL}")
    
    demo_1_without_memory()
    demo_2_manual_history()
    demo_3_chat_message_history()
    demo_4_runnable_with_history()
    demo_5_agent_with_memory()
    demo_6_memory_types()
    
    print("\n" + "=" * 60)
    print("Memory 教程完成！")
    print("=" * 60)
    print("""
核心要点：

1. LLM 本身无状态，需要 Memory 管理对话历史
2. 每次调用时把历史消息一起发给 LLM
3. 不同 session_id 可以有独立的对话历史
4. 长对话需要截断/压缩策略，避免 Token 超限

推荐方式：
- Chain: RunnableWithMessageHistory
- Agent: MemorySaver (LangGraph)

下一步：学习 RAG（检索增强生成）
""")

"""
Workflow vs Agentic 对比示例

运行前：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/workflow_vs_agentic.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 加载环境变量
load_dotenv()

# 心流平台配置
IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "TBStars2-200B-A13B")


def get_llm():
    """获取配置好的 LLM 实例"""
    return ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )


# ============================================================
# 示例 1: Workflow 模式 - 固定流程
# ============================================================

def workflow_example():
    """
    Workflow: 预定义的固定流程
    流程: 用户问题 → 分类 → 根据类别回答
    
    特点:
    - 每一步都是确定的
    - 开发者控制流程走向
    - 可预测、易调试
    """
    llm = get_llm()
    
    # Step 1: 分类
    classify_prompt = ChatPromptTemplate.from_template(
        "将以下问题分类为 'technical' 或 'general':\n{question}\n只输出分类结果，不要其他内容。"
    )
    
    # Step 2: 根据分类选择不同的回答模板
    technical_prompt = ChatPromptTemplate.from_template(
        "作为技术专家，用2-3句话简洁回答这个技术问题:\n{question}"
    )
    
    general_prompt = ChatPromptTemplate.from_template(
        "用简单易懂的语言，2-3句话回答这个问题:\n{question}"
    )
    
    # 构建 Chain（固定流程）
    classify_chain = classify_prompt | llm | StrOutputParser()
    technical_chain = technical_prompt | llm | StrOutputParser()
    general_chain = general_prompt | llm | StrOutputParser()
    
    def run_workflow(question: str) -> str:
        # 流程是写死的：先分类，再根据结果选择
        category = classify_chain.invoke({"question": question})
        print(f"[Workflow] 分类结果: {category.strip()}")
        
        if "technical" in category.lower():
            return technical_chain.invoke({"question": question})
        else:
            return general_chain.invoke({"question": question})
    
    return run_workflow


# ============================================================
# 示例 2: Agentic 模式 - LLM 自主决策
# ============================================================

@tool
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    return f"搜索结果: 关于 '{query}' 的最新信息是..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，如 '2+3*4'"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 模拟天气数据
    return f"{city}今天晴，温度 25°C，适合外出"


def agentic_example():
    """
    Agentic: LLM 自主决策使用什么工具、执行几次
    
    特点:
    - LLM 决定下一步做什么
    - 可以循环：思考 → 行动 → 观察 → 再思考
    - 灵活但不可预测
    """
    llm = get_llm()
    tools = [search_web, calculate, get_weather]
    
    # 使用 LangGraph 的 ReAct Agent
    agent = create_react_agent(llm, tools)
    
    return agent


# ============================================================
# 对比运行
# ============================================================

if __name__ == "__main__":
    # 检查 API Key
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print(f"使用模型: {IFLOW_MODEL}")
    print(f"API 地址: {IFLOW_BASE_URL}")
    print()
    
    # ========== Workflow 示例 ==========
    print("=" * 60)
    print("Workflow 示例: 固定流程 (分类 → 回答)")
    print("=" * 60)
    
    workflow = workflow_example()
    result = workflow("Python 的 GIL 是什么？")
    print(f"回答: {result}")
    
    # ========== Agentic 示例 ==========
    print("\n" + "=" * 60)
    print("Agentic 示例: LLM 自主决策")
    print("=" * 60)
    
    agent = agentic_example()
    
    # Agent 会自己决定：用 get_weather 获取天气，再用 calculate 计算
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京今天天气怎么样？如果温度超过20度，帮我算一下 (25-20)*1.5"}]
    })
    
    # 提取最终回答
    final_message = result["messages"][-1].content
    print(f"回答: {final_message}")


# ============================================================
# 关键区别总结
# ============================================================
"""
┌─────────────────┬────────────────────────────────────────┐
│                 │  Workflow              │  Agentic      │
├─────────────────┼────────────────────────┼───────────────┤
│ 流程控制        │  代码写死              │  LLM 决定     │
│ 工具调用        │  按顺序调用            │  按需调用     │
│ 循环能力        │  需要显式写循环        │  自动循环     │
│ 错误处理        │  需要写 try-catch      │  可以自我纠正 │
│ Token 消耗      │  可预测                │  不可预测     │
│ 调试难度        │  简单                  │  困难         │
└─────────────────┴────────────────────────┴───────────────┘

生产建议:
1. 能用 Workflow 解决的，不要用 Agentic
2. Agentic 适合探索性任务、复杂推理
3. 可以混合使用：Workflow 调度多个小 Agent
"""

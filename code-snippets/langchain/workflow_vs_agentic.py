"""
Workflow vs Agentic 对比示例

运行前安装依赖：
pip install langchain langchain-openai python-dotenv

配置：在项目根目录创建 .env 文件（已在 .gitignore 中）
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain import hub

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
        "将以下问题分类为 'technical' 或 'general':\n{question}\n只输出分类结果。"
    )
    
    # Step 2: 根据分类选择不同的回答模板
    technical_prompt = ChatPromptTemplate.from_template(
        "作为技术专家，详细回答这个技术问题:\n{question}"
    )
    
    general_prompt = ChatPromptTemplate.from_template(
        "用简单易懂的语言回答这个问题:\n{question}"
    )
    
    # 构建 Chain（固定流程）
    classify_chain = classify_prompt | llm | StrOutputParser()
    technical_chain = technical_prompt | llm | StrOutputParser()
    general_chain = general_prompt | llm | StrOutputParser()
    
    def run_workflow(question: str) -> str:
        # 流程是写死的：先分类，再根据结果选择
        category = classify_chain.invoke({"question": question})
        print(f"[Workflow] 分类结果: {category}")
        
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
    # 模拟搜索结果
    return f"搜索结果: 关于 '{query}' 的最新信息..."

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    # 模拟天气数据
    return f"{city}今天晴，温度 25°C"


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
    
    # 使用 ReAct 提示词模板
    prompt = hub.pull("hwchase17/react")
    
    # 创建 Agent（LLM 自己决定流程）
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


# ============================================================
# 对比运行
# ============================================================

if __name__ == "__main__":
    # 检查 API Key
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        print("示例:")
        print("  IFLOW_API_KEY=sk-xxx")
        print("  IFLOW_BASE_URL=https://apis.iflow.cn/v1")
        print("  IFLOW_MODEL=TBStars2-200B-A13B")
        exit(1)
    
    print(f"使用模型: {IFLOW_MODEL}")
    print(f"API 地址: {IFLOW_BASE_URL}")
    print()
    
    print("=" * 60)
    print("Workflow 示例")
    print("=" * 60)
    
    workflow = workflow_example()
    
    # Workflow: 永远按 分类→回答 的流程执行
    result = workflow("Python 的 GIL 是什么？")
    print(f"结果: {result[:500]}...")
    
    print("\n" + "=" * 60)
    print("Agentic 示例")
    print("=" * 60)
    
    agent = agentic_example()
    
    # Agent: 自己决定要不要用工具、用哪个、用几次
    result = agent.invoke({
        "input": "北京今天天气怎么样？如果温度超过20度，帮我算一下 (25-20)*1.5"
    })
    print(f"结果: {result['output']}")


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

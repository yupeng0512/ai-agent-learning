"""
LangChain 教程 - 阶段 2：Tool 和 Agent
理解 Agent 如何调用工具完成任务

运行前：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/02_tools_and_agents.py
"""

import os
import json
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "TBStars2-200B-A13B")


# ============================================================
# 第一部分：什么是 Tool？
# ============================================================

def demo_1_what_is_tool():
    """
    Tool = Agent 的"手"
    
    概念：
    - Tool 是 Agent 可以调用的函数
    - LLM 决定"调用哪个 Tool"和"传什么参数"
    - Tool 执行后返回结果给 LLM
    
    流程：
    用户问题 → LLM 思考 → 选择 Tool → 执行 → 返回结果 → LLM 总结
    """
    print("\n" + "=" * 60)
    print("Demo 1: 什么是 Tool？")
    print("=" * 60)
    
    from langchain_core.tools import tool
    
    # 使用 @tool 装饰器定义工具
    @tool
    def add(a: int, b: int) -> int:
        """两个数相加"""
        return a + b
    
    @tool
    def multiply(a: int, b: int) -> int:
        """两个数相乘"""
        return a * b
    
    # 查看 Tool 的属性
    print(f"Tool 名称: {add.name}")
    print(f"Tool 描述: {add.description}")
    print(f"Tool 参数: {add.args}")
    
    # 直接调用 Tool（测试）
    result = add.invoke({"a": 3, "b": 5})
    print(f"\n直接调用 add(3, 5) = {result}")
    
    print("""
关键概念：
┌─────────────────────────────────────────────────────────┐
│  @tool 装饰器会自动提取：                                │
│  1. 函数名 → Tool 名称                                  │
│  2. docstring → Tool 描述（LLM 用来理解功能）           │
│  3. 参数类型 → Tool 参数 schema                         │
└─────────────────────────────────────────────────────────┘
""")


# ============================================================
# 第二部分：创建实用的 Tool
# ============================================================

def demo_2_practical_tools():
    """
    创建实用的 Tool 示例
    """
    print("\n" + "=" * 60)
    print("Demo 2: 实用 Tool 示例")
    print("=" * 60)
    
    from langchain_core.tools import tool
    from datetime import datetime
    
    @tool
    def get_current_time() -> str:
        """获取当前时间，格式：YYYY-MM-DD HH:MM:SS"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @tool
    def search_weather(city: str) -> str:
        """
        查询指定城市的天气。
        
        Args:
            city: 城市名称，如"北京"、"上海"
        
        Returns:
            天气信息字符串
        """
        # 模拟天气 API（实际应调用真实 API）
        weather_data = {
            "北京": "晴，15°C，空气质量良好",
            "上海": "多云，18°C，有轻度雾霾",
            "深圳": "阴，22°C，可能有小雨",
        }
        return weather_data.get(city, f"未找到 {city} 的天气信息")
    
    @tool
    def calculate_expression(expression: str) -> str:
        """
        计算数学表达式。
        
        Args:
            expression: 数学表达式，如 "2 + 3 * 4"
        
        Returns:
            计算结果
        """
        try:
            # 安全起见，只允许基本运算
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误：{str(e)}"
    
    # 测试这些 Tool
    print("测试 Tool：")
    print(f"  当前时间: {get_current_time.invoke({})}")
    print(f"  北京天气: {search_weather.invoke({'city': '北京'})}")
    print(f"  计算 2+3*4: {calculate_expression.invoke({'expression': '2 + 3 * 4'})}")
    
    print("""
Tool 设计要点：
1. 描述要清晰 - LLM 靠描述理解何时使用
2. 参数要明确 - 类型注解帮助 LLM 传参
3. 返回要有意义 - 结果要能帮助 LLM 回答问题
""")
    
    return [get_current_time, search_weather, calculate_expression]


# ============================================================
# 第三部分：Agent 调用 Tool（ReAct 模式）
# ============================================================

def demo_3_agent_with_tools():
    """
    Agent + Tool：让 LLM 自主决定调用什么工具
    
    ReAct 模式：
    Thought → Action → Observation → Thought → ... → Final Answer
    """
    print("\n" + "=" * 60)
    print("Demo 3: Agent 调用 Tool（ReAct）")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 定义工具
    @tool
    def search_weather(city: str) -> str:
        """查询城市天气"""
        weather_data = {
            "北京": "晴，15°C",
            "上海": "多云，18°C",
            "深圳": "阴，22°C",
        }
        return weather_data.get(city, f"未找到 {city} 的天气")
    
    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式，如 '2 + 3 * 4'"""
        try:
            return str(eval(expression))
        except:
            return "计算错误"
    
    # 创建 ReAct Agent
    tools = [search_weather, calculate]
    agent = create_react_agent(llm, tools)
    
    # 测试 1：需要调用天气工具
    print("\n【测试 1】问：北京今天天气怎么样？")
    result = agent.invoke({"messages": [("human", "北京今天天气怎么样？")]})
    print(f"答：{result['messages'][-1].content}")
    
    # 测试 2：需要调用计算工具
    print("\n【测试 2】问：帮我算一下 15 * 8 + 32")
    result = agent.invoke({"messages": [("human", "帮我算一下 15 * 8 + 32")]})
    print(f"答：{result['messages'][-1].content}")
    
    # 测试 3：不需要工具
    print("\n【测试 3】问：你好，介绍一下你自己")
    result = agent.invoke({"messages": [("human", "你好，介绍一下你自己")]})
    print(f"答：{result['messages'][-1].content}")
    
    print("""
ReAct Agent 执行流程：
┌─────────────────────────────────────────────────────────┐
│  用户: "北京天气怎么样？"                                │
│     ↓                                                   │
│  LLM 思考: 需要查天气，调用 search_weather              │
│     ↓                                                   │
│  执行 Tool: search_weather("北京")                      │
│     ↓                                                   │
│  Tool 返回: "晴，15°C"                                  │
│     ↓                                                   │
│  LLM 总结: "北京今天天气晴朗，气温15度"                  │
└─────────────────────────────────────────────────────────┘
""")


# ============================================================
# 第四部分：Tool 的高级用法
# ============================================================

def demo_4_advanced_tools():
    """
    Tool 高级用法：
    1. 带复杂参数的 Tool
    2. 返回结构化数据的 Tool
    3. 异步 Tool
    """
    print("\n" + "=" * 60)
    print("Demo 4: Tool 高级用法")
    print("=" * 60)
    
    from langchain_core.tools import tool, StructuredTool
    from pydantic import BaseModel, Field
    from typing import List
    
    # 方式 1：使用 Pydantic 定义复杂参数
    class SearchParams(BaseModel):
        """搜索参数"""
        query: str = Field(description="搜索关键词")
        max_results: int = Field(default=5, description="最大返回数量")
        category: Optional[str] = Field(default=None, description="分类过滤")
    
    @tool(args_schema=SearchParams)
    def advanced_search(query: str, max_results: int = 5, category: Optional[str] = None) -> str:
        """
        高级搜索功能，支持关键词、数量限制和分类过滤。
        """
        result = f"搜索 '{query}'，最多 {max_results} 条"
        if category:
            result += f"，分类: {category}"
        return result
    
    print("高级搜索 Tool：")
    print(f"  参数 Schema: {advanced_search.args}")
    print(f"  调用结果: {advanced_search.invoke({'query': 'AI Agent', 'max_results': 3, 'category': '技术'})}")
    
    # 方式 2：使用 StructuredTool 从函数创建
    def fetch_user_info(user_id: int, include_details: bool = False) -> dict:
        """获取用户信息"""
        user = {"id": user_id, "name": f"用户{user_id}"}
        if include_details:
            user["email"] = f"user{user_id}@example.com"
        return user
    
    user_tool = StructuredTool.from_function(
        func=fetch_user_info,
        name="get_user",
        description="根据用户ID获取用户信息"
    )
    
    print(f"\nStructuredTool：")
    print(f"  调用结果: {user_tool.invoke({'user_id': 123, 'include_details': True})}")
    
    print("""
Tool 创建方式对比：
┌──────────────────────┬─────────────────────────────────┐
│  方式                 │  适用场景                       │
├──────────────────────┼─────────────────────────────────┤
│  @tool 装饰器         │  简单工具，快速定义             │
│  @tool + args_schema │  复杂参数，需要详细描述         │
│  StructuredTool      │  从现有函数创建，更灵活         │
└──────────────────────┴─────────────────────────────────┘
""")


# ============================================================
# 第五部分：多 Tool 协作
# ============================================================

def demo_5_multi_tool_agent():
    """
    多 Tool 协作：Agent 自动选择和组合多个工具
    """
    print("\n" + "=" * 60)
    print("Demo 5: 多 Tool 协作")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langgraph.prebuilt import create_react_agent
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 定义多个相关工具
    @tool
    def get_stock_price(symbol: str) -> str:
        """获取股票当前价格。symbol: 股票代码，如 AAPL、GOOGL"""
        prices = {"AAPL": 178.5, "GOOGL": 141.2, "MSFT": 378.9, "TSLA": 248.3}
        price = prices.get(symbol.upper())
        if price:
            return f"{symbol.upper()} 当前价格: ${price}"
        return f"未找到 {symbol} 的股票信息"
    
    @tool
    def get_company_info(company: str) -> str:
        """获取公司基本信息。company: 公司名称"""
        info = {
            "apple": "Apple Inc. 是一家美国科技公司，主要产品包括 iPhone、Mac、iPad",
            "google": "Google 是 Alphabet 旗下的搜索引擎和科技公司",
            "microsoft": "Microsoft 是全球最大的软件公司之一，产品包括 Windows、Office",
        }
        return info.get(company.lower(), f"未找到 {company} 的信息")
    
    @tool
    def calculate_return(buy_price: float, current_price: float, shares: int) -> str:
        """计算股票收益。buy_price: 买入价，current_price: 当前价，shares: 股数"""
        profit = (current_price - buy_price) * shares
        percent = ((current_price - buy_price) / buy_price) * 100
        return f"收益: ${profit:.2f} ({percent:.1f}%)"
    
    # 创建 Agent
    tools = [get_stock_price, get_company_info, calculate_return]
    agent = create_react_agent(llm, tools)
    
    # 测试：需要组合多个工具
    print("【测试】问：我以 150 美元买了 100 股苹果股票，现在赚了多少？")
    result = agent.invoke({
        "messages": [("human", "我以 150 美元买了 100 股苹果股票，现在赚了多少？先查一下苹果现在的股价")]
    })
    print(f"答：{result['messages'][-1].content}")
    
    print("""
多 Tool 协作流程：
┌─────────────────────────────────────────────────────────┐
│  用户: "我以150美元买了100股苹果，现在赚了多少？"         │
│     ↓                                                   │
│  LLM: 需要先查股价 → 调用 get_stock_price("AAPL")       │
│     ↓                                                   │
│  Tool 返回: "AAPL 当前价格: $178.5"                     │
│     ↓                                                   │
│  LLM: 计算收益 → 调用 calculate_return(150, 178.5, 100) │
│     ↓                                                   │
│  Tool 返回: "收益: $2850.00 (19.0%)"                    │
│     ↓                                                   │
│  LLM: 组合结果回答用户                                   │
└─────────────────────────────────────────────────────────┘
""")


# ============================================================
# 运行所有 Demo
# ============================================================

if __name__ == "__main__":
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print(f"使用模型: {IFLOW_MODEL}")
    
    demo_1_what_is_tool()
    demo_2_practical_tools()
    demo_3_agent_with_tools()
    demo_4_advanced_tools()
    demo_5_multi_tool_agent()
    
    print("\n" + "=" * 60)
    print("Tool 和 Agent 教程完成！")
    print("=" * 60)
    print("""
知识点回顾：

1. Tool 定义
   - @tool 装饰器
   - docstring 是关键（LLM 靠它理解功能）
   - 参数类型注解

2. Agent 调用 Tool
   - create_react_agent 创建 ReAct Agent
   - LLM 自主决定调用哪个 Tool
   - Tool 返回结果供 LLM 使用

3. 高级用法
   - args_schema 定义复杂参数
   - StructuredTool 从函数创建
   - 多 Tool 协作

下一步：学习 Memory（记忆）和 RAG
""")

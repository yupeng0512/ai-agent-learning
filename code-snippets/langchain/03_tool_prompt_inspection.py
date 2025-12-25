"""
Tool Prompt 检查 - 看看 LLM 实际收到什么信息

这个脚本展示：
1. Tool 的哪些信息会传给 LLM
2. LLM 是如何"看到" Tool 的
3. 为什么 docstring 如此重要

运行：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/03_tool_prompt_inspection.py
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# ============================================================
# 定义几个 Tool
# ============================================================

@tool
def calculate_area(length: float, width: float) -> float:
    """计算矩形面积。当用户需要计算长方形或矩形的面积时使用。
    
    Args:
        length: 矩形的长度（单位：米）
        width: 矩形的宽度（单位：米）
    """
    # 这里的代码 LLM 完全看不到！
    result = length * width
    print(f"[内部日志] 计算 {length} x {width} = {result}")
    return result


@tool
def search_weather(city: str) -> str:
    """查询指定城市的天气。当用户询问某个城市的天气情况时使用。
    
    Args:
        city: 城市名称，如"北京"、"上海"
    """
    # 这里的实现 LLM 也看不到
    # 可能是调用 API，可能是返回假数据
    return f"{city}今天晴，温度 25°C"


@tool  
def bad_tool_example(x):
    """处理数据"""  # 描述太简单！
    return x * 2


# ============================================================
# 检查 Tool 的元信息
# ============================================================

def inspect_tool_metadata():
    """查看 Tool 的元数据 - 这就是 LLM 能看到的全部"""
    print("=" * 60)
    print("Tool 元数据检查 - LLM 能看到的信息")
    print("=" * 60)
    
    tools = [calculate_area, search_weather, bad_tool_example]
    
    for t in tools:
        print(f"\n📦 Tool: {t.name}")
        print(f"   描述: {t.description}")
        print(f"   参数 Schema: {t.args_schema.schema() if t.args_schema else 'None'}")
        print("-" * 50)


# ============================================================
# 查看实际发送给 LLM 的 Prompt
# ============================================================

def show_agent_prompt():
    """展示 Agent 实际发送给 LLM 的 Prompt"""
    print("\n" + "=" * 60)
    print("Agent 发送给 LLM 的 Prompt（简化版）")
    print("=" * 60)
    
    tools = [calculate_area, search_weather]
    
    # 模拟 Agent 构建的 Tool 描述
    tool_descriptions = []
    for t in tools:
        desc = f"- {t.name}: {t.description}"
        if t.args_schema:
            args = t.args_schema.schema().get("properties", {})
            args_str = ", ".join([f"{k}: {v.get('type', 'any')}" for k, v in args.items()])
            desc += f"\n  参数: {args_str}"
        tool_descriptions.append(desc)
    
    prompt = f"""你是一个有用的助手，可以使用以下工具：

{chr(10).join(tool_descriptions)}

当你需要使用工具时，请按以下格式输出：
Thought: 我需要...
Action: 工具名称
Action Input: {{"参数": "值"}}

用户问题：计算一个 5 米长、3 米宽的房间面积
"""
    
    print(prompt)
    print("-" * 60)
    print("👆 注意：LLM 只看到 Tool 的名称、描述、参数")
    print("   函数内部的 print、计算逻辑等，LLM 完全不知道！")


# ============================================================
# 演示描述质量的影响
# ============================================================

def demo_description_quality():
    """演示 Tool 描述质量的重要性"""
    print("\n" + "=" * 60)
    print("Tool 描述质量对比")
    print("=" * 60)
    
    print("""
❌ 差的描述：
   @tool
   def process(x):
       \"\"\"处理数据\"\"\"
       ...
   
   问题：
   - LLM 不知道"处理"是什么意思
   - 不知道什么时候该用这个 Tool
   - 不知道参数 x 应该传什么

✅ 好的描述：
   @tool
   def calculate_area(length: float, width: float) -> float:
       \"\"\"计算矩形面积。当用户需要计算长方形或矩形的面积时使用。
       
       Args:
           length: 矩形的长度（单位：米）
           width: 矩形的宽度（单位：米）
       \"\"\"
       ...
   
   优点：
   - 清楚说明功能：计算矩形面积
   - 说明使用场景：用户需要计算面积时
   - 参数有类型注解和说明
""")


# ============================================================
# 实际运行 Agent 看效果
# ============================================================

def run_agent_demo():
    """运行 Agent 演示 Tool 选择"""
    print("\n" + "=" * 60)
    print("实际运行 Agent - 观察 Tool 选择")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    
    IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
    IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
    IFLOW_MODEL = os.getenv("IFLOW_MODEL", "TBStars2-200B-A13B")
    
    if not IFLOW_API_KEY:
        print("跳过：未配置 IFLOW_API_KEY")
        return
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    tools = [calculate_area, search_weather]
    agent = create_react_agent(llm, tools)
    
    # 测试 1：应该选择 calculate_area
    print("\n问题 1: 我的房间长 5 米，宽 3 米，面积是多少？")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "我的房间长 5 米，宽 3 米，面积是多少？"}]
    })
    print(f"回答: {result['messages'][-1].content}")
    
    # 测试 2：应该选择 search_weather
    print("\n问题 2: 北京今天天气怎么样？")
    result = agent.invoke({
        "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
    })
    print(f"回答: {result['messages'][-1].content}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    inspect_tool_metadata()
    show_agent_prompt()
    demo_description_quality()
    run_agent_demo()
    
    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────────┐
│  LLM 选择 Tool 的依据：                                      │
│                                                             │
│  1. Tool 名称 (name)                                        │
│  2. Tool 描述 (docstring)  ← 最重要！                       │
│  3. 参数名称和类型                                          │
│  4. 参数描述 (Args 部分)                                    │
│                                                             │
│  LLM 完全不知道：                                            │
│  - 函数内部的代码逻辑                                        │
│  - 实际的 API 调用                                          │
│  - 数据处理过程                                             │
│                                                             │
│  所以：描述写得好 = Agent 选对工具                           │
│       描述写得差 = Agent 乱选或不选                          │
└─────────────────────────────────────────────────────────────┘
""")

"""
Agent 架构对比：ReAct vs Plan-and-Execute

运行前：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/agent_architectures.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from typing import List
import json

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
# 共用的模拟工具
# ============================================================

@tool
def search_company_info(company_name: str) -> str:
    """搜索公司基本信息"""
    # 模拟数据
    mock_data = {
        "langchain": "LangChain Inc. - 成立于2022年，融资3500万美元，主打LLM应用开发框架",
        "cohere": "Cohere - 成立于2019年，融资4.5亿美元，专注企业级LLM和RAG",
        "anthropic": "Anthropic - 成立于2021年，融资超40亿美元，开发Claude系列模型",
    }
    key = company_name.lower()
    for k, v in mock_data.items():
        if k in key:
            return v
    return f"未找到 {company_name} 的信息"


@tool
def search_product_info(company_name: str) -> str:
    """搜索公司产品信息"""
    mock_data = {
        "langchain": "产品：LangChain框架、LangSmith(调试平台)、LangGraph(工作流)。特点：开源生态最大",
        "cohere": "产品：Command(生成)、Embed(向量)、Rerank(重排序)。特点：企业级RAG方案",
        "anthropic": "产品：Claude 3系列(Opus/Sonnet/Haiku)、Claude API。特点：安全对齐领先",
    }
    key = company_name.lower()
    for k, v in mock_data.items():
        if k in key:
            return v
    return f"未找到 {company_name} 的产品信息"


@tool
def compare_products(products_info: str) -> str:
    """对比多个产品的优劣势"""
    return f"对比分析：根据提供的信息，各产品定位不同。LangChain专注开发框架，Cohere专注企业RAG，Anthropic专注模型能力。"


@tool
def write_report(content: str) -> str:
    """生成分析报告"""
    return f"报告已生成，包含：公司概况、产品对比、投资建议等章节"


# ============================================================
# 架构 1: ReAct - 逐步反应（回顾）
# ============================================================

def react_agent_demo():
    """
    ReAct: 思考→行动→观察 循环
    
    问题：
    1. 没有全局规划，容易迷失
    2. 不知道什么时候该结束
    3. 复杂任务容易遗漏步骤
    """
    print("\n" + "=" * 60)
    print("架构 1: ReAct（逐步反应）")
    print("=" * 60)
    
    llm = get_llm()
    tools = [search_company_info, search_product_info, compare_products, write_report]
    
    # ReAct Agent - 每一步都是 LLM 临时决定
    agent = create_react_agent(llm, tools)
    
    result = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "帮我调研 LangChain、Cohere、Anthropic 这3家AI公司，对比产品，写分析报告"
        }]
    })
    
    print("\n【ReAct 执行过程】")
    for msg in result["messages"]:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  → 调用工具: {tc['name']}({tc['args']})")
        elif msg.type == "tool":
            print(f"  ← 工具返回: {msg.content[:50]}...")
    
    print(f"\n【最终回答】\n{result['messages'][-1].content[:500]}...")
    
    return result


# ============================================================
# 架构 2: Plan-and-Execute - 先规划后执行
# ============================================================

def plan_and_execute_demo():
    """
    Plan-and-Execute: 先制定计划，再逐步执行
    
    优势：
    1. 有全局视角，不会遗漏
    2. 计划可以被检查和修正
    3. 执行过程更可控
    """
    print("\n" + "=" * 60)
    print("架构 2: Plan-and-Execute（先规划后执行）")
    print("=" * 60)
    
    llm = get_llm()
    tools = [search_company_info, search_product_info, compare_products, write_report]
    
    # ========== 第一阶段：规划 ==========
    planner_prompt = ChatPromptTemplate.from_template("""
你是一个任务规划专家。根据用户的请求，制定一个详细的执行计划。

用户请求：{task}

可用工具：
- search_company_info: 搜索公司基本信息
- search_product_info: 搜索公司产品信息  
- compare_products: 对比多个产品
- write_report: 生成报告

请输出一个 JSON 格式的计划，包含步骤列表：
{{"steps": [
    {{"step": 1, "action": "工具名", "input": "参数", "purpose": "目的"}},
    ...
]}}

只输出 JSON，不要其他内容。
""")
    
    planner_chain = planner_prompt | llm | StrOutputParser()
    
    task = "帮我调研 LangChain、Cohere、Anthropic 这3家AI公司，对比产品，写分析报告"
    
    print("\n【第一阶段：制定计划】")
    plan_text = planner_chain.invoke({"task": task})
    print(f"生成的计划：\n{plan_text}")
    
    # 解析计划
    try:
        # 提取 JSON 部分
        import re
        json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
        else:
            plan = json.loads(plan_text)
    except json.JSONDecodeError:
        print("计划解析失败，使用默认计划")
        plan = {
            "steps": [
                {"step": 1, "action": "search_company_info", "input": "LangChain", "purpose": "获取LangChain公司信息"},
                {"step": 2, "action": "search_company_info", "input": "Cohere", "purpose": "获取Cohere公司信息"},
                {"step": 3, "action": "search_company_info", "input": "Anthropic", "purpose": "获取Anthropic公司信息"},
                {"step": 4, "action": "search_product_info", "input": "LangChain", "purpose": "获取LangChain产品信息"},
                {"step": 5, "action": "search_product_info", "input": "Cohere", "purpose": "获取Cohere产品信息"},
                {"step": 6, "action": "search_product_info", "input": "Anthropic", "purpose": "获取Anthropic产品信息"},
                {"step": 7, "action": "compare_products", "input": "all", "purpose": "对比三家产品"},
                {"step": 8, "action": "write_report", "input": "all", "purpose": "生成报告"},
            ]
        }
    
    # ========== 第二阶段：执行 ==========
    print("\n【第二阶段：按计划执行】")
    
    tool_map = {
        "search_company_info": search_company_info,
        "search_product_info": search_product_info,
        "compare_products": compare_products,
        "write_report": write_report,
    }
    
    results = []
    for step in plan.get("steps", []):
        action = step.get("action", "")
        input_val = step.get("input", "")
        purpose = step.get("purpose", "")
        
        print(f"\n  步骤 {step.get('step', '?')}: {purpose}")
        print(f"    → 执行: {action}({input_val})")
        
        if action in tool_map:
            result = tool_map[action].invoke(input_val)
            print(f"    ← 结果: {result[:60]}...")
            results.append({"step": step.get("step"), "result": result})
    
    # ========== 第三阶段：整合 ==========
    print("\n【第三阶段：整合结果】")
    
    synthesize_prompt = ChatPromptTemplate.from_template("""
根据以下收集到的信息，生成一份简洁的分析报告：

{results}

要求：
1. 包含公司概况
2. 产品对比表格
3. 简要结论

用中文输出，控制在300字以内。
""")
    
    synthesize_chain = synthesize_prompt | llm | StrOutputParser()
    
    results_text = "\n".join([f"步骤{r['step']}: {r['result']}" for r in results])
    final_report = synthesize_chain.invoke({"results": results_text})
    
    print(f"\n【最终报告】\n{final_report}")
    
    return final_report


# ============================================================
# 架构 3: Reflexion - 自我反思（简化演示）
# ============================================================

def reflexion_demo():
    """
    Reflexion: 执行后反思，从错误中学习
    
    流程：执行 → 评估 → 反思 → 改进 → 再执行
    
    适用：需要迭代优化的任务
    """
    print("\n" + "=" * 60)
    print("架构 3: Reflexion（自我反思）")
    print("=" * 60)
    
    llm = get_llm()
    
    # 模拟一个需要反思改进的任务
    task = "写一段介绍 AI Agent 的文字，要求通俗易懂"
    
    # 第一次尝试
    first_attempt_prompt = ChatPromptTemplate.from_template(
        "请完成以下任务：{task}\n\n直接输出结果，控制在100字以内。"
    )
    first_chain = first_attempt_prompt | llm | StrOutputParser()
    
    print(f"\n任务：{task}")
    first_result = first_chain.invoke({"task": task})
    print(f"\n【第一次尝试】\n{first_result}")
    
    # 反思评估
    reflect_prompt = ChatPromptTemplate.from_template("""
评估以下输出是否满足要求，并给出改进建议：

任务要求：{task}

当前输出：
{output}

请指出：
1. 做得好的地方
2. 需要改进的地方
3. 具体改进建议

简洁输出，控制在100字以内。
""")
    reflect_chain = reflect_prompt | llm | StrOutputParser()
    
    reflection = reflect_chain.invoke({"task": task, "output": first_result})
    print(f"\n【自我反思】\n{reflection}")
    
    # 根据反思改进
    improve_prompt = ChatPromptTemplate.from_template("""
根据反思意见，改进你的输出：

原任务：{task}
原输出：{original}
反思意见：{reflection}

请输出改进后的版本，控制在100字以内。
""")
    improve_chain = improve_prompt | llm | StrOutputParser()
    
    improved_result = improve_chain.invoke({
        "task": task,
        "original": first_result,
        "reflection": reflection
    })
    print(f"\n【改进后】\n{improved_result}")
    
    return improved_result


# ============================================================
# 主程序：对比运行
# ============================================================

if __name__ == "__main__":
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print(f"使用模型: {IFLOW_MODEL}")
    print(f"API 地址: {IFLOW_BASE_URL}")
    
    # 选择要运行的架构
    print("\n请选择要演示的架构：")
    print("1. ReAct（逐步反应）")
    print("2. Plan-and-Execute（先规划后执行）")
    print("3. Reflexion（自我反思）")
    print("4. 全部运行")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == "1":
        react_agent_demo()
    elif choice == "2":
        plan_and_execute_demo()
    elif choice == "3":
        reflexion_demo()
    elif choice == "4":
        react_agent_demo()
        plan_and_execute_demo()
        reflexion_demo()
    else:
        print("运行默认: Plan-and-Execute")
        plan_and_execute_demo()


# ============================================================
# 架构选型指南
# ============================================================
"""
┌─────────────────────┬─────────────────────────────────────────────┐
│ 架构                │ 适用场景                                     │
├─────────────────────┼─────────────────────────────────────────────┤
│ ReAct               │ • 简单的工具调用任务                         │
│                     │ • 步骤少、依赖少的任务                       │
│                     │ • 需要快速响应的场景                         │
├─────────────────────┼─────────────────────────────────────────────┤
│ Plan-and-Execute    │ • 复杂的多步骤任务                           │
│                     │ • 需要全局规划的任务（如调研、分析）          │
│                     │ • 步骤之间有依赖关系                         │
├─────────────────────┼─────────────────────────────────────────────┤
│ Reflexion           │ • 需要迭代优化的任务（如写作、代码）          │
│                     │ • 质量要求高的任务                           │
│                     │ • 有明确评估标准的任务                       │
├─────────────────────┼─────────────────────────────────────────────┤
│ LATS                │ • 复杂推理任务                               │
│ (树搜索)            │ • 需要探索多条路径                           │
│                     │ • 数学证明、代码生成等                       │
└─────────────────────┴─────────────────────────────────────────────┘

面试常见问题：
Q: ReAct 和 Plan-and-Execute 的核心区别？
A: ReAct 是"边走边看"，每一步都由 LLM 临时决定；
   Plan-and-Execute 是"先画蓝图再施工"，有全局规划。

Q: 什么时候用 Reflexion？
A: 当任务需要迭代优化、有明确质量标准时。比如写代码、写文章。
   Reflexion 的核心是"做完了回头看，看完了再改进"。

Q: 生产环境中如何选择？
A: 
   1. 能用简单架构解决的，不要用复杂的（成本、延迟）
   2. 可以混合使用：Plan-and-Execute 规划 + ReAct 执行每个步骤
   3. 关键是匹配任务特点，不是越复杂越好
"""

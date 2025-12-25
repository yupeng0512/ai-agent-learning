"""
LangChain 基础教程 - 阶段 1
从零开始理解 LangChain 核心概念

运行前：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/01_langchain_basics.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "TBStars2-200B-A13B")


# ============================================================
# 第一部分：最基础的 LLM 调用
# ============================================================

def demo_1_basic_llm():
    """
    最简单的 LLM 调用
    
    概念：ChatOpenAI 是 LangChain 对 OpenAI API 的封装
    """
    print("\n" + "=" * 60)
    print("Demo 1: 基础 LLM 调用")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    
    # 创建 LLM 实例
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 最简单的调用方式
    response = llm.invoke("用一句话解释什么是 AI Agent")
    
    print(f"输入: 用一句话解释什么是 AI Agent")
    print(f"输出: {response.content}")
    
    # 查看返回对象的结构
    print(f"\n返回类型: {type(response)}")
    print(f"Token 使用: {response.response_metadata.get('token_usage', 'N/A')}")


# ============================================================
# 第二部分：Prompt Template（提示词模板）
# ============================================================

def demo_2_prompt_template():
    """
    使用 Prompt Template 管理提示词
    
    概念：
    - PromptTemplate：简单的字符串模板
    - ChatPromptTemplate：对话格式的模板（推荐）
    """
    print("\n" + "=" * 60)
    print("Demo 2: Prompt Template")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 方式 1：简单模板
    simple_prompt = ChatPromptTemplate.from_template(
        "你是一个{role}专家。请用简单的语言解释：{question}"
    )
    
    # 格式化模板
    formatted = simple_prompt.format(role="Python", question="什么是装饰器？")
    print(f"格式化后的 Prompt:\n{formatted}\n")
    
    # 方式 2：多角色模板（更常用）
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}专家，回答要简洁，不超过50字。"),
        ("human", "{question}")
    ])
    
    # 调用 LLM
    messages = chat_prompt.format_messages(role="AI Agent", question="ReAct 架构是什么？")
    response = llm.invoke(messages)
    
    print(f"System: 你是一个AI Agent专家...")
    print(f"Human: ReAct 架构是什么？")
    print(f"AI: {response.content}")


# ============================================================
# 第三部分：LCEL（LangChain Expression Language）
# ============================================================

def demo_3_lcel_chain():
    """
    LCEL：LangChain 的链式语法
    
    核心概念：使用 | 管道符连接组件
    
    prompt | llm | parser
       ↓      ↓      ↓
    格式化 → 调用 → 解析输出
    """
    print("\n" + "=" * 60)
    print("Demo 3: LCEL 链式语法")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 定义 Prompt
    prompt = ChatPromptTemplate.from_template(
        "给 {product} 起 3 个创意名字，用逗号分隔，只输出名字"
    )
    
    # 定义 Parser（输出解析器）
    parser = StrOutputParser()  # 最简单的解析器，直接返回字符串
    
    # 使用 LCEL 组合成 Chain
    chain = prompt | llm | parser
    
    # 调用 Chain
    result = chain.invoke({"product": "AI 编程助手"})
    
    print(f"产品: AI 编程助手")
    print(f"创意名字: {result}")
    
    # 展示 Chain 的结构
    print(f"\nChain 结构:")
    print(f"  prompt | llm | parser")
    print(f"     ↓      ↓      ↓")
    print(f"  格式化 → 调用 → 解析")


# ============================================================
# 第四部分：Output Parser（输出解析器）
# ============================================================

def demo_4_output_parser():
    """
    Output Parser：结构化输出
    
    常用 Parser：
    - StrOutputParser：返回纯字符串
    - JsonOutputParser：返回 JSON
    - PydanticOutputParser：返回 Pydantic 对象
    """
    print("\n" + "=" * 60)
    print("Demo 4: Output Parser 结构化输出")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from pydantic import BaseModel, Field
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    
    # 定义输出结构
    class BookRecommendation(BaseModel):
        title: str = Field(description="书名")
        author: str = Field(description="作者")
        reason: str = Field(description="推荐理由，一句话")
    
    # 创建 Parser
    parser = JsonOutputParser(pydantic_object=BookRecommendation)
    
    # 创建 Prompt（包含格式说明）
    prompt = ChatPromptTemplate.from_template(
        """推荐一本关于 {topic} 的书。

{format_instructions}

只输出 JSON，不要其他内容。"""
    )
    
    # 组合 Chain
    chain = prompt | llm | parser
    
    # 调用
    result = chain.invoke({
        "topic": "AI Agent 开发",
        "format_instructions": parser.get_format_instructions()
    })
    
    print(f"主题: AI Agent 开发")
    print(f"推荐结果:")
    print(f"  书名: {result.get('title', 'N/A')}")
    print(f"  作者: {result.get('author', 'N/A')}")
    print(f"  理由: {result.get('reason', 'N/A')}")


# ============================================================
# 第五部分：Chain 组合（多步骤处理）
# ============================================================

def demo_5_chain_composition():
    """
    Chain 组合：多个 Chain 串联
    
    场景：翻译 → 总结 → 格式化
    """
    print("\n" + "=" * 60)
    print("Demo 5: Chain 组合（多步骤处理）")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )
    parser = StrOutputParser()
    
    # Chain 1: 翻译
    translate_prompt = ChatPromptTemplate.from_template(
        "将以下英文翻译成中文，只输出翻译结果：\n{text}"
    )
    translate_chain = translate_prompt | llm | parser
    
    # Chain 2: 总结
    summarize_prompt = ChatPromptTemplate.from_template(
        "用一句话总结以下内容：\n{text}"
    )
    summarize_chain = summarize_prompt | llm | parser
    
    # 组合：翻译 → 总结
    # RunnablePassthrough 用于传递中间结果
    combined_chain = (
        {"text": translate_chain}  # 先翻译
        | summarize_chain          # 再总结
    )
    
    # 测试
    english_text = "AI Agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve specific goals."
    
    print(f"原文: {english_text}")
    print(f"\n处理流程: 翻译 → 总结")
    
    result = combined_chain.invoke({"text": english_text})
    print(f"\n最终结果: {result}")


# ============================================================
# 第六部分：流式输出（Streaming）
# ============================================================

def demo_6_streaming():
    """
    流式输出：实时显示生成内容
    
    适用场景：长文本生成、聊天界面
    """
    print("\n" + "=" * 60)
    print("Demo 6: 流式输出")
    print("=" * 60)
    
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    llm = ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
        streaming=True,  # 启用流式
    )
    
    prompt = ChatPromptTemplate.from_template(
        "用 3 个要点介绍 {topic}"
    )
    
    chain = prompt | llm | StrOutputParser()
    
    print(f"问题: 用 3 个要点介绍 LangChain")
    print(f"流式输出: ", end="", flush=True)
    
    # 流式调用
    for chunk in chain.stream({"topic": "LangChain"}):
        print(chunk, end="", flush=True)
    
    print("\n")


# ============================================================
# 运行所有 Demo
# ============================================================

if __name__ == "__main__":
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print(f"使用模型: {IFLOW_MODEL}")
    
    # 按顺序运行 Demo
    demo_1_basic_llm()
    demo_2_prompt_template()
    demo_3_lcel_chain()
    demo_4_output_parser()
    demo_5_chain_composition()
    demo_6_streaming()
    
    print("\n" + "=" * 60)
    print("LangChain 基础教程完成！")
    print("=" * 60)
    print("""
知识点回顾：

1. ChatOpenAI - LLM 封装
2. ChatPromptTemplate - 提示词模板
3. LCEL (|) - 链式语法，LCEL 的 | 确实是语法糖，底层是 __or__ 方法重载
4. OutputParser - 结构化输出
5. Chain 组合 - 多步骤处理
6. Streaming - 流式输出

下一步：学习 Tool 和 Agent
""")

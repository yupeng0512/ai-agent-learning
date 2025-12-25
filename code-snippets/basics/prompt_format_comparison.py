"""
Prompt 格式对比实验：XML vs Markdown vs 自然语言

运行：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/basics/prompt_format_comparison.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1"),
    api_key=os.getenv("IFLOW_API_KEY"),
)
MODEL = os.getenv("IFLOW_MODEL", "TBStars2-200B-A13B")


# ============================================================
# 测试用例：代码审查任务
# ============================================================

CODE_TO_REVIEW = '''
def login(user, pwd):
    if user == "admin":
        return True
    return False
'''

# ------------------------------------------------------------
# 格式1：纯自然语言
# ------------------------------------------------------------
PROMPT_NATURAL = f"""
你是一个代码审查专家。请审查下面的代码，找出问题并给出建议。
代码是 Python 写的，主要功能是用户登录。

{CODE_TO_REVIEW}

请列出问题和改进建议。
"""

# ------------------------------------------------------------
# 格式2：Markdown 结构
# ------------------------------------------------------------
PROMPT_MARKDOWN = f"""
## 角色
你是一个资深代码审查专家

## 任务
审查以下代码，找出安全问题和改进点

## 代码
```python
{CODE_TO_REVIEW}
```

## 输出格式
1. 问题列表（按严重程度排序）
2. 改进建议
3. 修复后的代码
"""

# ------------------------------------------------------------
# 格式3：XML 结构（Claude 推荐）
# ------------------------------------------------------------
PROMPT_XML = f"""
<role>资深代码审查专家，专注于安全和最佳实践</role>

<task>
  <objective>审查代码，找出安全问题和改进点</objective>
  <focus_areas>
    <area>安全漏洞</area>
    <area>代码质量</area>
    <area>最佳实践</area>
  </focus_areas>
</task>

<code language="python">
{CODE_TO_REVIEW}
</code>

<output_format>
  <section name="问题列表">按严重程度排序，标注 [严重/中等/轻微]</section>
  <section name="改进建议">具体可执行的建议</section>
  <section name="修复代码">完整的修复后代码</section>
</output_format>

<constraints>
  <constraint>不要遗漏任何安全问题</constraint>
  <constraint>建议必须具体可执行</constraint>
</constraints>
"""


def call_llm(prompt: str, label: str) -> str:
    """调用 LLM 并返回结果"""
    print(f"\n{'='*60}")
    print(f"测试: {label}")
    print(f"{'='*60}")
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # 降低随机性，便于对比
    )
    
    result = response.choices[0].message.content
    
    # 统计 token 使用
    usage = response.usage
    print(f"输入 tokens: {usage.prompt_tokens}")
    print(f"输出 tokens: {usage.completion_tokens}")
    print(f"\n回答:\n{result[:1000]}...")  # 只显示前1000字符
    
    return result


def main():
    if not os.getenv("IFLOW_API_KEY"):
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        return
    
    print(f"使用模型: {MODEL}")
    
    # 运行三种格式的测试
    results = {}
    
    results['natural'] = call_llm(PROMPT_NATURAL, "自然语言格式")
    results['markdown'] = call_llm(PROMPT_MARKDOWN, "Markdown 格式")
    results['xml'] = call_llm(PROMPT_XML, "XML 格式")
    
    # 总结对比
    print("\n" + "="*60)
    print("格式对比总结")
    print("="*60)
    print("""
┌─────────────┬─────────────────────────────────────────────┐
│ 格式        │ 特点                                        │
├─────────────┼─────────────────────────────────────────────┤
│ 自然语言    │ 简单直接，但结构不明确，输出格式不稳定      │
│ Markdown    │ 人类可读性好，但边界不够清晰                │
│ XML         │ 结构最清晰，边界明确，输出更稳定可控        │
└─────────────┴─────────────────────────────────────────────┘

使用建议：
1. 简单任务 → 自然语言即可
2. 需要格式化输出 → Markdown
3. 复杂任务/多段输入/需要稳定输出 → XML
4. 混合使用：XML 做结构，内容用 Markdown
""")


if __name__ == "__main__":
    main()


# ============================================================
# 知识点总结
# ============================================================
"""
## 为什么 XML 对 Claude 效果更好？

1. **训练数据**：Claude 训练时大量使用 XML 标签结构
2. **边界清晰**：<tag>...</tag> 比 ## 标题更明确
3. **嵌套支持**：XML 天然支持复杂嵌套结构
4. **解析友好**：模型更容易理解输入的不同部分

## XML Prompt 最佳实践

1. 常用标签：
   - <role> 角色定义
   - <task> 任务描述
   - <context> 背景信息
   - <input> 用户输入
   - <output_format> 输出格式要求
   - <constraints> 约束条件
   - <examples> 示例

2. 标签命名：
   - 使用语义化名称
   - 保持一致性
   - 避免过度嵌套（3层以内）

3. 混合使用：
   ```xml
   <task>
     ## 目标
     完成代码审查
     
     ## 重点
     - 安全性
     - 性能
   </task>
   ```

## 面试考点

Q: 为什么要用 XML 格式写 Prompt？
A: 
1. 边界清晰，减少歧义
2. 结构化便于模型理解
3. 输出更稳定可控
4. Claude 等模型对 XML 有特殊优化
"""

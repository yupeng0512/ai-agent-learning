"""
代码 Review Agent - 组合架构示例
架构：Plan-and-Execute + Reflexion

运行前：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/code_review_agent.py
"""

import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import List, Dict

# 加载环境变量
load_dotenv()

IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "TBStars2-200B-A13B")


def get_llm():
    return ChatOpenAI(
        model=IFLOW_MODEL,
        openai_api_key=IFLOW_API_KEY,
        openai_api_base=IFLOW_BASE_URL,
    )


# ============================================================
# 模拟工具：实际项目中会调用真实的代码分析工具
# ============================================================

@tool
def read_code_changes(pr_id: str) -> str:
    """读取 PR 中的代码变更"""
    # 模拟 PR 代码变更
    return """
文件: user_service.py
变更类型: 修改

+ def get_user(user_id):
+     query = f"SELECT * FROM users WHERE id = {user_id}"
+     result = db.execute(query)
+     return result
+
+ def update_password(user_id, new_password):
+     db.execute(f"UPDATE users SET password = '{new_password}' WHERE id = {user_id}")
+     print(f"Password updated for user {user_id}: {new_password}")
"""


@tool
def check_code_style(code: str) -> str:
    """检查代码风格问题"""
    issues = []
    if "f\"SELECT" in code or "f'SELECT" in code:
        issues.append("⚠️ 风格问题: 使用 f-string 构建 SQL 不推荐，应使用参数化查询")
    if "print(" in code:
        issues.append("⚠️ 风格问题: 生产代码不应使用 print，应使用 logging")
    if not issues:
        issues.append("✅ 代码风格检查通过")
    return "\n".join(issues)


@tool
def check_potential_bugs(code: str) -> str:
    """检查潜在 bug"""
    issues = []
    if "db.execute" in code and "try" not in code:
        issues.append("🐛 潜在 Bug: 数据库操作没有异常处理")
    if "return result" in code and "if result" not in code:
        issues.append("🐛 潜在 Bug: 没有检查查询结果是否为空")
    if not issues:
        issues.append("✅ 未发现明显 bug")
    return "\n".join(issues)


@tool
def check_security_issues(code: str) -> str:
    """检查安全漏洞"""
    issues = []
    if "f\"SELECT" in code or "f'SELECT" in code:
        issues.append("🔴 严重安全问题: SQL 注入漏洞！使用 f-string 拼接 SQL 极其危险")
    if "password" in code.lower() and "print" in code:
        issues.append("🔴 严重安全问题: 日志中打印了密码明文！")
    if "password" in code.lower() and "hash" not in code.lower():
        issues.append("🟡 安全建议: 密码应该加密存储，未见 hash 处理")
    if not issues:
        issues.append("✅ 未发现安全问题")
    return "\n".join(issues)


# ============================================================
# 组合架构：Plan-and-Execute + Reflexion
# ============================================================

class CodeReviewAgent:
    """
    代码 Review Agent
    
    架构组合：
    1. Plan-and-Execute: 规划检查步骤，逐个执行
    2. Reflexion: 检查完成后反思，确保没有遗漏
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.tools = {
            "read_code_changes": read_code_changes,
            "check_code_style": check_code_style,
            "check_potential_bugs": check_potential_bugs,
            "check_security_issues": check_security_issues,
        }
        self.execution_results = []
    
    def plan(self, task: str) -> List[Dict]:
        """
        第一阶段：制定 Review 计划
        """
        planner_prompt = ChatPromptTemplate.from_template("""
你是一个代码审查专家。根据任务要求，制定详细的审查计划。

任务：{task}

可用工具：
- read_code_changes: 读取代码变更
- check_code_style: 检查代码风格
- check_potential_bugs: 检查潜在 bug
- check_security_issues: 检查安全问题

请输出 JSON 格式的计划：
{{"steps": [
    {{"step": 1, "tool": "工具名", "input": "参数", "purpose": "目的"}},
    ...
]}}

只输出 JSON，不要其他内容。
""")
        
        chain = planner_prompt | self.llm | StrOutputParser()
        plan_text = chain.invoke({"task": task})
        
        # 解析 JSON
        try:
            json_match = re.search(r'\{.*\}', plan_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group()).get("steps", [])
        except:
            pass
        
        # 默认计划
        return [
            {"step": 1, "tool": "read_code_changes", "input": "PR-123", "purpose": "读取代码变更"},
            {"step": 2, "tool": "check_code_style", "input": "code", "purpose": "检查代码风格"},
            {"step": 3, "tool": "check_potential_bugs", "input": "code", "purpose": "检查潜在bug"},
            {"step": 4, "tool": "check_security_issues", "input": "code", "purpose": "检查安全问题"},
        ]
    
    def execute(self, plan: List[Dict]) -> List[Dict]:
        """
        第二阶段：按计划执行检查
        """
        results = []
        code_content = ""
        
        for step in plan:
            tool_name = step.get("tool", "")
            purpose = step.get("purpose", "")
            
            print(f"\n  📋 步骤 {step.get('step')}: {purpose}")
            
            if tool_name in self.tools:
                # 特殊处理：代码检查工具需要用读取到的代码
                if tool_name == "read_code_changes":
                    result = self.tools[tool_name].invoke(step.get("input", ""))
                    code_content = result
                else:
                    result = self.tools[tool_name].invoke(code_content)
                
                print(f"     → 结果: {result[:100]}...")
                results.append({
                    "step": step.get("step"),
                    "tool": tool_name,
                    "purpose": purpose,
                    "result": result
                })
        
        self.execution_results = results
        return results
    
    def synthesize(self, results: List[Dict]) -> str:
        """
        第三阶段：整合检查结果
        """
        synthesize_prompt = ChatPromptTemplate.from_template("""
根据以下代码审查结果，生成一份结构化的 Review 意见：

{results}

要求：
1. 按严重程度分类（严重/警告/建议）
2. 每个问题给出具体修复建议
3. 最后给出是否可以合并的结论

用中文输出。
""")
        
        chain = synthesize_prompt | self.llm | StrOutputParser()
        results_text = "\n".join([
            f"【{r['purpose']}】\n{r['result']}" for r in results
        ])
        
        return chain.invoke({"results": results_text})
    
    def reflect(self, review_result: str) -> str:
        """
        第四阶段：反思检查（Reflexion）
        确保没有遗漏重要问题
        """
        reflect_prompt = ChatPromptTemplate.from_template("""
你是一个资深代码审查专家。请检查以下 Review 意见是否完整：

当前 Review 意见：
{review}

原始检查结果：
{raw_results}

请检查：
1. 是否有重要问题被遗漏？
2. 修复建议是否具体可行？
3. 严重程度判断是否准确？
4. 是否需要补充其他检查维度？

如果发现遗漏或需要补充，请直接输出补充内容。
如果 Review 已经完整，输出"Review 意见完整，无需补充"。
""")
        
        chain = reflect_prompt | self.llm | StrOutputParser()
        raw_results = "\n".join([r['result'] for r in self.execution_results])
        
        return chain.invoke({
            "review": review_result,
            "raw_results": raw_results
        })
    
    def review(self, pr_id: str) -> str:
        """
        完整的 Review 流程
        """
        task = f"对 PR {pr_id} 进行全面的代码审查"
        
        print("=" * 60)
        print(f"🔍 开始审查: {pr_id}")
        print("=" * 60)
        
        # 阶段 1: 规划
        print("\n【阶段 1: 制定审查计划】")
        plan = self.plan(task)
        print(f"  计划步骤数: {len(plan)}")
        for p in plan:
            print(f"    {p.get('step')}. {p.get('purpose')}")
        
        # 阶段 2: 执行
        print("\n【阶段 2: 执行检查】")
        results = self.execute(plan)
        
        # 阶段 3: 整合
        print("\n【阶段 3: 整合结果】")
        review_result = self.synthesize(results)
        print(f"\n初步 Review 意见:\n{review_result}")
        
        # 阶段 4: 反思
        print("\n【阶段 4: 反思检查 (Reflexion)】")
        reflection = self.reflect(review_result)
        print(f"\n反思结果:\n{reflection}")
        
        # 最终输出
        print("\n" + "=" * 60)
        print("📝 最终 Review 报告")
        print("=" * 60)
        
        if "无需补充" in reflection or "完整" in reflection:
            final_report = review_result
        else:
            final_report = f"{review_result}\n\n【补充意见】\n{reflection}"
        
        print(final_report)
        return final_report


# ============================================================
# 运行示例
# ============================================================

if __name__ == "__main__":
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print(f"使用模型: {IFLOW_MODEL}")
    
    agent = CodeReviewAgent()
    result = agent.review("PR-123")


# ============================================================
# 架构设计要点总结
# ============================================================
"""
这个 Code Review Agent 展示了组合架构的设计思路：

┌─────────────────────────────────────────────────────────────┐
│                    Code Review Agent                        │
├─────────────────────────────────────────────────────────────┤
│  Plan-and-Execute 部分：                                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ Planner │ →  │Executor │ →  │Synthesizer│               │
│  │ 制定计划 │    │ 执行检查 │    │ 整合结果  │               │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       ↓              ↓              ↓                       │
│   检查清单      逐项执行       初步报告                      │
├─────────────────────────────────────────────────────────────┤
│  Reflexion 部分：                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │初步报告  │ →  │Reflector│ →  │最终报告  │                │
│  │         │    │ 反思检查 │    │         │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
│                      ↓                                      │
│              "是否遗漏？是否准确？"                          │
└─────────────────────────────────────────────────────────────┘

为什么这样设计？

1. Plan-and-Execute 保证全面性
   - 不会忘记检查某个维度
   - 执行过程可追踪
   - 计划可以根据代码特点调整

2. Reflexion 保证质量
   - 自我检查遗漏
   - 验证建议的可行性
   - 提高 Review 的专业度

3. 组合的优势
   - 单独用 Plan-Execute：可能输出不够完善
   - 单独用 Reflexion：没有结构化的检查流程
   - 组合使用：既全面又精准

面试考点：
Q: 为什么不直接用 ReAct？
A: ReAct 缺乏全局规划，可能漏检。Code Review 需要系统性检查。

Q: Reflexion 阶段的价值是什么？
A: 1) 发现遗漏 2) 验证建议质量 3) 提高输出专业度
   相当于"资深工程师的二次审核"

Q: 这个架构的成本如何？
A: 比单纯 ReAct 高（多了规划和反思的 LLM 调用）
   但对于代码审查这种质量敏感的任务，值得投入
"""

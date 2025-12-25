"""
分布式 Code Review Agent - 处理大型 PR
架构：Map-Reduce + 分层 Reflexion

解决问题：
1. Token 限制 → 分片处理，独立上下文
2. 执行时间 → 并行处理
3. 检查质量 → 分层反思

运行前：
cd ai-agent-learning
source .venv/bin/activate
python code-snippets/langchain/distributed_review_agent.py
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from dataclasses import dataclass

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
# 数据结构
# ============================================================

@dataclass
class FileChange:
    """单个文件变更"""
    path: str
    change_type: str  # added, modified, deleted
    content: str
    
@dataclass
class FileReviewResult:
    """单个文件的 Review 结果"""
    path: str
    issues: List[str]
    severity: str  # critical, warning, info
    suggestions: List[str]


# ============================================================
# 模拟数据：100+ 文件的大型 PR
# ============================================================

def mock_large_pr() -> List[FileChange]:
    """模拟一个大型 PR 的文件变更"""
    files = []
    
    # 模拟不同类型的文件
    templates = [
        ("user_service.py", "modified", """
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
"""),
        ("auth_handler.py", "modified", """
def login(username, password):
    print(f"Login attempt: {username}:{password}")
    return check_credentials(username, password)
"""),
        ("config.py", "added", """
DATABASE_URL = "postgresql://admin:password123@localhost/db"
SECRET_KEY = "hardcoded-secret-key-12345"
"""),
        ("utils.py", "modified", """
def process_data(data):
    result = eval(data)  # 危险！
    return result
"""),
        ("api_routes.py", "modified", """
@app.route('/admin')
def admin_panel():
    # 没有权限检查
    return render_template('admin.html')
"""),
    ]
    
    # 生成 25 个文件（模拟大 PR，实际可以更多）
    for i in range(25):
        template = templates[i % len(templates)]
        files.append(FileChange(
            path=f"src/module_{i}/{template[0]}",
            change_type=template[1],
            content=template[2]
        ))
    
    return files


# ============================================================
# 子 Agent：处理单个文件（独立上下文）
# ============================================================

class FileReviewWorker:
    """
    文件级别的 Review Worker（子 Agent）
    
    特点：
    1. 独立上下文 - 只加载当前文件
    2. 轻量级 - 处理完即释放
    3. 包含局部 Reflexion
    """
    
    def __init__(self):
        self.llm = get_llm()
    
    def review_file(self, file: FileChange) -> FileReviewResult:
        """
        Review 单个文件
        包含：检查 + 局部反思
        """
        # 第一步：执行检查
        review_prompt = ChatPromptTemplate.from_template("""
你是代码审查专家。请审查以下代码变更：

文件路径: {path}
变更类型: {change_type}
代码内容:
```
{content}
```

请检查：
1. 安全问题（SQL注入、XSS、敏感信息泄露等）
2. 代码质量（异常处理、日志规范等）
3. 潜在 Bug

输出 JSON 格式：
{{"issues": ["问题1", "问题2"], "severity": "critical/warning/info", "suggestions": ["建议1"]}}

只输出 JSON。
""")
        
        chain = review_prompt | self.llm | StrOutputParser()
        result_text = chain.invoke({
            "path": file.path,
            "change_type": file.change_type,
            "content": file.content
        })
        
        # 解析结果
        try:
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {"issues": [], "severity": "info", "suggestions": []}
        except:
            data = {"issues": ["解析失败"], "severity": "warning", "suggestions": []}
        
        # 第二步：局部反思（快速检查）
        if data.get("issues"):
            reflection = self._local_reflect(file, data)
            if reflection:
                data["issues"].extend(reflection)
        
        return FileReviewResult(
            path=file.path,
            issues=data.get("issues", []),
            severity=data.get("severity", "info"),
            suggestions=data.get("suggestions", [])
        )
    
    def _local_reflect(self, file: FileChange, initial_result: dict) -> List[str]:
        """
        局部反思：快速检查是否遗漏明显问题
        （轻量级，不是完整的 Reflexion）
        """
        # 简单的规则检查作为补充
        additional_issues = []
        content = file.content.lower()
        
        # 检查初步结果是否遗漏了明显问题
        if "eval(" in content and "eval" not in str(initial_result):
            additional_issues.append("🔴 遗漏: 发现 eval() 调用，存在代码注入风险")
        
        if "password" in content and "hardcoded" not in str(initial_result).lower():
            if "password123" in content or "secret" in content:
                additional_issues.append("🔴 遗漏: 发现硬编码的密码/密钥")
        
        return additional_issues


# ============================================================
# 主 Agent：协调和汇总（Map-Reduce 模式）
# ============================================================

class DistributedReviewAgent:
    """
    分布式 Code Review Agent（主 Agent）
    
    架构：
    1. Planner: 分析 PR，制定分片策略
    2. Dispatcher: 分发任务给 Worker
    3. Aggregator: 汇总结果
    4. Global Reflector: 全局反思
    """
    
    def __init__(self, max_workers: int = 5):
        self.llm = get_llm()
        self.max_workers = max_workers
        self.worker = FileReviewWorker()
    
    def review_pr(self, files: List[FileChange]) -> str:
        """
        完整的分布式 Review 流程
        """
        print("=" * 70)
        print(f"🔍 分布式 Code Review - 共 {len(files)} 个文件")
        print("=" * 70)
        
        # 阶段 1: 规划分片
        print("\n【阶段 1: 规划分片策略】")
        batches = self._plan_batches(files)
        print(f"  分成 {len(batches)} 个批次处理")
        
        # 阶段 2: 并行执行（Map）
        print("\n【阶段 2: 并行执行 Review】")
        all_results = self._parallel_review(batches)
        print(f"  完成 {len(all_results)} 个文件的审查")
        
        # 阶段 3: 汇总结果（Reduce）
        print("\n【阶段 3: 汇总结果】")
        summary = self._aggregate_results(all_results)
        print(f"\n初步汇总:\n{summary[:500]}...")
        
        # 阶段 4: 全局反思
        print("\n【阶段 4: 全局反思 (Global Reflexion)】")
        final_report = self._global_reflect(all_results, summary)
        
        print("\n" + "=" * 70)
        print("📝 最终 Review 报告")
        print("=" * 70)
        print(final_report)
        
        return final_report
    
    def _plan_batches(self, files: List[FileChange]) -> List[List[FileChange]]:
        """
        规划分片策略
        
        策略：
        1. 按文件数量分批
        2. 可以按文件类型、目录等更智能地分组
        """
        batch_size = max(1, len(files) // self.max_workers)
        batches = []
        
        for i in range(0, len(files), batch_size):
            batches.append(files[i:i + batch_size])
        
        for i, batch in enumerate(batches):
            print(f"  批次 {i+1}: {len(batch)} 个文件")
        
        return batches
    
    def _parallel_review(self, batches: List[List[FileChange]]) -> List[FileReviewResult]:
        """
        并行执行 Review（Map 阶段）
        
        使用线程池并行处理多个批次
        """
        all_results = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = []
            for batch_idx, batch in enumerate(batches):
                for file in batch:
                    future = executor.submit(self._review_single_file, file, batch_idx)
                    futures.append(future)
            
            # 收集结果
            for future in futures:
                result = future.result()
                if result:
                    all_results.append(result)
        
        return all_results
    
    def _review_single_file(self, file: FileChange, batch_idx: int) -> FileReviewResult:
        """
        Review 单个文件（Worker 调用）
        """
        print(f"    [批次{batch_idx+1}] 审查: {file.path}")
        return self.worker.review_file(file)
    
    def _aggregate_results(self, results: List[FileReviewResult]) -> str:
        """
        汇总所有 Review 结果（Reduce 阶段）
        """
        # 按严重程度分类
        critical = [r for r in results if r.severity == "critical"]
        warning = [r for r in results if r.severity == "warning"]
        info = [r for r in results if r.severity == "info"]
        
        # 构建汇总
        aggregate_prompt = ChatPromptTemplate.from_template("""
请汇总以下代码审查结果，生成一份结构化报告：

严重问题 ({critical_count} 个文件):
{critical_issues}

警告 ({warning_count} 个文件):
{warning_issues}

建议 ({info_count} 个文件):
{info_issues}

要求：
1. 按问题类型归类（安全、代码质量、Bug）
2. 列出受影响的文件
3. 给出优先级排序
4. 提供修复建议

用中文输出，控制在500字以内。
""")
        
        def format_results(results_list):
            if not results_list:
                return "无"
            return "\n".join([
                f"- {r.path}: {', '.join(r.issues[:2])}" 
                for r in results_list[:10]  # 限制数量避免 token 爆炸
            ])
        
        chain = aggregate_prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "critical_count": len(critical),
            "critical_issues": format_results(critical),
            "warning_count": len(warning),
            "warning_issues": format_results(warning),
            "info_count": len(info),
            "info_issues": format_results(info),
        })
    
    def _global_reflect(self, results: List[FileReviewResult], summary: str) -> str:
        """
        全局反思（Global Reflexion）
        
        检查：
        1. 是否有跨文件的问题被遗漏
        2. 问题分类是否准确
        3. 是否需要补充系统性建议
        """
        reflect_prompt = ChatPromptTemplate.from_template("""
你是资深安全架构师。请对以下 Code Review 报告进行全局审查：

当前报告：
{summary}

审查统计：
- 总文件数: {total_files}
- 严重问题文件: {critical_count}
- 警告文件: {warning_count}

请检查：
1. 是否有跨文件/系统性的安全问题被遗漏？
2. 问题的严重程度判断是否准确？
3. 是否需要补充架构级别的建议？
4. 这个 PR 是否可以合并？给出明确结论。

如果发现遗漏，直接补充。最后给出合并建议。
""")
        
        critical_count = len([r for r in results if r.severity == "critical"])
        warning_count = len([r for r in results if r.severity == "warning"])
        
        chain = reflect_prompt | self.llm | StrOutputParser()
        
        reflection = chain.invoke({
            "summary": summary,
            "total_files": len(results),
            "critical_count": critical_count,
            "warning_count": warning_count,
        })
        
        # 组合最终报告
        final_report = f"""
{summary}

---
【全局审查意见】
{reflection}

---
【审查统计】
- 总文件数: {len(results)}
- 严重问题: {critical_count} 个文件
- 警告: {warning_count} 个文件
- 建议: {len(results) - critical_count - warning_count} 个文件
"""
        
        return final_report


# ============================================================
# 运行示例
# ============================================================

if __name__ == "__main__":
    if not IFLOW_API_KEY:
        print("错误: 请在 .env 文件中配置 IFLOW_API_KEY")
        exit(1)
    
    print(f"使用模型: {IFLOW_MODEL}")
    
    # 模拟大型 PR
    files = mock_large_pr()
    
    # 创建分布式 Agent
    agent = DistributedReviewAgent(max_workers=5)
    
    # 执行 Review
    result = agent.review_pr(files)


# ============================================================
# 架构设计总结
# ============================================================
"""
分布式 Code Review Agent 架构：

┌─────────────────────────────────────────────────────────────────┐
│                        主 Agent (Orchestrator)                  │
│  ┌─────────┐   ┌─────────────┐   ┌──────────┐   ┌───────────┐  │
│  │ Planner │ → │ Dispatcher  │ → │Aggregator│ → │ Reflector │  │
│  │ 分片策略 │   │ 任务分发    │   │ 结果汇总 │   │ 全局反思  │  │
│  └─────────┘   └─────────────┘   └──────────┘   └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Worker 1 │   │ Worker 2 │   │ Worker 3 │   (并行)
    │ 文件1-10 │   │ 文件11-20│   │ 文件21-30│
    │ +局部反思│   │ +局部反思│   │ +局部反思│
    └──────────┘   └──────────┘   └──────────┘


解决的问题：

1. Token 限制
   ✅ 每个 Worker 独立上下文，只加载当前文件
   ✅ 主 Agent 只处理汇总信息，不加载原始代码
   ✅ 类似你说的 Skill 按需加载思路

2. 执行时间
   ✅ 多 Worker 并行处理（ThreadPoolExecutor）
   ✅ 100 文件分 5 批，理论上提速 5 倍
   ✅ 可以根据机器资源调整 max_workers

3. 检查质量（分层 Reflexion）
   ✅ Worker 级别：局部反思（检查单文件是否遗漏）
   ✅ 主 Agent 级别：全局反思（检查跨文件问题）


面试考点：

Q: 为什么要分层反思？
A: 局部反思快速、针对性强；全局反思发现系统性问题。
   类似"开发自测 + QA 测试"的分层质量保证。

Q: 并行处理有什么注意事项？
A: 1) API 限流 - 需要控制并发数
   2) 错误处理 - 单个 Worker 失败不应影响整体
   3) 结果合并 - 需要处理冲突和去重

Q: 这个架构的成本如何优化？
A: 1) 增量 Review - 只处理变更文件
   2) 缓存 - 相同文件不重复检查
   3) 分级 - 简单文件用规则，复杂文件用 LLM

Q: 与 CrewAI/AutoGen 的多 Agent 有什么区别？
A: 这里是"同质 Agent 并行"（都是 Reviewer）
   CrewAI 是"异质 Agent 协作"（不同角色）
   可以结合：每个 Worker 内部用 CrewAI 的多角色
"""

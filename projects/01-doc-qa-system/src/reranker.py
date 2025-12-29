"""
Reranker 模块 - 优化检索效果

功能：
- 对初步检索结果进行重排序
- 使用交叉编码器或 LLM 进行精细化排序
- 提高检索相关性

原理：
1. 初步检索（向量搜索）：快速但粗糙
2. Rerank（重排序）：慢但精确

流程：
Query → 向量搜索(top-k=20) → Rerank → 返回(top-n=5)
"""

import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class RerankResult:
    """重排序结果"""
    document: Document
    score: float
    original_rank: int


class LLMReranker:
    """
    基于 LLM 的重排序器
    
    使用 LLM 对检索结果进行相关性评分
    """
    
    def __init__(self, llm):
        """
        初始化
        
        Args:
            llm: LangChain LLM 实例
        """
        self.llm = llm
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """
        对文档进行重排序
        
        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_n: 返回的文档数量
            
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []
        
        results = []
        
        for i, doc in enumerate(documents):
            # 使用 LLM 评估相关性
            score = self._score_relevance(query, doc.page_content)
            results.append(RerankResult(
                document=doc,
                score=score,
                original_rank=i,
            ))
        
        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_n]
    
    def _score_relevance(self, query: str, content: str) -> float:
        """
        评估文档与查询的相关性
        
        Args:
            query: 用户查询
            content: 文档内容
            
        Returns:
            相关性分数 (0-1)
        """
        prompt = f"""请评估以下文档内容与用户问题的相关性。

用户问题：{query}

文档内容：{content[:500]}

请只输出一个 0 到 1 之间的数字，表示相关性分数：
- 0: 完全不相关
- 0.5: 部分相关
- 1: 高度相关

分数："""
        
        try:
            response = self.llm.invoke(prompt)
            score_str = response.content.strip()
            # 提取数字
            import re
            match = re.search(r'([0-9]*\.?[0-9]+)', score_str)
            if match:
                score = float(match.group(1))
                return min(max(score, 0), 1)  # 限制在 0-1 之间
            return 0.5
        except Exception:
            return 0.5


class SimpleReranker:
    """
    简单重排序器
    
    基于关键词匹配和位置信息进行重排序
    不需要额外 API 调用，速度快
    """
    
    def __init__(self):
        pass
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """
        对文档进行重排序
        
        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_n: 返回的文档数量
            
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []
        
        results = []
        query_terms = set(self._tokenize(query))
        
        for i, doc in enumerate(documents):
            score = self._calculate_score(query_terms, doc.page_content, i)
            results.append(RerankResult(
                document=doc,
                score=score,
                original_rank=i,
            ))
        
        # 按分数降序排序
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_n]
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        # 中文按字符，英文按单词
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        english = re.findall(r'[a-zA-Z]+', text.lower())
        
        tokens = []
        for word in chinese:
            tokens.extend(list(word))  # 中文按字
        tokens.extend(english)
        
        return tokens
    
    def _calculate_score(
        self,
        query_terms: set,
        content: str,
        original_rank: int,
    ) -> float:
        """
        计算相关性分数
        
        考虑因素：
        1. 关键词匹配度
        2. 原始排名（向量搜索结果）
        3. 关键词位置
        """
        content_terms = set(self._tokenize(content))
        
        # 1. 关键词匹配度 (0-0.6)
        if not query_terms:
            keyword_score = 0
        else:
            overlap = len(query_terms & content_terms)
            keyword_score = (overlap / len(query_terms)) * 0.6
        
        # 2. 原始排名加权 (0-0.3)
        # 原始排名越靠前，分数越高
        rank_score = max(0, 0.3 - original_rank * 0.03)
        
        # 3. 位置加权 (0-0.1)
        # 如果关键词出现在开头，加分
        position_score = 0
        for term in query_terms:
            if term in content[:100]:
                position_score = 0.1
                break
        
        return keyword_score + rank_score + position_score


class HybridReranker:
    """
    混合重排序器
    
    结合简单重排序和 LLM 重排序的优点：
    1. 先用简单方法快速筛选
    2. 对 top 候选使用 LLM 精排
    """
    
    def __init__(self, llm=None, use_llm_for_top: int = 3):
        """
        初始化
        
        Args:
            llm: LangChain LLM 实例（可选）
            use_llm_for_top: 对前 N 个候选使用 LLM 精排
        """
        self.simple_reranker = SimpleReranker()
        self.llm_reranker = LLMReranker(llm) if llm else None
        self.use_llm_for_top = use_llm_for_top
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[RerankResult]:
        """
        混合重排序
        
        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_n: 返回的文档数量
            
        Returns:
            重排序后的结果列表
        """
        if not documents:
            return []
        
        # 第一阶段：简单重排序
        simple_results = self.simple_reranker.rerank(
            query,
            documents,
            top_n=max(top_n, self.use_llm_for_top * 2),
        )
        
        # 如果没有 LLM 或文档太少，直接返回简单结果
        if not self.llm_reranker or len(simple_results) <= self.use_llm_for_top:
            return simple_results[:top_n]
        
        # 第二阶段：对 top 候选使用 LLM 精排
        top_candidates = [r.document for r in simple_results[:self.use_llm_for_top * 2]]
        llm_results = self.llm_reranker.rerank(query, top_candidates, top_n=top_n)
        
        return llm_results


def create_reranker(
    reranker_type: str = "simple",
    llm=None,
) -> SimpleReranker | LLMReranker | HybridReranker:
    """
    创建重排序器
    
    Args:
        reranker_type: 重排序器类型 ("simple", "llm", "hybrid")
        llm: LangChain LLM 实例（llm/hybrid 类型需要）
        
    Returns:
        重排序器实例
    """
    if reranker_type == "simple":
        return SimpleReranker()
    elif reranker_type == "llm":
        if not llm:
            raise ValueError("LLM reranker requires an LLM instance")
        return LLMReranker(llm)
    elif reranker_type == "hybrid":
        return HybridReranker(llm)
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reranker 测试")
    print("=" * 60)
    
    # 测试文档
    docs = [
        Document(page_content="AI Agent 是能够自主感知、决策、行动的智能系统", metadata={"source": "doc1"}),
        Document(page_content="Python 是一种流行的编程语言", metadata={"source": "doc2"}),
        Document(page_content="RAG 是检索增强生成，通过检索外部知识增强 LLM", metadata={"source": "doc3"}),
        Document(page_content="LangChain 是用于构建 AI Agent 的框架", metadata={"source": "doc4"}),
        Document(page_content="机器学习是人工智能的一个分支", metadata={"source": "doc5"}),
    ]
    
    query = "什么是 AI Agent"
    
    # 测试简单重排序
    print("\n--- 简单重排序 ---")
    simple = SimpleReranker()
    results = simple.rerank(query, docs, top_n=3)
    
    for r in results:
        print(f"  [{r.original_rank}→{results.index(r)}] 分数: {r.score:.3f} | {r.document.page_content[:30]}...")
    
    print("\n✅ 测试完成")

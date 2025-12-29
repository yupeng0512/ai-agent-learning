"""
向量存储模块

功能：
- 创建向量数据库
- 添加文档
- 相似度搜索
- 持久化存储
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore:
    """向量存储管理器"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: Optional[str] = None,
    ):
        """
        初始化向量存储
        
        Args:
            embeddings: Embedding 模型
            persist_directory: 持久化目录（可选）
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
        
        # 如果有持久化目录且存在，尝试加载
        if persist_directory and Path(persist_directory).exists():
            self._load_from_disk()
    
    def create_from_documents(self, documents: List[Document]) -> None:
        """
        从文档创建向量数据库
        
        Args:
            documents: 文档列表
        """
        from langchain_community.vectorstores import FAISS
        
        if not documents:
            raise ValueError("文档列表不能为空")
        
        print(f"正在创建向量数据库，共 {len(documents)} 个文档块...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("✅ 向量数据库创建完成")
        
        # 如果设置了持久化目录，保存到磁盘
        if self.persist_directory:
            self._save_to_disk()
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到现有向量数据库
        
        Args:
            documents: 要添加的文档列表
        """
        if self.vectorstore is None:
            self.create_from_documents(documents)
        else:
            self.vectorstore.add_documents(documents)
            print(f"✅ 已添加 {len(documents)} 个文档块")
            
            if self.persist_directory:
                self._save_to_disk()
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            score_threshold: 分数阈值（可选）
            
        Returns:
            相关文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化，请先添加文档")
        
        if score_threshold:
            # 带分数的搜索
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            # 过滤低于阈值的结果（注意：FAISS 分数越低越相似）
            filtered = [(doc, score) for doc, score in results if score <= score_threshold]
            return [doc for doc, _ in filtered]
        else:
            return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4,
    ) -> List[tuple]:
        """
        带分数的相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            (文档, 分数) 元组列表
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化，请先添加文档")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        获取检索器
        
        Args:
            search_kwargs: 搜索参数
            
        Returns:
            LangChain Retriever
        """
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化，请先添加文档")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def _save_to_disk(self) -> None:
        """保存到磁盘"""
        if self.vectorstore and self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self.vectorstore.save_local(self.persist_directory)
            print(f"✅ 向量数据库已保存到: {self.persist_directory}")
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        from langchain_community.vectorstores import FAISS
        
        if self.persist_directory and Path(self.persist_directory).exists():
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ 已从磁盘加载向量数据库: {self.persist_directory}")
            except Exception as e:
                print(f"⚠️ 加载向量数据库失败: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """检查向量数据库是否已初始化"""
        return self.vectorstore is not None
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        if not self.is_initialized:
            return {"status": "未初始化", "document_count": 0}
        
        # FAISS 没有直接获取文档数量的方法，这里用 index 的大小
        try:
            count = self.vectorstore.index.ntotal
        except:
            count = "未知"
        
        return {
            "status": "已初始化",
            "document_count": count,
            "persist_directory": self.persist_directory,
        }


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv(dotenv_path="../../.env")
    
    # 获取 API 配置
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    
    if not SILICONFLOW_API_KEY:
        print("❌ 请配置 SILICONFLOW_API_KEY")
        exit(1)
    
    # 创建 Embedding 模型
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=SILICONFLOW_API_KEY,
        openai_api_base=SILICONFLOW_BASE_URL,
    )
    
    # 创建向量存储
    store = VectorStore(embeddings)
    
    # 测试文档
    test_docs = [
        Document(page_content="AI Agent 是能够自主感知、决策、行动的智能系统", metadata={"source": "test1"}),
        Document(page_content="RAG 是检索增强生成，通过检索外部知识增强 LLM", metadata={"source": "test2"}),
        Document(page_content="LangChain 是 LLM 应用开发框架", metadata={"source": "test3"}),
    ]
    
    # 创建向量数据库
    store.create_from_documents(test_docs)
    
    # 测试搜索
    print("\n--- 测试搜索 ---")
    query = "什么是 AI Agent"
    results = store.similarity_search(query, k=2)
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content}")
    
    # 打印统计信息
    print(f"\n统计: {store.get_stats()}")

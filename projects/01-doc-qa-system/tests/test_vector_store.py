"""
VectorStore 单元测试
"""

import sys
import os
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.documents import Document


class TestVectorStore:
    """VectorStore 测试类（使用 Mock）"""
    
    @pytest.fixture
    def mock_embeddings(self):
        """创建 Mock Embeddings"""
        embeddings = Mock()
        # 模拟 embed_documents 返回向量
        embeddings.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])
        embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
        return embeddings
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        return [
            Document(page_content="AI Agent 是能够自主感知、决策、行动的智能系统", metadata={"source": "doc1"}),
            Document(page_content="RAG 是检索增强生成，通过检索外部知识增强 LLM", metadata={"source": "doc2"}),
            Document(page_content="LangChain 是 LLM 应用开发框架", metadata={"source": "doc3"}),
        ]
    
    def test_vector_store_init(self, mock_embeddings):
        """测试初始化"""
        from vector_store import VectorStore
        
        store = VectorStore(mock_embeddings)
        
        assert store.embeddings == mock_embeddings
        assert store.vectorstore is None
    
    def test_create_from_documents(self, mock_embeddings, sample_documents):
        """测试从文档创建向量库"""
        from vector_store import VectorStore
        
        # 需要真实的 embeddings 来测试 FAISS
        # 这里使用 Mock 测试接口
        store = VectorStore(mock_embeddings)
        
        # 由于 FAISS 需要真实向量，这里测试接口是否正确
        assert hasattr(store, 'create_from_documents')
        assert hasattr(store, 'add_documents')
        assert hasattr(store, 'similarity_search')
    
    def test_get_stats_empty(self, mock_embeddings):
        """测试空状态统计"""
        from vector_store import VectorStore
        
        store = VectorStore(mock_embeddings)
        stats = store.get_stats()
        
        assert stats["status"] == "未初始化"
        assert stats["document_count"] == 0
    
    def test_as_retriever_without_vectorstore(self, mock_embeddings):
        """测试未初始化时获取 retriever"""
        from vector_store import VectorStore
        
        store = VectorStore(mock_embeddings)
        
        with pytest.raises(ValueError):
            store.as_retriever()


class TestVectorStoreIntegration:
    """VectorStore 集成测试（需要真实 API）"""
    
    @pytest.fixture
    def real_embeddings(self):
        """创建真实 Embeddings（需要 API Key）"""
        from dotenv import load_dotenv
        
        # 加载环境变量
        env_paths = [
            Path(__file__).parent.parent.parent.parent / ".env",
            Path(__file__).parent.parent / ".env",
        ]
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                break
        
        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        
        if not api_key:
            pytest.skip("SILICONFLOW_API_KEY not configured")
        
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_key=api_key,
            openai_api_base=base_url,
        )
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        return [
            Document(page_content="AI Agent 是能够自主感知、决策、行动的智能系统", metadata={"source": "doc1"}),
            Document(page_content="RAG 是检索增强生成，通过检索外部知识增强 LLM", metadata={"source": "doc2"}),
            Document(page_content="LangChain 是 LLM 应用开发框架", metadata={"source": "doc3"}),
        ]
    
    @pytest.mark.integration
    def test_full_workflow(self, real_embeddings, sample_documents):
        """测试完整工作流程"""
        from vector_store import VectorStore
        
        store = VectorStore(real_embeddings)
        
        # 创建向量库
        store.create_from_documents(sample_documents)
        
        # 检查状态
        stats = store.get_stats()
        assert stats["status"] == "已就绪"
        assert stats["document_count"] == 3
        
        # 搜索测试
        results = store.similarity_search("什么是 AI Agent", k=2)
        assert len(results) == 2
        assert any("AI Agent" in doc.page_content for doc in results)
    
    @pytest.mark.integration
    def test_add_documents(self, real_embeddings, sample_documents):
        """测试添加文档"""
        from vector_store import VectorStore
        
        store = VectorStore(real_embeddings)
        
        # 先添加部分文档
        store.add_documents(sample_documents[:2])
        assert store.get_stats()["document_count"] == 2
        
        # 再添加更多文档
        store.add_documents(sample_documents[2:])
        assert store.get_stats()["document_count"] == 3
    
    @pytest.mark.integration
    def test_similarity_search_with_score(self, real_embeddings, sample_documents):
        """测试带分数的相似度搜索"""
        from vector_store import VectorStore
        
        store = VectorStore(real_embeddings)
        store.create_from_documents(sample_documents)
        
        results = store.similarity_search_with_score("AI Agent 智能系统", k=2)
        
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(score, float)
            assert hasattr(doc, 'page_content')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])

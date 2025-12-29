"""
QAEngine 单元测试
"""

import sys
import os
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document


class TestQAEngine:
    """QAEngine 测试类"""
    
    @pytest.fixture
    def mock_llm(self):
        """创建 Mock LLM"""
        llm = Mock()
        # 模拟 invoke 方法
        llm.invoke = Mock(return_value=Mock(content="这是一个测试回答"))
        return llm
    
    @pytest.fixture
    def mock_retriever(self):
        """创建 Mock Retriever"""
        retriever = Mock()
        retriever.invoke = Mock(return_value=[
            Document(page_content="相关文档内容1", metadata={"source": "doc1"}),
            Document(page_content="相关文档内容2", metadata={"source": "doc2"}),
        ])
        return retriever
    
    def test_qa_engine_init(self, mock_llm, mock_retriever):
        """测试初始化"""
        from qa_engine import QAEngine
        
        engine = QAEngine(mock_llm, mock_retriever)
        
        assert engine.llm == mock_llm
        assert engine.retriever == mock_retriever
        assert engine.chat_history == []
    
    def test_ask_returns_qa_response(self, mock_llm, mock_retriever):
        """测试 ask 返回 QAResponse"""
        from qa_engine import QAEngine, QAResponse
        
        engine = QAEngine(mock_llm, mock_retriever)
        result = engine.ask("测试问题")
        
        assert isinstance(result, QAResponse)
        assert hasattr(result, 'answer')
        assert hasattr(result, 'sources')
        assert hasattr(result, 'query')
    
    def test_ask_includes_sources(self, mock_llm, mock_retriever):
        """测试回答包含来源"""
        from qa_engine import QAEngine
        
        engine = QAEngine(mock_llm, mock_retriever)
        result = engine.ask("测试问题")
        
        assert len(result.sources) == 2
        assert result.sources[0].metadata["source"] == "doc1"
    
    def test_chat_history_updated(self, mock_llm, mock_retriever):
        """测试对话历史更新"""
        from qa_engine import QAEngine
        
        engine = QAEngine(mock_llm, mock_retriever)
        
        # 第一次提问
        engine.ask("问题1")
        assert len(engine.chat_history) == 2  # 用户问题 + AI回答
        
        # 第二次提问
        engine.ask("问题2")
        assert len(engine.chat_history) == 4
    
    def test_clear_history(self, mock_llm, mock_retriever):
        """测试清空历史"""
        from qa_engine import QAEngine
        
        engine = QAEngine(mock_llm, mock_retriever)
        engine.ask("问题1")
        
        engine.clear_history()
        
        assert len(engine.chat_history) == 0
    
    def test_get_history(self, mock_llm, mock_retriever):
        """测试获取历史"""
        from qa_engine import QAEngine
        
        engine = QAEngine(mock_llm, mock_retriever)
        engine.ask("问题1")
        
        history = engine.get_history()
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestQAEngineIntegration:
    """QAEngine 集成测试（需要真实 API）"""
    
    @pytest.fixture
    def real_setup(self):
        """创建真实的 LLM 和 Retriever"""
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
        
        iflow_key = os.getenv("IFLOW_API_KEY")
        silicon_key = os.getenv("SILICONFLOW_API_KEY")
        
        if not iflow_key or not silicon_key:
            pytest.skip("API keys not configured")
        
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from vector_store import VectorStore
        
        # 创建 LLM
        llm = ChatOpenAI(
            model=os.getenv("IFLOW_MODEL", "qwen3-coder-plus"),
            openai_api_key=iflow_key,
            openai_api_base=os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1"),
            temperature=0,
        )
        
        # 创建 Embeddings
        embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            openai_api_key=silicon_key,
            openai_api_base=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        )
        
        # 创建向量库
        store = VectorStore(embeddings)
        docs = [
            Document(page_content="公司年假政策：入职满一年的员工享有10天年假，满三年15天，满五年20天。", metadata={"source": "policy.md"}),
            Document(page_content="请假流程：提前3天在OA系统提交申请，由直属领导审批。紧急情况可事后补假。", metadata={"source": "policy.md"}),
            Document(page_content="报销流程：填写报销单，附上发票原件，提交财务部门审核，审核通过后7个工作日内到账。", metadata={"source": "finance.md"}),
        ]
        store.create_from_documents(docs)
        
        return llm, store.as_retriever(search_kwargs={"k": 2})
    
    @pytest.mark.integration
    def test_real_qa(self, real_setup):
        """测试真实问答"""
        from qa_engine import QAEngine
        
        llm, retriever = real_setup
        engine = QAEngine(llm, retriever)
        
        result = engine.ask("公司年假有多少天？")
        
        assert "年假" in result.answer or "10" in result.answer or "天" in result.answer
        assert len(result.sources) > 0
    
    @pytest.mark.integration
    def test_multi_turn_conversation(self, real_setup):
        """测试多轮对话"""
        from qa_engine import QAEngine
        
        llm, retriever = real_setup
        engine = QAEngine(llm, retriever)
        
        # 第一轮
        result1 = engine.ask("如何请假？")
        assert "请假" in result1.answer or "OA" in result1.answer or "申请" in result1.answer
        
        # 第二轮（基于上下文）
        result2 = engine.ask("需要提前多久？")
        assert "3" in result2.answer or "天" in result2.answer


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])

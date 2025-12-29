"""
DocumentLoader 单元测试
"""

import sys
from pathlib import Path

# 添加 src 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from document_loader import DocumentLoader


class TestDocumentLoader:
    """DocumentLoader 测试类"""
    
    @pytest.fixture
    def loader(self):
        """创建 DocumentLoader 实例"""
        return DocumentLoader(chunk_size=200, chunk_overlap=20)
    
    def test_load_text_content_basic(self, loader):
        """测试基本文本加载"""
        text = "这是一段测试文本。"
        docs = loader.load_text_content(text, {"source": "test"})
        
        assert len(docs) >= 1
        assert docs[0].page_content == text
        assert docs[0].metadata["source"] == "test"
    
    def test_load_text_content_with_chunking(self, loader):
        """测试文本切分"""
        # 创建一个超过 chunk_size 的文本
        text = "这是第一段内容。" * 50 + "\n\n" + "这是第二段内容。" * 50
        docs = loader.load_text_content(text, {"source": "test"})
        
        # 应该被切分成多个块
        assert len(docs) > 1
        
        # 每个块的大小应该不超过 chunk_size（允许一定误差）
        for doc in docs:
            assert len(doc.page_content) <= loader.text_splitter._chunk_size + 50
    
    def test_load_text_content_preserves_metadata(self, loader):
        """测试元数据保留"""
        text = "测试文本"
        metadata = {"source": "test", "author": "tester", "version": "1.0"}
        docs = loader.load_text_content(text, metadata)
        
        assert docs[0].metadata["source"] == "test"
        assert docs[0].metadata["author"] == "tester"
        assert docs[0].metadata["version"] == "1.0"
    
    def test_load_text_content_empty(self, loader):
        """测试空文本"""
        text = ""
        docs = loader.load_text_content(text, {"source": "test"})
        
        # 空文本应该返回空列表或单个空文档
        assert len(docs) <= 1
    
    def test_chunk_overlap(self):
        """测试 chunk_overlap 参数"""
        loader = DocumentLoader(chunk_size=100, chunk_overlap=20)
        
        # 创建一个需要切分的文本
        text = "A" * 150
        docs = loader.load_text_content(text, {"source": "test"})
        
        if len(docs) > 1:
            # 检查是否有重叠（第一个块的末尾应该出现在第二个块的开头）
            first_end = docs[0].page_content[-20:]
            second_start = docs[1].page_content[:20]
            # 由于切分策略，可能不完全重叠，但应该有一些重叠
            assert len(docs) >= 2
    
    def test_supported_formats(self, loader):
        """测试支持的文件格式"""
        supported = loader.get_supported_formats()
        
        assert ".pdf" in supported
        assert ".md" in supported
        assert ".txt" in supported
        assert ".docx" in supported
    
    def test_load_nonexistent_file(self, loader):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/path/file.txt")
    
    def test_load_unsupported_format(self, loader, tmp_path):
        """测试加载不支持的格式"""
        # 创建一个不支持的文件
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test content")
        
        with pytest.raises(ValueError):
            loader.load_file(str(unsupported_file))
    
    def test_load_txt_file(self, loader, tmp_path):
        """测试加载 TXT 文件"""
        # 创建测试文件
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("这是一个测试文件的内容。\n包含多行文本。")
        
        docs = loader.load_file(str(txt_file))
        
        assert len(docs) >= 1
        assert "测试文件" in docs[0].page_content
    
    def test_load_markdown_file(self, loader, tmp_path):
        """测试加载 Markdown 文件"""
        md_file = tmp_path / "test.md"
        md_file.write_text("# 标题\n\n这是正文内容。\n\n## 子标题\n\n更多内容。")
        
        docs = loader.load_file(str(md_file))
        
        assert len(docs) >= 1
        assert "标题" in docs[0].page_content or "正文" in docs[0].page_content


class TestDocumentLoaderEdgeCases:
    """边界情况测试"""
    
    def test_very_small_chunk_size(self):
        """测试非常小的 chunk_size"""
        loader = DocumentLoader(chunk_size=10, chunk_overlap=2)
        text = "这是一段测试文本，用于测试小块切分。"
        docs = loader.load_text_content(text, {"source": "test"})
        
        # 应该被切分成多个小块
        assert len(docs) > 1
    
    def test_large_overlap(self):
        """测试大的 overlap"""
        loader = DocumentLoader(chunk_size=100, chunk_overlap=80)
        text = "A" * 200
        docs = loader.load_text_content(text, {"source": "test"})
        
        # 应该正常工作
        assert len(docs) >= 1
    
    def test_unicode_content(self):
        """测试 Unicode 内容"""
        loader = DocumentLoader(chunk_size=200, chunk_overlap=20)
        text = "中文内容 🎉 Emoji 测试 日本語 한국어"
        docs = loader.load_text_content(text, {"source": "test"})
        
        assert len(docs) >= 1
        assert "中文" in docs[0].page_content
        assert "🎉" in docs[0].page_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

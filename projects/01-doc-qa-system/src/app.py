"""
智能文档问答系统 - 主应用

功能：
- 上传文档（PDF/Markdown/TXT）
- 构建知识库
- 智能问答
- 显示引用来源

运行：
cd projects/01-doc-qa-system
pip install -r requirements.txt
python src/app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# 加载环境变量（尝试多个路径）
env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # ai-agent-learning/.env
    Path(__file__).parent.parent / ".env",          # 01-doc-qa-system/.env
    Path.cwd() / ".env",                            # 当前目录
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ 加载环境变量: {env_path}")
        break

# API 配置
IFLOW_API_KEY = os.getenv("IFLOW_API_KEY")
IFLOW_BASE_URL = os.getenv("IFLOW_BASE_URL", "https://apis.iflow.cn/v1")
IFLOW_MODEL = os.getenv("IFLOW_MODEL", "qwen3-coder-plus")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICONFLOW_EMBEDDING_MODEL = os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3")


class DocQAApp:
    """文档问答应用"""
    
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.qa_engine = None
        self.documents = []
        
        self._init_models()
    
    def _init_models(self):
        """初始化模型"""
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        
        # 检查 API Key
        if not IFLOW_API_KEY:
            raise ValueError("请配置 IFLOW_API_KEY")
        if not SILICONFLOW_API_KEY:
            raise ValueError("请配置 SILICONFLOW_API_KEY")
        
        # 创建 LLM（用于对话）
        self.llm = ChatOpenAI(
            model=IFLOW_MODEL,
            openai_api_key=IFLOW_API_KEY,
            openai_api_base=IFLOW_BASE_URL,
            temperature=0,
        )
        
        # 创建 Embedding（用于向量化）
        self.embeddings = OpenAIEmbeddings(
            model=SILICONFLOW_EMBEDDING_MODEL,
            openai_api_key=SILICONFLOW_API_KEY,
            openai_api_base=SILICONFLOW_BASE_URL,
        )
        
        print("✅ 模型初始化完成")
        print(f"   对话模型: {IFLOW_MODEL}")
        print(f"   Embedding: {SILICONFLOW_EMBEDDING_MODEL}")
    
    def upload_files(self, files: List) -> str:
        """
        上传并处理文件
        
        Args:
            files: 上传的文件列表
            
        Returns:
            处理结果消息
        """
        from document_loader import DocumentLoader
        from vector_store import VectorStore
        
        if not files:
            return "❌ 请选择要上传的文件"
        
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        all_docs = []
        results = []
        
        for file in files:
            try:
                # Gradio 返回的是临时文件路径
                file_path = file.name if hasattr(file, 'name') else file
                docs = loader.load_file(file_path)
                all_docs.extend(docs)
                results.append(f"✅ {Path(file_path).name}: {len(docs)} 块")
            except Exception as e:
                results.append(f"❌ {Path(file_path).name}: {str(e)}")
        
        if all_docs:
            # 创建或更新向量数据库
            if self.vector_store is None:
                self.vector_store = VectorStore(self.embeddings)
            
            self.vector_store.add_documents(all_docs)
            self.documents.extend(all_docs)
            
            # 创建问答引擎
            from qa_engine import QAEngine
            self.qa_engine = QAEngine(
                self.llm,
                self.vector_store.as_retriever(search_kwargs={"k": 3}),
            )
            
            results.append(f"\n📊 知识库状态: 共 {len(self.documents)} 个文档块")
        
        return "\n".join(results)
    
    def upload_text(self, text: str, source_name: str = "用户输入") -> str:
        """
        直接上传文本内容
        
        Args:
            text: 文本内容
            source_name: 来源名称
            
        Returns:
            处理结果消息
        """
        from document_loader import DocumentLoader
        from vector_store import VectorStore
        
        if not text.strip():
            return "❌ 请输入文本内容"
        
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        docs = loader.load_text_content(text, {"source": source_name})
        
        # 创建或更新向量数据库
        if self.vector_store is None:
            self.vector_store = VectorStore(self.embeddings)
        
        self.vector_store.add_documents(docs)
        self.documents.extend(docs)
        
        # 创建问答引擎
        from qa_engine import QAEngine
        self.qa_engine = QAEngine(
            self.llm,
            self.vector_store.as_retriever(search_kwargs={"k": 3}),
        )
        
        return f"✅ 已添加 {len(docs)} 个文档块\n📊 知识库状态: 共 {len(self.documents)} 个文档块"
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        对话
        
        Args:
            message: 用户消息
            history: 对话历史
            
        Returns:
            (回复, 更新后的历史)
        """
        if not message.strip():
            return "", history
        
        if self.qa_engine is None:
            response = "⚠️ 请先上传文档构建知识库"
        else:
            try:
                result = self.qa_engine.ask(message)
                
                # 格式化回复（包含来源）
                response = result.answer
                if result.sources:
                    sources = set()
                    for doc in result.sources:
                        source = doc.metadata.get("source", "未知")
                        if isinstance(source, str):
                            sources.add(Path(source).name if "/" in source or "\\" in source else source)
                    if sources:
                        response += f"\n\n📚 参考来源: {', '.join(sources)}"
            except Exception as e:
                response = f"❌ 发生错误: {str(e)}"
        
        history.append((message, response))
        return "", history
    
    def clear_knowledge_base(self) -> str:
        """清空知识库"""
        self.vector_store = None
        self.qa_engine = None
        self.documents = []
        return "✅ 知识库已清空"
    
    def get_status(self) -> str:
        """获取当前状态"""
        if self.vector_store is None:
            return "📊 知识库状态: 未初始化\n请上传文档开始使用"
        
        stats = self.vector_store.get_stats()
        return f"""📊 知识库状态: {stats['status']}
📄 文档块数量: {stats['document_count']}
🤖 对话模型: {IFLOW_MODEL}
🔢 Embedding: {SILICONFLOW_EMBEDDING_MODEL}"""


def create_ui():
    """创建 Gradio UI"""
    import gradio as gr
    
    app = DocQAApp()
    
    with gr.Blocks(title="智能文档问答系统") as demo:
        gr.Markdown("""
        # 📚 智能文档问答系统
        
        上传文档，构建知识库，进行智能问答。支持 PDF、Markdown、TXT 格式。
        """)
        
        with gr.Row():
            # 左侧：文档上传
            with gr.Column(scale=1):
                gr.Markdown("### 📁 文档管理")
                
                # 文件上传
                file_upload = gr.File(
                    label="上传文档",
                    file_count="multiple",
                    file_types=[".pdf", ".md", ".txt", ".docx"],
                )
                upload_btn = gr.Button("📤 上传并处理", variant="primary")
                
                # 文本输入
                gr.Markdown("---")
                text_input = gr.Textbox(
                    label="或直接输入文本",
                    placeholder="在这里粘贴文本内容...",
                    lines=5,
                )
                text_source = gr.Textbox(
                    label="来源名称",
                    placeholder="例如：公司规章制度",
                    value="用户输入",
                )
                text_btn = gr.Button("📝 添加文本")
                
                # 状态显示
                gr.Markdown("---")
                status_display = gr.Textbox(
                    label="系统状态",
                    value=app.get_status(),
                    interactive=False,
                    lines=5,
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新状态")
                    clear_btn = gr.Button("🗑️ 清空知识库", variant="stop")
                
                # 上传结果
                upload_result = gr.Textbox(
                    label="处理结果",
                    interactive=False,
                    lines=3,
                )
            
            # 右侧：对话
            with gr.Column(scale=2):
                gr.Markdown("### 💬 智能问答")
                
                chatbot = gr.Chatbot(
                    label="对话",
                    height=500,
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入问题",
                        placeholder="请输入您的问题...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                
                clear_chat_btn = gr.Button("🗑️ 清空对话")
        
        # 示例问题
        gr.Markdown("### 💡 示例问题")
        gr.Examples(
            examples=[
                "这份文档的主要内容是什么？",
                "请总结一下关键信息",
                "有哪些重要的注意事项？",
            ],
            inputs=msg_input,
        )
        
        # 事件绑定
        upload_btn.click(
            fn=app.upload_files,
            inputs=[file_upload],
            outputs=[upload_result],
        ).then(
            fn=app.get_status,
            outputs=[status_display],
        )
        
        text_btn.click(
            fn=app.upload_text,
            inputs=[text_input, text_source],
            outputs=[upload_result],
        ).then(
            fn=app.get_status,
            outputs=[status_display],
        ).then(
            fn=lambda: "",
            outputs=[text_input],
        )
        
        send_btn.click(
            fn=app.chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        msg_input.submit(
            fn=app.chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        clear_chat_btn.click(
            fn=lambda: [],
            outputs=[chatbot],
        )
        
        refresh_btn.click(
            fn=app.get_status,
            outputs=[status_display],
        )
        
        clear_btn.click(
            fn=app.clear_knowledge_base,
            outputs=[upload_result],
        ).then(
            fn=app.get_status,
            outputs=[status_display],
        ).then(
            fn=lambda: [],
            outputs=[chatbot],
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("智能文档问答系统")
    print("=" * 60)
    
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

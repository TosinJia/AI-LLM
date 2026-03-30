from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.readers.file import FlatReader
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()
model = "qwen3-max-2026-01-23"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# LlamaIndex默认使用的大模型被替换为百炼
Settings.llm = DashScope(model=model, api_key=api_key, api_base=api_base_url)
# 加载本地的嵌入模型
Settings.embed_model = HuggingFaceEmbedding(model_name="/home/tosinjia/LLM/Local_model/BAAI/bge-large-zh-v1___5")

# 自定义你的提示词
# 建议：明确告诉 AI 保持简洁，并提取关键的关键词（如命令、字段名）
MY_CUSTOM_SUMMARY_QUERY = (
    "你是一个技术文档解析助手。请提取以下 Markdown 表格或内容的极简摘要。"
    "要求：1. 严禁啰嗦；2. 必须包含表格中的关键实体词（如 API 路径、参数名、状态码）；"
    "3. 如果是代码相关内容，请保留具体的命令名称。请用中文摘要"
)

# 读取文件+解析文档
md_docs = FlatReader().load_data(Path("/home/tosinjia/LLM/files/test.md"))
parser1 = MarkdownElementNodeParser(include_prev_next_rel=True, summary_query_str=MY_CUSTOM_SUMMARY_QUERY)
nodes = parser1.get_nodes_from_documents(md_docs)
print(nodes)

# 构建向量索引
index = VectorStoreIndex(nodes)


retriever = index.as_retriever(similarity_top_k=5)


# retriever.retrieve("张三多少岁？")
# 创建查询引擎
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

# 3. 组合成查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# 6. 测试查询
print("\n--- 测试查询 1：针对表格数据 ---")
response = query_engine.query("张三多少岁？")
print(response)
#
print("\n--- 测试查询 2：针对文本/代码内容 ---")
response = query_engine.query("李四是做什么的？")
print(response)
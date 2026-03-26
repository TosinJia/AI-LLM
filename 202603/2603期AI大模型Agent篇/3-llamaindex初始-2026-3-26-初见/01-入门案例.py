from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings  # llamaindex 的默认配置模块
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv
import os

load_dotenv()

model = "qwen3-max"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# 初始化千问模型(设置成默认)
Settings.llm = DashScope(model_name=model, api_key=api_key, api_base_url=api_base_url)
# 初始化嵌入模型
Settings.embed_model = HuggingFaceEmbedding(r"/home/tosinjia/LLM/Local_model/BAAI/bge-large-zh-v1___5")

# print(Settings.llm.complete("你好"))
# 1.加载文档
documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/公司规章制度.txt"]).load_data()
# 2.创建索引
index = VectorStoreIndex.from_documents(documents)

# 3.创建查询引擎
query_engine = index.as_query_engine()
print(query_engine.query("公司的上下班时间？"))

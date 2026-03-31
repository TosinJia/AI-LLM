from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
from llama_index.llms.deepseek import DeepSeek
from dotenv import load_dotenv
import os
import torch

load_dotenv()

# 检查GPU可用性
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_llm(model: str = "qwen3-max-2026-01-23"):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    api_base_url = os.getenv("DASHSCOPE_BASE_URL")

    # LlamaIndex默认使用的大模型被替换为百炼
    llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
    Settings.llm = llm

    # 加载本地的嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=r"/home/tosinjia/LLM/Local_model/BAAI/bge-small-zh-v1___5",
                                       device=device, embed_batch_size=2)
    # 设置默认的向量模型为本地模型
    Settings.embed_model = embed_model

    return llm, embed_model


def get_deepseek_llm(model: str = "deepseek-chat"):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base_url = os.getenv("DEEPSEEK_BASE_URL")

    # LlamaIndex默认使用的大模型被替换为百炼
    llm = DeepSeek(model=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
    Settings.llm = llm

    # 加载本地的嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=r"/home/tosinjia/LLM/Local_model/BAAI/bge-small-zh-v1___5",
                                       device=device, embed_batch_size=2)
    # 设置默认的向量模型为本地模型
    Settings.embed_model = embed_model

    return llm, embed_model


def get_qianfan_llm(model: str = "deepseek-chat"):
    api_key = os.getenv("QIANFAN_API_KEY")
    api_base_url = os.getenv("QIANFAN_BASE_URL")

    # LlamaIndex默认使用的大模型被替换为百炼
    llm = DeepSeek(model=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
    Settings.llm = llm

    # 加载本地的嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=r"/home/tosinjia/LLM/Local_model/BAAI/bge-small-zh-v1___5",
                                       device=device, embed_batch_size=2)
    # 设置默认的向量模型为本地模型
    Settings.embed_model = embed_model

    return llm, embed_model

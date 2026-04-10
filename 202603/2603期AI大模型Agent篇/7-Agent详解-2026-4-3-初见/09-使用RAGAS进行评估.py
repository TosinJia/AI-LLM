import os
import json
import pandas as pd
from dotenv import load_dotenv

# LlamaIndex 核心
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Ragas 核心
from ragas import EvaluationDataset, RunConfig # pip install ragas rapidfuzz
from ragas.testset import TestsetGenerator
from ragas.integrations.llama_index import evaluate
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas.metrics import (  # 很搞人，在官方文档中是过时版本的，
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

load_dotenv()


# ==========================================
# 1. 核心工具：中文生成拦截器 (解决英文数据集问题)
# ==========================================
class ChineseTestGenLLM:
    """
    包装 LlamaIndex 的 LLM，强制要求所有输出为中文。
    """

    def __init__(self, original_llm):
        self.llm = original_llm
        # 继承原模型的元数据（Ragas 可能会读取）
        self.metadata = original_llm.metadata

    async def acomplete(self, prompt, **kwargs):
        # 在 Prompt 末尾强行注入中文指令
        chinese_prompt = f"{prompt}\n\n[重要指令：请务必使用中文生成所有内容!]"
        return await self.llm.acomplete(chinese_prompt, **kwargs)

    async def achat(self, messages, **kwargs):
        # 针对聊天模式的汉化
        if messages:
            messages[-1].content += "\n(请务必使用中文进行回复)"
        return await self.llm.achat(messages, **kwargs)

    def __getattr__(self, name):
        # 代理其他所有原生方法
        return getattr(self.llm, name)


# ==========================================
# 2. 初始化配置
# ==========================================
def init_all():
    # 原始 DeepSeek 实例
    base_llm = DeepSeek(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base=os.getenv("DEEPSEEK_BASE_URL"),
        is_chat_model=True,
    )

    # 本地 Embedding
    embed_model = HuggingFaceEmbedding(
        model_name=r"/home/tosinjia/LLM/Local_model/BAAI/bge-large-zh-v1___5"
    )

    # 汉化后的 LLM（仅用于生成）
    chinese_gen_llm = ChineseTestGenLLM(base_llm)

    Settings.llm = base_llm
    Settings.embed_model = embed_model

    return base_llm, chinese_gen_llm, embed_model


# ==========================================
# 3. 数据集逻辑 (自动判断：加载 vs 生成)
# ==========================================
def get_or_generate_chinese_dataset(documents, gen_llm, embed_model, file_path="ragas_chinese_dataset.json"):
    if os.path.exists(file_path):
        print(f"--- 发现本地中文数据集，直接加载: {file_path} ---")
        return EvaluationDataset.from_pandas(pd.read_json(file_path))

    print("--- 正在生成中文测试集 (基于拦截器模式) ---")

    # 使用汉化版的 LLM 初始化生成器
    generator = TestsetGenerator.from_llama_index(
        llm=gen_llm,
        embedding_model=embed_model,
    )

    # 生成 5 组测试数据
    testset = generator.generate_with_llamaindex_docs(
        documents,
        testset_size=5,
    )

    # 导出并保存
    df_local = testset.to_pandas()
    # 补齐 persona_name
    if 'persona_name' not in df_local.columns or df_local['persona_name'].isnull().any():
        df_local['persona_name'] = df_local['persona_name'].fillna("通用玄幻书迷")

    # 补齐 query_style
    if 'query_style' not in df_local.columns or df_local['query_style'].isnull().any():
        df_local['query_style'] = df_local['query_style'].fillna("Standard_Chinese")

    # 补齐 query_length
    if 'query_length' not in df_local.columns or df_local['query_length'].isnull().any():
        df_local['query_length'] = df_local['query_length'].fillna("Medium")
    df_local.to_json(path_or_buf=file_path, orient="records", force_ascii=False)
    print(f"--- 中文测试集已保存至: {file_path} ---")

    return EvaluationDataset.from_pandas(df_local)


# ==========================================
# 4. 执行主流程
# ==========================================
if __name__ == "__main__":
    # A. 初始化
    base_llm, chinese_gen_llm, embed_model = init_all()

    # B. 加载本地文档
    doc_path = "/home/tosinjia/LLM/files/小说.txt"
    if not os.path.exists(doc_path):
        print("错误：未找到文档文件。")
        exit()
    documents = SimpleDirectoryReader(input_files=[doc_path]).load_data()

    # C. 生成/获取中文数据集
    ragas_dataset = get_or_generate_chinese_dataset(documents, chinese_gen_llm, embed_model)

    # D. 构建评估用的查询引擎
    vector_index = VectorStoreIndex.from_documents(documents)
    query_engine = vector_index.as_query_engine()

    # E. 配置评估指标 (评估时使用原生的 base_llm)， 需要将llamaindex的模型和嵌入模型转变成RAGAS支持的LlamaIndexLLMWrapper，LlamaIndexEmbeddingsWrapper
    eval_llm = LlamaIndexLLMWrapper(base_llm)
    eval_embed = LlamaIndexEmbeddingsWrapper(embed_model)

    # 初始化四种评估器
    metrics = [
        Faithfulness(llm=eval_llm),  # 忠实度
        AnswerRelevancy(llm=eval_llm, embeddings=eval_embed, strictness=1), # 答案相关性
        ContextPrecision(llm=eval_llm), # 上下文精确度
        ContextRecall(llm=eval_llm),  # 上下文的召回率
    ]

    # F. 开启评估
    print("--- 开始 RAG 性能评估 ---")
    result = evaluate(
        query_engine=query_engine,
        metrics=metrics,
        dataset=ragas_dataset,
        run_config=RunConfig(max_retries=3, timeout=120)
    )

    # G. 结果分析
    print("\n[ 评估总结 ]")
    print(result)
    result.to_pandas().to_csv("chinese_evaluation_report.csv", index=False, encoding="utf-8-sig")

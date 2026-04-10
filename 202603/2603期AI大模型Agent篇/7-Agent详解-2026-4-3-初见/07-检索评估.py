import asyncio
import pandas as pd
import random
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    generate_question_context_pairs
)
from 加载模型 import get_llm

# --- 配置区 ---
DATA_PATH = "/home/tosinjia/LLM/files/小说.txt"
SAVE_PATH = "小说_eval_dataset.json"
SAMPLE_NODE_COUNT = 30  # 抽样节点数，设为 None 则处理全量
TOP_K_LIST = [2, 5]  # 想要对比的检索深度


async def main():
    # 1. 环境初始化
    llm, embed_model = get_llm()

    # 2. 加载与解析文档
    print("📖 加载文档中...")
    documents = SimpleDirectoryReader(input_files=[DATA_PATH]).load_data()
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    # 抽样逻辑：节点太多是导致生成变慢的根本原因
    eval_nodes = nodes
    if SAMPLE_NODE_COUNT and len(nodes) > SAMPLE_NODE_COUNT:
        print(f"随机抽样 {SAMPLE_NODE_COUNT} 个节点进行评估...")
        eval_nodes = random.sample(nodes, SAMPLE_NODE_COUNT)

    # 3. 自动化数据集处理

    print(f"⏳ 开始生成评估数据集（节点数: {len(eval_nodes)}）...")
    # 优化提示词，确保生成质量
    qa_generate_prompt_tmpl = """基于以下上下文，生成 {num_questions_per_chunk} 个测验问题。
    仅使用上下文信息，不要结合外部知识。格式清晰。
    上下文：{context_str}"""

    qa_dataset = generate_question_context_pairs(
        eval_nodes,
        llm=llm,
        num_questions_per_chunk=1,  # 速度优先，每个块生成一个问题
        qa_generate_prompt_tmpl=qa_generate_prompt_tmpl
    )
    qa_dataset.save_json(SAVE_PATH)
    print(f"✅ 数据集已保存至 {SAVE_PATH}")

    # 4. 构建索引（仅需一次）
    index = VectorStoreIndex(nodes)

    # 5. 多维度异步评估对比
    metrics = ["hit_rate", "mrr", "precision", "recall"]
    results_list = []

    print(f"📊 开始异步评估检索器 (对比 Top-K: {TOP_K_LIST})...")

    # 创建所有待执行的评估任务
    eval_tasks = []
    for k in TOP_K_LIST:
        retriever = index.as_retriever(similarity_top_k=k)
        evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)
        # 将协程任务加入列表
        eval_tasks.append(evaluator.aevaluate_dataset(qa_dataset))

    # 并行执行所有评估任务
    all_eval_results = await asyncio.gather(*eval_tasks)

    # 6. 整理并展示结果
    for i, eval_results in enumerate(all_eval_results):
        name = f"Top-{TOP_K_LIST[i]}"
        df = pd.DataFrame([res.metric_vals_dict for res in eval_results])
        avg_df = df.mean().to_frame(name=name).T
        results_list.append(avg_df)

    final_report = pd.concat(results_list)
    print("\n" + "=" * 50)
    print("🏆 检索器性能最终报告")
    print("=" * 50)
    print(final_results_format(final_report))
    print("=" * 50)


def final_results_format(df):
    """美化输出格式"""
    return df.style.format("{:.4f}").to_string()


if __name__ == '__main__':
    asyncio.run(main())
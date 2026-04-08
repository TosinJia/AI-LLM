import asyncio
import os
import pandas as pd
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    generate_question_context_pairs,
)
from 加载模型 import get_llm


# --- 1. 固定 ID 的层次化解析函数 ---
def get_hierarchical_nodes_fixed(documents, chunk_sizes=[1024, 512, 128]):
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes


async def main():
    # 初始化模型
    llm, embed_model = get_llm()

    # 加载文档
    documents = SimpleDirectoryReader(input_files=["../data/小说.txt"]).load_data()

    # 2. 生成层次化节点
    print("正在进行层次化解析 (1024 -> 512 -> 128)...")
    all_nodes = get_hierarchical_nodes_fixed(documents)
    leaf_nodes = get_leaf_nodes(all_nodes)  # 叶子节点用于向量检索

    # 3. 设置存储上下文 (AutoMergingRetriever 需要 docstore 记录父子关系)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(all_nodes)
    
    # 4. 构建索引（仅针对叶子节点）
    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

    # 5. 生成评估数据集
    # 注意：生成问题时建议使用叶子节点，这样 ID 匹配最精准
    dataset_path = "hierarchical_eval_dataset.json"
    print("正在生成评估数据集（基于叶子节点）...")
    qa_dataset = generate_question_context_pairs(
        leaf_nodes[:30],  # 抽样 30 个叶子节点
        llm=llm,
        num_questions_per_chunk=1
    )
    qa_dataset.save_json(dataset_path)

    # 6. 定义自动合并检索器
    # 它会先找 10 个叶子节点，如果某个父节点下的子节点够多，就自动合并成父节点
    base_retriever = index.as_retriever(similarity_top_k=10)
    merging_retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,
        verbose=False
    )

    # 7. 评估
    print("开始评估层次化检索性能...")
    metrics = ["hit_rate", "mrr", "precision", "recall"]
    evaluator = RetrieverEvaluator.from_metric_names(metrics, retriever=merging_retriever)

    eval_results = await evaluator.aevaluate_dataset(qa_dataset)

    # 8. 展示结果
    df = pd.DataFrame([res.metric_vals_dict for res in eval_results])
    final_res = df.mean().to_frame(name="Hierarchical-AutoMerging").T

    print("\n" + "=" * 50)
    print("层次化 RAG 评估报告")
    print("=" * 50)
    print(final_res)
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
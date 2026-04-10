import asyncio
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator, BatchEvalRunner
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from 加载模型 import get_llm


async def main():
    # 1. 初始化模型设置 (全局配置)
    llm, embed_model = get_llm()

    # 2. 准备索引
    print("正在构建索引...")
    documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # 3. 初始化评估器
    # 建议：评估模型可以选比生成模型更强的（如 GPT-4），结果更客观
    faith_evaluator = FaithfulnessEvaluator(llm=llm)  # 忠实度评估器
    rel_evaluator = RelevancyEvaluator(llm=llm)  # 答案相关性评估器

    # 4. 批量查询与评估 (核心优化点)
    queries = [
        "萧炎的爸爸是谁？",  # 基础题（预期：1.0）
        "萧炎最喜欢的现代流行歌手是谁？",  # 跨时空无关（预期：Faithfulness 应该为 1.0，但回答应为“不知道”）
        "萧炎在第一章里一共喝了几杯咖啡？",  # 逻辑陷阱（玄幻小说没咖啡，看它是否产生幻觉）
        "作者天蚕土豆的家庭住址在哪里？",  # 外部元数据（文档里没写作者隐私，看它是否拒绝）
        "萧炎用什么牌子的智能手机和药老联系？"  # 严重干扰项
    ]

    print(f"\n开始批量执行 {len(queries)} 组评估...")

    # 使用 BatchEvalRunner 进行并行异步评估
    # 相比于 for 循环逐个评估，BatchEvalRunner 能显著提高 Token 利用率和执行速度
    runner = BatchEvalRunner(
        {
            "faithfulness": faith_evaluator,
            "relevancy": rel_evaluator,
        },
        show_progress=True,
        workers=4  # 根据 API 限制调整并发数
    )

    # aevaluate_queries 会自动执行：查询 -> 获取 Response -> 调用各评估器
    eval_results = await runner.aevaluate_queries(
        query_engine,
        queries=queries
    )

    # 5. 格式化结果输出
    print("\n" + "=" * 50)
    print("评估报告汇总")
    print("=" * 50)

    for query in queries:
        print(f"查询问题: {query}")

        print("RAG最终的回复：", query_engine.query(query).response)

        # 提取各个维度的结果
        f_res = eval_results["faithfulness"][queries.index(query)]
        r_res = eval_results["relevancy"][queries.index(query)]

        # 打印详细打分
        print(f"  [忠实度 Faithfulness]: {'通过' if f_res.passing else '❌ 失败'} (得分: {f_res.score:.2f})")
        if not f_res.passing:
            print(f"    └─ 反馈: {f_res.feedback}")

        print(f"  [相关性 Relevancy   ]: {'通过' if r_res.passing else '❌ 失败'} (得分: {r_res.score:.2f})")
        if not r_res.passing:
            print(f"    └─ 反馈: {r_res.feedback}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.retrievers.bm25 import BM25Retriever # pip install llama-index-retrievers-bm25
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.workflow import (
    Context,
    Event,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.response_synthesizers import get_response_synthesizer
from typing import List
import asyncio
from 加载模型 import get_llm


# 定义工作流中的事件类型
class QueryEvent(Event):
    """查询事件"""
    query: str


class VectorRetrievalEvent(Event):
    """向量检索事件"""
    nodes: List[NodeWithScore]
    query: str


class BM25RetrievalEvent(Event):
    """关键词检索事件"""
    nodes: List[NodeWithScore]
    query: str


class CombinedRetrievalEvent(Event):
    """合并检索结果事件"""
    vector_nodes: List[NodeWithScore]
    bm25_nodes: List[NodeWithScore]
    query: str


class PostProcessEvent(Event):
    """后处理事件"""
    processed_nodes: List[NodeWithScore]
    query: str


class RAGWorkflow(Workflow):
    """RAG工作流类"""

    def __init__(self, retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever):
        super().__init__()
        self.retriever = retriever
        self.bm25_retriever = bm25_retriever
        self.postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)
        self.response_synthesizer = get_response_synthesizer()

    @step
    async def query_step(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """步骤1: 处理用户查询"""
        # 进行问题改写，多问题的生成
        query = ev.query
        print(f"🔍 接收查询: {query}")
        processed_query = query.strip()
        return QueryEvent(query=processed_query)

    @step
    async def vector_retrieval_step(self, ctx: Context, ev: QueryEvent) -> VectorRetrievalEvent:
        """步骤2: vector向量数据库检索相关文档"""
        print(f"📚 vector开始检索相关文档...")

        query_bundle = QueryBundle(query_str=ev.query)
        retrieved_nodes = await self.retriever.aretrieve(query_bundle)

        print(f"✅ vector检索到 {len(retrieved_nodes)} 个相关文档片段")
        return VectorRetrievalEvent(nodes=retrieved_nodes, query=ev.query)

    @step
    async def bm25_retrieval_step(self, ctx: Context, ev: QueryEvent) -> BM25RetrievalEvent:
        """步骤2: bm25检索相关文档"""
        print(f"📚 bm25开始检索相关文档...")

        query_bundle = QueryBundle(query_str=ev.query)
        retrieved_nodes = await self.bm25_retriever.aretrieve(query_bundle)

        print(f"✅ bm25检索到 {len(retrieved_nodes)} 个相关文档片段")
        return BM25RetrievalEvent(nodes=retrieved_nodes, query=ev.query)

    @step
    async def combine_results_step(
            self,
            ctx: Context,
            ev: VectorRetrievalEvent | BM25RetrievalEvent
    ) -> CombinedRetrievalEvent:
        """步骤3: 收集并合并两个检索结果"""
        print(f"🔧 开始收集检索结果...")

        # 使用 collect_events 收集两种类型的事件
        events = ctx.collect_events(ev, [VectorRetrievalEvent, BM25RetrievalEvent])

        # 如果说只有VectorRetrievalEvent节点完成进入当前合并节点，那么合并节点会等待BM25RetrievalEvent完成成功进入BM25RetrievalEvent
        if not events or len(events) < 2:
            print(f"⚠️ 只收集到 {len(events) if events else 0} 个事件，等待更多...")
            # 如果没有收集到足够的事件，返回 None 让工作流继续等待
            return None

        print(f"✅ 已收集到 {len(events)} 个检索事件")

        # 分离不同类型的检索结果
        vector_nodes = []
        bm25_nodes = []
        query = ""

        for event in events:
            if isinstance(event, VectorRetrievalEvent):
                vector_nodes = event.nodes
                query = event.query
                print(f"  - Vector检索: {len(event.nodes)} 个节点")
            elif isinstance(event, BM25RetrievalEvent):
                bm25_nodes = event.nodes
                query = event.query
                print(f"  - BM25检索: {len(event.nodes)} 个节点")

        return CombinedRetrievalEvent(
            vector_nodes=vector_nodes,
            bm25_nodes=bm25_nodes,
            query=query
        )

    @step
    async def postprocess_step(self, ctx: Context, ev: CombinedRetrievalEvent) -> PostProcessEvent:
        """步骤4: 对合并的检索结果进行后处理"""
        print(f"🔄 开始后处理检索结果...")

        # 合并所有检索结果
        all_nodes = []
        all_nodes.extend(ev.vector_nodes)
        all_nodes.extend(ev.bm25_nodes)

        if not all_nodes:
            print("⚠️  没有找到任何检索结果")
            return PostProcessEvent(processed_nodes=[], query=ev.query)

        print(f"🔄 开始后处理 {len(all_nodes)} 个文档片段...")
        print(f"  - Vector节点: {len(ev.vector_nodes)} 个")
        print(f"  - BM25节点: {len(ev.bm25_nodes)} 个")

        # 创建查询束用于后处理
        query_bundle = QueryBundle(query_str=ev.query)

        # 执行后处理（去重、过滤、重排序等）
        processed_nodes = self.postprocessor.postprocess_nodes(
            nodes=all_nodes, query_bundle=query_bundle
        )

        print(f"✅ 后处理完成，保留 {len(processed_nodes)} 个高质量文档片段")

        # 打印每个节点的相似度分数
        for i, node in enumerate(processed_nodes[:3]):
            score = node.score if node.score else 0
            print(f"  - 文档片段 {i + 1}: 相似度 {score:.3f}")

        return PostProcessEvent(processed_nodes=processed_nodes, query=ev.query)

    @step
    async def synthesis_step(self, ctx: Context, ev: PostProcessEvent) -> StopEvent:
        """步骤5: 基于检索到的上下文生成最终答案"""
        print(f"🤖 开始生成答案...")

        if not ev.processed_nodes:
            return StopEvent(result={
                "response": "抱歉，没有找到相关信息来回答的问题。",
                "source_nodes": []
            })

        # 创建查询束
        query_bundle = QueryBundle(query_str=ev.query)

        # 使用响应合成器生成答案
        response = await self.response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=ev.processed_nodes
        )

        print(f"✅ 答案生成完成")

        return StopEvent(result={
            "response": str(response),
            "source_nodes": ev.processed_nodes,
            "metadata": {
                "num_sources": len(ev.processed_nodes),
                "query": ev.query
            }
        })


# 使用示例
async def main():
    """主函数示例"""
    print("📖 正在构建向量索引...")

    get_llm()

    documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()
    splitter = SentenceSplitter(chunk_size=512)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index, similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=3
    )

    print("✅ 向量索引构建完成")

    workflow = RAGWorkflow(retriever=retriever, bm25_retriever=bm25_retriever)

    test_queries = [
        "萧炎的爸爸是谁？",
        "萧炎的妹妹是谁？"
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"🎯 测试查询: {query}")
        print(f"{'=' * 50}")

        result = await workflow.run(query=query)

        print(f"\n📝 生成的答案:")
        print(f"{result['response']}")
        print(f"\n📊 元数据:")
        print(f"- 使用了 {result['metadata']['num_sources']} 个文档片段")
        print(f"- 原始查询: {result['metadata']['query']}")


if __name__ == "__main__":
    asyncio.run(main())
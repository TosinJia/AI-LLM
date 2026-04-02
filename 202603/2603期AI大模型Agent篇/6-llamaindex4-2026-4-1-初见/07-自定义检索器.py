from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from typing import List
from 加载模型 import get_llm

# 加载大模型和嵌入模型
llm, embed_model = get_llm()

# 加载文档
documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()
# 初始化节点解析器
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)


class CustomRetriever(BaseRetriever):
    """执行语义搜索和简单关键字搜索的自定义检索器。"""
    # 手动创建一个混合检索

    def __init__(
            self,
            vector_retriever: VectorIndexRetriever,
            keyword_retriever: KeywordTableSimpleRetriever,
            mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        # 分别从两个引擎中检索相关的节点（卡片）
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        # 提取两组结果的 Node ID，方便做数学集合运算  {"1", "2", "3", "4", "5", "6"}
        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        # 创建一个字典，把 ID 和节点对象关联起来，方便通过 ID 找回对象；去重后的上下文
        combined_dict = {n.node.node_id: n for n in vector_nodes}  # 将vector_nodes中的数据变成{"1": node, "2": node}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes}) # 将keyword_nodes中的数据变成{"1": node, "2": node}

        if self._mode == "AND":
            # 获取两个检索器交集(重叠)的数据
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            # 获取两个检索器并集（不重叠）的数据
            retrieve_ids = vector_ids.union(keyword_ids)

        # 根据最终确定的 ID 列表，从字典中取出完整的节点对象
        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


# 初始化上下文存储器
storage_context = StorageContext.from_defaults()

# 创建对应的索引
vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
# 简单关键词索引，适合结构化数据或者短文本查询
keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

# 定义自定义检索器
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=2)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
# 使用自己创建的检索器类
custom_retriever = CustomRetriever(vector_retriever, keyword_retriever)

# 定义响应合成器
response_synthesizer = get_response_synthesizer()

# 加载查询引擎
custom_query_engine = RetrieverQueryEngine(
    retriever=custom_retriever,
    response_synthesizer=response_synthesizer,
)

response = custom_query_engine.query("斗之气：九段！级别：高级！是谁")
print(response)
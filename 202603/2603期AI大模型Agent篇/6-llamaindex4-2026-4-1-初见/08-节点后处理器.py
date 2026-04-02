from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from datetime import datetime, timedelta
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import TimeWeightedPostprocessor
from llama_index.core import SimpleDirectoryReader
from 加载模型 import get_llm

# 加载大模型和嵌入模型
llm, embed_model = get_llm()

nodes = [
    NodeWithScore(node=Node(text="张三的爱车是小丽"), score=0.5),
    NodeWithScore(node=Node(text="张三的女朋友是晓丽"), score=0.8),
]

# 基于相似度的后处理器：过滤相似度得分低于0.75
processor = SimilarityPostprocessor(similarity_cutoff=0.75)
# 过滤节点
filtered_nodes = processor.postprocess_nodes(nodes)
print(filtered_nodes)

# 使用本地的重排序模型进行重排
reranker = SentenceTransformerRerank(model=r"/home/tosinjia/LLM/Local_model/BAAI/bge-reranker-large", top_n=2)
print(reranker.postprocess_nodes(nodes, query_str="张三的女朋友是谁？"))

print("------------与检索到的文档一起使用-------------------------")
# 加载文档
documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()
# 创建向量索引
index = VectorStoreIndex.from_documents(documents)
# 进行向量检索出相似的文档
response_nodes = index.as_retriever(similarity_top_k=5).retrieve("萧炎的妹妹是谁？")
# 基于相似度的后处理器：过滤相似度得分低于0.5
processor = SimilarityPostprocessor(similarity_cutoff=0.52)
print(processor.postprocess_nodes(response_nodes))

print("------------使用查询引擎----------------")
# 1. 构造带时间戳的文档数据
now = datetime.now()
documents = [
    Document(
        text="我们的退货政策是：在30天内可退货。",
        metadata={"created_at": now - timedelta(days=40)}  # 较早
    ),
    Document(
        text="我们最近更新了退货政策，现在是15天内可退货。",
        metadata={"created_at": now - timedelta(days=10)}  # 比较新
    ),
    Document(
        text="退货政策是，目前可以20天内可退货",
        metadata={"created_at": now - timedelta(days=1)}  # 最新
    )
]

# 2. 构建索引和向量检索器
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)

# 3. 创建 TimeWeightedPostprocessor
#  TimeWeightedPostprocessor 是 LlamaIndex 中的一个后处理器，用于根据文档节点的 时间戳（timestamp）进行加权排序或过滤，以优先考虑更新更近、时间更相关的内容。
# 本质还是会先按照语义搜索，尽管你有一个最新的文档，但是如果他的内容和问题相差太大也是和最终检索的文档排序有影响的
time_postprocessor = TimeWeightedPostprocessor(
    time_decay=0.5,  # 控制文档的“新旧信息”衰减速度。值越大，越快忽略旧的内容。
    top_k=3  # 最多返回3条
)

# 4. 构建 QueryEngine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    node_postprocessors=[time_postprocessor]
)

# 5. 用户提问
query = "你们现在的退货政策是怎样的？"
response = query_engine.query(query)

print("📌 回答：", response)

for node in response.source_nodes:
    print(node.text)
    print("score:", node.score)
    print("created_at:", node.metadata.get("created_at"))
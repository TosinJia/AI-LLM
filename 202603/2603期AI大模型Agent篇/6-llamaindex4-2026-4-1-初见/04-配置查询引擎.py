# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from 加载模型 import get_llm

# 加载大模型和嵌入模型
llm, embed_model = get_llm()

# # 加载文档
documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()
# # # 创建索引对象
# index = VectorStoreIndex.from_documents(documents)
# #
# # # 查询引擎用来提问
# res = index.as_query_engine(streaming=True).query("萧炎，斗之力？")
# print(res)
#
# # 流式输出
# res.print_response_stream()
""" 
    虽然通过以下代码对易用性进行了优化，但它并未公开全部的可配置性。
        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True,
        )
    如果需要更精细的控制，可以使用低级组合 API。具体来说，你需要显式地构造一个QueryEngine对象，而不是调用index.as_query_engine(...)
)
"""
print("================显式构造QueryEngine=====================")
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode

# 创建索引
index = VectorStoreIndex.from_documents(documents)
# 创建检索器
retriever = index.as_retriever(
    similarity_top_k=2,
)

# 配置响应合成器
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    streaming=True
)

# 组装查询引擎
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer
)

# 提问
response = query_engine.query("萧炎的妹妹叫什么名字?")
# 普通输出
# print(response)

# 流式输出
response.print_response_stream()
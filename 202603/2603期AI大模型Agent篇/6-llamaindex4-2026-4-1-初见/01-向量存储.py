from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from 加载模型 import get_llm

# 初始化模型
llm, embed_model = get_llm()

# 加载文档
documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()
#
# # 初始化向量存储,创建一个内存的向量存储
# vector_store = SimpleVectorStore()
#
# # 创建一个存储的容器
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# # 创建一个向量存储索引
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#
# # 创建一个查询引擎
# # res = index.as_query_engine().query("萧炎是谁？")
# # print(res)
#
# storage_context.persist("./storage")
# # 加载本地存储
# new_storage_context = StorageContext.from_defaults(persist_dir="./storage")
# 初始化chroma
chroma_client = chromadb.PersistentClient()
# 创建连接对应
chroma_collection = chroma_client.get_or_create_collection("test")
# 创建向量存储
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 创建一个存储的容器
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# 创建一个向量存储索引
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

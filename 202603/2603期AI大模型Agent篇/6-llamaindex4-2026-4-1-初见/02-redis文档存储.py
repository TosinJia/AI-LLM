# llama-index-storage-docstore-redis

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.storage.docstore.redis import RedisDocumentStore   # pip install llama-index-storage-docstore-redis

from 加载模型 import get_llm

# 初始化模型
llm, embed_model = get_llm()

# 加载文档
documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()

# 初始化RedisDocumentStore对象
doc_store = RedisDocumentStore.from_host_and_port(host="localhost", port=6379,
                                      namespace="redis_document_store")
# 第一种方式
# doc_store.add_documents(documents)



storage_context = StorageContext.from_defaults(docstore=doc_store)
# 第二种方式：创建VectorStoreIndex才能进行存储
# 创建上下文存储容器VectorStoreIndex
VectorStoreIndex.from_documents(documents, storage_context=storage_context)

print(storage_context.docstore.docs)


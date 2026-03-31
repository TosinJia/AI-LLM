# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.extractors import TitleExtractor
# from llama_index.core.ingestion import IngestionPipeline, IngestionCache
# from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache    # pip install llama_index.storage.kvstore.redis
# from llama_index.core import SimpleDirectoryReader
# import redis
# from 加载模型 import get_llm
#
# llm, embed_model = get_llm()
#
# # 定义数据连接器去读取数据
# documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()
#
# ingest_cache = IngestionCache(
#     cache=RedisCache.from_redis_client(redis.Redis(
#         host="127.0.0.1",
#         port=6379,
#         decode_responses=True,
#         charset="utf-8",
#         encoding="utf-8"
#     )),
#     collection="my_test_cache",
# )
#
# pipeline = IngestionPipeline(
#     transformations=[
#         SentenceSplitter(chunk_size=250, chunk_overlap=50),
#         TitleExtractor(),
#         embed_model,
#     ],
#     cache=ingest_cache,
# )
#
# # 直接将数据摄取到向量数据库
# pipeline.run(documents=documents)
#
# # 加载和恢复状态
# new_pipeline = IngestionPipeline(
#     transformations=[
#         SentenceSplitter(chunk_size=250, chunk_overlap=50),
#         TitleExtractor(),
#         embed_model,
#     ],
#     cache=ingest_cache,
# )
#
# # 由于缓存的存在会立即执行
# nodes = new_pipeline.run(documents=documents)
#
# print(nodes)
# for node in nodes:
#     print(node, "\n\n")

print("----------------------直接查询 Redis 数据库----------------------")
import redis

redis_client = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True, charset="utf-8")

# 查看所有 keys
all_keys = redis_client.keys("my_test_cache*")
print(f"Redis 中的所有相关 keys: {len(all_keys)} 个")

# 查看前几个 key 的内容
for key in all_keys[:3]:
    value = redis_client.hgetall(key)
    print(f"Key: {key}")
    for k, v in value.items():
        # 将存储的内容进行转义
        print(f"Value: {v.encode('utf-8').decode('unicode_escape')}")
    print("-" * 30)

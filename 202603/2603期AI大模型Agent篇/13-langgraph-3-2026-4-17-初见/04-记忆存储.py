from langgraph.store.memory import InMemoryStore
import uuid

# # 创建存储
# in_memory_store = InMemoryStore()
#
# # 定义命名空间   是用来区分不同用户
# namespace_for_memory = ("user_id", "memories")
#
# # 存储记忆
# memory_id = str(uuid.uuid4())
# memory = {"hobby": "篮球、音乐、美食、编程..."}
# # 手动存储相关的内容
# in_memory_store.put(namespace_for_memory, memory_id, memory)
#
#
# # 和redis
# in_memory_store.get(namespace_for_memory, memory_id)
#
# # 搜索记忆
# memories = in_memory_store.search(namespace_for_memory)
# # 打印数据
# print(memories[-1].dict())

print("-" * 8, "语义搜索", "-" * 8)
from langchain_huggingface import HuggingFaceEmbeddings # pip install langchain-huggingface ; pip install sentence-transformers

namespace_for_memory = ("user_id", "memories")

store = InMemoryStore(
    index={
        "embed": HuggingFaceEmbeddings(model_name=r"/home/tosinjia/LLM/Local_model/BAAI/bge-large-zh-v1___5"),
        "dims": 1024,
        "fields": ["hobby", "food_preference"]  # 存储的key
    }
)

# 3. 存储数据并检查
memory_id_1 = str(uuid.uuid4())
memory_1 = {"hobby": "我的爱好是：篮球、音乐、美食、编程..."}
store.put(namespace_for_memory, memory_id_1, memory_1)
print(f"✓ 存储 hobby 记忆: {memory_id_1}")

memory_id_2 = str(uuid.uuid4())
memory_2 = {"food_preference": "我最喜欢的美食是：臭豆腐、小龙虾、红烧肉..."}
store.put(namespace_for_memory, memory_id_2, memory_2)
print(f"✓ 存储 food_preference 记忆: {memory_id_2}")

# 4. 检查存储的数据
print("\n=== 调试信息 ===")
print(f"Namespace: {namespace_for_memory}")
print(f"存储的记忆数量: {len(store.search(namespace_for_memory))}")

# 5. 搜索测试
print("\n=== 搜索测试 ===")

# 测试 1: 搜索食物偏好
print("搜索: 用户喜欢吃什么？")
memories = store.search(
    namespace_for_memory,
    query="用户喜欢吃什么？",
    limit=3
)
print(f"搜索结果数量: {len(memories)}")
if memories:
    print(f"最相关结果: {memories[0].dict()}")
else:
    print("没有找到结果")

# 测试 2: 搜索爱好
print("\n搜索: 用户的爱好有哪些？")
memories = store.search(
    namespace_for_memory,
    query="用户的爱好有哪些？",
    limit=3
)
print(f"搜索结果数量: {len(memories)}")
if memories:
    print(f"最相关结果: {memories[0].dict()}")
else:
    print("没有找到结果")
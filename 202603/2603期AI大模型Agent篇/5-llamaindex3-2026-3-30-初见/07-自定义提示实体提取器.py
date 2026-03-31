from typing import List

from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core import PropertyGraphIndex
from llama_index.core import SimpleDirectoryReader
from 加载模型 import get_llm

llm, embed_model = get_llm()

# 加载文档并构建索引
documents = SimpleDirectoryReader(
    input_files=["/home/tosinjia/LLM/files/小说.txt"]
).load_data()

prompt = """从以下文本中提取实体和它们之间的关系。
            请按照以下格式输出，每行一个关系：
            实体1|关系|实体2

            文本: {text}

            提取的关系:
        """


def parse_function(llm_output: str) -> List[List[str]]:
    """
    基础解析函数 - 解析简单的三元组格式
    输入: "实体1|关系|实体2" 格式的文本
    输出: [["实体1", "关系", "实体2"], ...] 格式的列表
    """
    paths = []
    lines = llm_output.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # 分割实体和关系
        parts = line.split('|')
        if len(parts) == 3:
            entity1, relation, entity2 = [part.strip() for part in parts]
            if entity1 and relation and entity2:
                paths.append([entity1, relation, entity2])

    return paths


kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    extract_prompt=prompt,
    parse_fn=parse_function,
)

print("kg_extractor->", kg_extractor)

# 创建属性图
index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    show_progress=True  # 显示提取进度
)
# 查看结果
response = index.property_graph_store.get_triplets(entity_names=["萧炎"])
print("response->", response)
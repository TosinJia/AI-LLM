from llama_index.core.indices.property_graph import SchemaLLMPathExtractor
from llama_index.core import PromptTemplate
from llama_index.core import PropertyGraphIndex
from llama_index.core import Document
from typing import Literal
from 加载模型 import get_deepseek_llm

llm, embed_model = get_deepseek_llm()
# 定义提取模式
doc = [
    Document(
        text="张伟是北京大学的教授，研究方向是人工智能。他是李娜的博士导师，李娜现在在阿里巴巴达摩院从事自然语言处理相关的工作。王强是李娜的同事，"
             "他们一起参与了一个关于大模型推理的项目。"),
    Document(
        text="张伟教授发表了多篇关于深度学习的论文，他的研究团队包括3名博士生和5名硕士生。张伟教授在北京大学人工智能学院任教，专注于计算机视觉研究。"
    ),
    Document(
        text="李娜在阿里巴巴的项目涉及多模态AI，她和王强共同负责模型优化部分。李娜从北京大学获得了人工智能专业的博士学位。"
    ),
    Document(
        text="北京大学人工智能学院与阿里巴巴达摩院建立了合作关系，共同推进AI技术发展。双方在深度学习和自然语言处理领域开展深度合作。"
    ),
    Document(
        text="王强之前在腾讯工作，后来跳槽到阿里巴巴，专注于大模型推理加速技术。王强拥有清华大学计算机科学硕士学位。"
    ),
    Document(
        text="张伟教授的研究领域还包括计算机视觉和强化学习，他指导的学生分布在各大科技公司。他领导的AI实验室在国际会议上发表了超过50篇论文。",
    )
]

# 定义提取模式
entities = Literal["Person", "Location", "Organization", "Product", "Event"]
relations = Literal[
    "SUPPLIER_OF",
    "COMPETITOR",
    "PARTNERSHIP",
    "ACQUISITION",
    "WORKS_AT",
    "SUBSIDIARY",
    "BOARD_MEMBER",
    "CEO",
    "PROVIDES",
    "HAS_EVENT",
    "IN_LOCATION",
]
# 定义更详细的图谱模式
schema = {
    "Person": ["WORKS_AT", "BOARD_MEMBER", "CEO", "HAS_EVENT"],
    "Organization": [
        "SUPPLIER_OF",
        "COMPETITOR",
        "PARTNERSHIP",
        "ACQUISITION",
        "WORKS_AT",
        "SUBSIDIARY",
        "BOARD_MEMBER",
        "CEO",
        "PROVIDES",
        "HAS_EVENT",
        "IN_LOCATION",
    ],
    "Product": ["PROVIDES"],
    "Event": ["HAS_EVENT", "IN_LOCATION"],
    "Location": ["HAPPENED_AT", "IN_LOCATION"],
}
zh_extract_prompt_str = """
你是一个专业的知识图谱提取助手。你的任务是从给定的文本中提取结构化的三元组（主体-关系-客体）。

### 严格约束条件 (Schema)
1. **允许的实体类型**: {allowed_entity_types}
2. **允许的关系类型**: {allowed_relation_types}

### 关键规则 (必须严格遵守)
- **禁止造词**：提取出的 `type` 必须严格完全匹配上述列表中的英文单词。
- **自动归类**：如果文本中出现了列表之外的具体概念，请将其归类为列表中最接近的父类。
    - 例如：如果允许列表只有 "Organization"，但文本提到 "University" 或 "Company"，你必须输出 "Organization"。
- **关系归一化**：如果文本中的动词不在列表中，请映射为含义最接近的允许关系。
    - 例如：如果允许列表只有 "work_in"，但文本提到 "works at" 或 "employed by"，你必须输出 "work_in"。
- **格式要求**：仅输出标准的 JSON 格式，不要包含任何解释性文字。

### 示例 (Few-Shot)
文本: "张伟在北京大学任教。"
允许实体: ["Person", "Organization"]
允许关系: ["work_in"]
思考过程: "北京大学"是大学，属于 "Organization"。"任教"意味着在某处工作，映射为 "work_in"。
输出: {{ "triplets": [ {{ "subject": {{ "name": "张伟", "type": "Person" }}, "relation": {{ "type": "work_in" }}, "object": {{ "name": "北京大学", "type": "Organization" }} }} ] }}

### 开始任务
文本: {text}
输出:  
"""

# 创建基于模式的提取器
kg_extractor = SchemaLLMPathExtractor(
    extract_prompt=PromptTemplate(zh_extract_prompt_str),
    llm=llm,
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=schema,
    strict=False,  # 如果为 false，将允许超出模式范围的三元组
    num_workers=4,  # 并行处理
)

# 创建属性图
index = PropertyGraphIndex.from_documents(
    doc,
    llm=llm,
    embed_model=embed_model,
    kg_extractors=[kg_extractor],
    show_progress=True  # 显示提取进度
)
# 查看结果
# 检查属性是否存在
print(hasattr(index.property_graph_store, "graph"))

index.as_retriever()

# 如果存在，直接打印数量
if hasattr(index.property_graph_store, "graph"):
    print("内存图谱中的三元组总数:", index.property_graph_store.graph.triplets)
# response = index.property_graph_store.graph.get_triplets("张伟")
# print("response->", response)
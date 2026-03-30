from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document

# 示例文档
document = Document(text="这是第一个句子. 这是第二个句子. 这是第三个句子. 这是第四个句子. ")

# 创建句子窗口节点解析器
node_parser = SentenceWindowNodeParser(
    window_size=1,  # 窗口大小，即每个节点包含的句子数量
    window_metadata_key="window", # 对应的上下文的key
    original_text_metadata_key="original_text"  # 原有node的key
)

# 从文档中获取节点
nodes = node_parser.get_nodes_from_documents([document])
# 打印生成的节点
for node in nodes:
    print(node.text, node.metadata, "\n\n")
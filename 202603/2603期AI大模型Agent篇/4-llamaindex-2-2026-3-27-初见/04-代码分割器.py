# pip install tree_sitter
# pip install tree_sitter_language_pack
from llama_index.core.node_parser import CodeSplitter
from llama_index.core import SimpleDirectoryReader

# 读取文件
documents = SimpleDirectoryReader(input_files=['./03-MarkdownElement解析器.py']).load_data()
# 初始化代码分割器
splitter = CodeSplitter(
    language="python",
    chunk_lines=50,  # 每块行数
    chunk_lines_overlap=10,  # 重叠的数量
    max_chars=300,  # 块最大的数量
)
# 将文档转换成节点
nodes = splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(f"Type: {node.metadata}\nText: {node.text}\n{'='*50}")
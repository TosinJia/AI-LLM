from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

# 读取文件
md_docs = FlatReader().load_data(Path("/home/tosinjia/LLM/files/test.md"))
parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(md_docs)
print(nodes)
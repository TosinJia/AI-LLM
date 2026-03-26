from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope
from llama_index.core.ingestion import IngestionPipeline  # 创建摄取管道
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()
model = "qwen3-max"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

# LlamaIndex默认使用的大模型被替换为百炼
Settings.llm = DashScope(model_name=model, api_key=api_key, api_base=api_base_url, is_chat_model=True)
# 加载本地的嵌入模型
Settings.embed_model = HuggingFaceEmbedding(model_name=r"/home/tosinjia/LLM/Local_model/BAAI/bge-large-zh-v1___5")

documents = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/小说.txt"]).load_data()

# 分割文本设置-按照token数量进行分割，，，，，token被真正命名了->“词元”
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)
# 为每一个节点生成问题-默认的提示词是英文，手动添加提示词
question_prompt_template = """
以下是参考内容：
{context_str}

请根据上述上下文信息，生成 {num_questions} 个该内容能够具体回答的问题，这些问题的答案最好是该内容独有的，不容易在其他地方找到。

你也可以参考上下文中可能提供的更高层次的总结信息，结合这些总结，尽可能生成更优质、更具有针对性的问题。请用中文输出！
"""
# 进行标题的提取，不是给每一个节点生成一个标题，而是根据指定的nodes数量去给所以的系欸但生成统一的标题
title_extractor = TitleExtractor(nodes=5, node_template="请为以下文档生成一个简洁的标题: {context_str}", num_workers=5)
# 进行问题的提取
qa_extractor = QuestionsAnsweredExtractor(questions=3, prompt_template=question_prompt_template, num_workers=5)

async def main():
    # 获取节点  截取前三个节点进行测试
    nodes = text_splitter.get_nodes_from_documents(documents)[:3]
    # 异步等待结果, 根据所有的节点提取的标题生成一个整体标题，get_title_candidates()可以给所有node生成标题
    titles = await title_extractor.aextract(nodes)
    qas = await qa_extractor.aextract(nodes)
    # torch
    # 手动生成的元数据，需要手动的塞回到节点中。将生成的标题和问题回填到节点中
    for node, t, q in zip(nodes, titles, qas):
        node.metadata.update(t)  # 把 document_title 加入 metadata
        node.metadata.update(q)  # 把 questions 加入 metadata
    # 输出内容
    for node in nodes:
        print(node.metadata)

    # 官方建议使用摄取管道进行元数据提取
    # 将原始数据转换为可用于查询的结构化格式
    # pipeline = IngestionPipeline(
    #     transformations=[text_splitter, title_extractor, qa_extractor]
    # )
    # # 开始执行将原始数据转换为可索引的文档格式
    # nodes = pipeline.run(
    #     documents=documents,
    #     in_place=True,
    #     show_progress=True,
    # )
    # print(nodes)


if __name__ == '__main__':
    asyncio.run(main())
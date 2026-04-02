from llama_index.core import VectorStoreIndex, Document
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from 加载模型 import get_llm

# 加载大模型和嵌入模型
llm, embed_model = get_llm()

# 定义自定义的提示模板
qa_prompt_tmpl = PromptTemplate(
    """你是一个专业的问答助手，请根据以下提供的多个参考信息，整合出一个准确、简洁且清晰的答案：

    参考信息如下：
    ---------------------
    {context_str}
    ---------------------

    请根据上述信息回答用户提出的问题。如果参考信息中没有明确提到，请明确说明“在提供的信息中没有找到相关答案”，不要编造内容。

    用户问题: {query_str}

    你的回答："""
)

summary_prompt_template = PromptTemplate("""你是一名专业的内容总结助手，请根据以下信息生成简洁、准确的摘要。

上下文内容：
---------------------
{context_str}
---------------------

请将上述内容总结为关键要点，并保留其中的重要事实信息。
""")

documents = [
    Document(
        text="最初我们的会员制度只有两个等级：普通会员和高级会员。"
    ),
    Document(
        text="随后我们引入了一个新的等级——白金会员，介于高级与钻石之间。"
    ),
    Document(
        text="最近更新：我们取消了高级会员，所有高级用户将自动升级为白金会员。"
    )
]

# 2. 构建索引和向量检索器
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)

# 3. 配置响应合成器
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT,
    streaming=True,
    # 如果想使用自定义的提示模板，
    text_qa_template=qa_prompt_tmpl,
    summary_template=summary_prompt_template,  # 只有"tree_summarize"模式才需要摘要提示词
)
# 手动测试
response = synthesizer.synthesize(query="请总结会员等级制度的演变过程。",
                                  nodes=retriever.retrieve("请总结会员等级制度的演变过程。"))
print(response)
# 4. 配置查询引擎
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    response_synthesizer=synthesizer
)
response = query_engine.query("请总结会员等级制度的演变过程。")
print(response)
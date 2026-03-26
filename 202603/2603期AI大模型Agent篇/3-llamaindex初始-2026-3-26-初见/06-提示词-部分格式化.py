from llama_index.core.prompts import RichPromptTemplate

qa_prompt_tmpl_str = """\
上下文信息如下。
---------------------
{{ context_str }}
---------------------
给定上下文信息而不是先验知识，回答查询。
请以 {{ tone_name }} 的格式写出答案
查询: {{ query_str }}
答案: 
"""
# 可以将目前已有的变量内容先填充到提示词模板中，没有的变量可以稍后在进行填充（和langchain的区别就是可以先进行一部分的变量填充）
prompt_tmpl = RichPromptTemplate(qa_prompt_tmpl_str)
#  partial_format是专门去做某一个或者多个占位符的填充
partial_prompt_tmpl = prompt_tmpl.partial_format(tone_name="莎士比亚")
print(partial_prompt_tmpl)
print("*" * 80)
# format就必须所以的提示词中的占位符都填充好
fmt_prompt = partial_prompt_tmpl.format(
    context_str="在这项工作中，我们开发并发布了 Llama 2，这是一组经过预训练和微调的大型语言模型 (LLM)，其规模从 70 亿到 700 亿个参数不等",
    query_str="llama 2 有多少个参数", )
print(fmt_prompt)
print("*" * 80)
# 格式化为聊天消息列表
fmt_prompt = partial_prompt_tmpl.format_messages(
    context_str="在这项工作中，我们开发并发布了 Llama 2，这是一组经过预训练和微调的大型语言模型 (LLM)，其规模从 70 亿到 700 亿个参数",
    query_str="llama 2 有多少个参数",
)
print(fmt_prompt)
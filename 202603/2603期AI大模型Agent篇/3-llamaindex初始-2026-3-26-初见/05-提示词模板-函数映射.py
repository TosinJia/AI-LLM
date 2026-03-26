from llama_index.core.prompts import RichPromptTemplate
import re

qa_prompt_tmpl_str = """
上下文信息如下。
--------------------- 
{{ context_str }} 
---------------------
给定上下文信息而不是先验知识，回答查询。
查询：{{ query_str }}
答案：
"""


def hide_sensitive_info(text):
    """隐藏文本中的敏感信息"""
    # 隐藏姓名（假设格式为 "姓名：XXX"）
    text = re.sub(r'姓名：[^\n\r]+', '姓名：[已隐藏]', text)

    # 隐藏身份证号码（15位或18位数字）
    text = re.sub(r'身份证：\d{15}(\d{2}[0-9Xx])?', '身份证：[已隐藏]', text)

    # 也可以隐藏其他格式的身份证
    text = re.sub(r'身份证[：:]\s*\d+[0-9Xx]*', '身份证：[已隐藏]', text)

    return text


def format_context_fn(**kwargs):
    # 用项目符号格式化上下文
    context_str = kwargs["context_str"]
    # 隐藏敏感信息
    context_str = hide_sensitive_info(context_str)

    # 用项目符号格式化上下文
    context_list = context_str.split("\n\n")
    context_list = [c.strip() for c in context_list if c.strip()]  # 修复原代码的bug

    fmtted_context = "\n\n".join([f"- {c}" for c in context_list])
    return fmtted_context


prompt_tmpl = RichPromptTemplate(
    qa_prompt_tmpl_str, function_mappings={"context_str": format_context_fn}
)

context_str = """\
姓名：初见

身份证：123456798123456

这项工作中，我们开发并发布了 Llama 2，这是一组经过预训练和微调的大型语言模型 (LLM)，其规模从 70 亿到 700 亿个参数不等。

我们经过微调的 LLM 称为 Llama 2-Chat，针对对话用例进行了优化。

在我们测试的大多数基准测试中，我们的模型都优于开源聊天模型，并且根据我们对有用性和安全性的人工评估，它们可能是闭源模型的合适替代品。
"""
# 用户问题改写
fmt_prompt = prompt_tmpl.format(
    context_str=context_str, query_str="llama2有多少参数？"
)
print(fmt_prompt)
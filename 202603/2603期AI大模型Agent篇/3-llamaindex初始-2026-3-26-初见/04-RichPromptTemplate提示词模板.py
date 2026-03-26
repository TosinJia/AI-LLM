# from llama_index.core.prompts import RichPromptTemplate
#
# context_str = """
# 【企业基础信息】
# 公司全称：杭州深度求索人工智能基础技术研究有限公司
# 成立时间：2023年7月17日（工商注册日期）
# 核心技术：数据蒸馏技术（用于优化大语言模型训练数据）
# 股东背景：由幻方量化（知名私募机构）孵化
# 注册地址：浙江省杭州市拱墅区环城北路169号汇金国际大厦西1幢1201室
# 法定代表人：裴湉
# 核心业务：大语言模型（LLM）研发、技术服务、软件开发、技术转让
#
# 【补充说明】
# 1. 公司成立后6个月内完成首轮融资，估值超10亿人民币；
# 2. 数据蒸馏技术为公司核心专利，已应用于多款自研大模型。
# """
#
#
# question = 'DeepSeek公司的工商注册成立年份是哪一年？请仅给出数字答案'
#
# template = RichPromptTemplate(
#     """
# # 任务说明
# 你是企业信息问答助手，需严格基于提供的上下文信息回答问题，不得编造内容。
#
# # 上下文信息
# ---------------------
# {{ context_str }}
# ---------------------
#
# # 待回答问题
# {{ query_str }}
#
# # 回答要求
# 1. 严格按照问题要求的格式回答；
# 2. 仅使用上下文里的信息，不添加额外解释；
# 3. 若上下文无相关信息，回复："未查询到相关信息"。
#     """
# )
#
# # 格式化为纯字符串（适用于非聊天型大模型/API）
# prompt_str = template.format(context_str=context_str, query_str=question)
# print("=== 格式化后的纯字符串Prompt ===")
# print(prompt_str)
#
# # 格式化聊天消息列表（适用于ChatGPT/文心一言等聊天型大模型）
# messages = template.format_messages(context_str=context_str, query_str=question)
# print("\n=== 格式化后的聊天消息列表 ===")
# # 优化点4：美化输出格式，清晰展示消息结构
# for msg in messages:
#     print(f"角色： {msg.role}")
#     print(f"内容：{msg.content}\n")



from llama_index.core.prompts import RichPromptTemplate

template = RichPromptTemplate(
    """
{% chat role="system" %}
你是多模态文档分析助手，需要结合图片内容和文本描述回答用户问题。
核心规则：
1. 优先基于图片对应的文本描述分析信息；
2. 若图片路径包含"合同"关键词，重点关注文本中的金额、日期信息；
3. 回答需简洁明了，分点说明关键信息。
{% endchat %}

{% chat role="user" %}
请分析以下图片和对应的文本信息，总结每份文件的核心内容：
{% for img_path, text_content in multi_modal_data %}
- 文件路径：{{ img_path }}
- 文本描述：{{ text_content }}
- 图片内容：{{ img_path | image }}  # 标记为图片类型，供多模态模型解析
{% endfor %}

我的问题：这些文件中是否包含合同类文件？如果有，核心信息是什么？
{% endchat %}
"""
)

messages = template.format_messages(
    multi_modal_data=[
        ("contract_202403.png", "2024年3月采购合同：甲方为XX科技，乙方为YY制造，合同金额50万元，有效期1年"),
        ("contract_202403.png", "2024年Q1销售报告：总销售额1200万元，同比增长15%，覆盖3个省份"),
        ("invoice_202404.png", "2024年4月发票：金额8.5万元，对应项目为服务器采购")
    ]
)

print("=== 格式化后的多模态聊天消息列表 ===")
for idx, msg in enumerate(messages):
    print(f"\n【消息{idx+1}】")
    print(f"角色：{msg.role}")
    print(f"内容：{msg.content.strip()}")
import json
import random
import re
from typing import Literal, TypedDict, Annotated
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. 配置与模拟数据 ---

# 模拟数据库：菜单与价格
MENU_DB = {
    "汉堡": 25.0,
    "芝士汉堡": 28.0,
    "薯条": 12.0,
    "可乐": 8.0,
    "雪碧": 8.0,
    "炸鸡": 35.0
}

# 初始化大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model='kimi-k2.5',
    temperature=0.5
)


# --- 2. 工具函数 ---

def parse_json_output(text: str) -> dict:
    """
    鲁棒的JSON解析器：自动清洗Markdown标记，处理常见的LLM格式问题
    """
    try:
        # 移除 ```json 和 ``` 标记
        cleaned_text = re.sub(r'```json\s*', '', text)
        cleaned_text = re.sub(r'```', '', cleaned_text)
        return json.loads(cleaned_text.strip())
    except Exception as e:
        print(f"JSON解析警告: {e}, 原始内容: {text[:50]}...")
        return {}


# 封装调用模型的方法
def call_llm(system_prompt: str, user_message: str, temperature: float = 0.7) -> str:
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        llm.temperature = temperature
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"


# 定义state
class OrderState(TypedDict):
    customer_name: str  # 客户姓名
    # 使用 add_messages 保留对话历史
    messages: Annotated[list[BaseMessage], add_messages]
    order_items: list  # 结构: [{"name": "汉堡", "price": 25.0}]
    raw_order_text: str  # 用户原始输入
    total_amount: float  # 总金额
    payment_status: str  # 支付结果
    delivery_address: str  # 配送地址
    order_status: str  # 订单状态
    llm_analysis: dict  # 大模型返回的分析结果


# 定义节点

def order_receiver(state: OrderState) -> Command[Literal["payment_processor", "order_validator"]]:
    # 1.获取用户的原始输入
    raw_order_text = state["raw_order_text"]
    customer_name = state["customer_name"]  # 用户名称
    print(f"正在处理{customer_name}的需求:{raw_order_text}")

    # 如果已经有 items（可能是重试或手动输入的），跳过解析
    if state.get("order_items") and not raw_order_text:
        total = sum(item['price'] for item in state['order_items'])
        return Command(goto="payment_processor", update={"total_amount": total})

    # 2.调用模型进行意图拆解
    system_prompt = f"""你是一个餐厅订单解析员。
    当前菜单包括：{', '.join(MENU_DB.keys())}。
    请分析用户输入，提取菜品。如果用户说的菜不在菜单里，请标记为未知。

    返回严格的JSON格式：
    {{
        "items_identified": ["菜名1", "菜名2"],
        "unknown_items": ["不在菜单的词"],
        "intent_clarity": "high/medium/low",
        "missing_info": "缺少的关键信息(如地址等)"
    }}"""
    llm_response = call_llm(system_prompt, raw_order_text)
    analysis = parse_json_output(llm_response)

    print(f"模型分析后得到的内容:{analysis}")
    # 3.将识别出来的菜名映射成订单
    identified_names = analysis.get("items_identified")

    new_order_items = []  # 当前用户所需要买的商品-价格

    for name in identified_names:
        # 简单的模糊匹配逻辑（实际项目中可用向量搜索）
        if name in MENU_DB:
            new_order_items.append({"name": name, "price": MENU_DB[name]})
        else:
            # 尝试查找最接近的菜单项（简化版）
            for menu_item in MENU_DB:
                if menu_item in name or name in menu_item:
                    new_order_items.append({"name": menu_item, "price": MENU_DB[menu_item]})
                    break

    # 计算总金额
    total = sum(item['price'] for item in new_order_items)

    # 检查是否有未知商品   大于0就代表有不属于我们的商品
    has_unknown_items = len(analysis.get("unknown_items")) > 0

    # 检查是否完全没识别出商品
    no_valid_items = len(new_order_items) == 0

    # 检查置信度
    is_unclear = analysis.get("intent_clarity") == "low"

    # 只要满足以上三种的任意一种条件就需要去往验证节点
    if has_unknown_items or no_valid_items or is_unclear:
        # 构建提示消息
        if has_unknown_items:
            msg_content = f"系统检测到未知商品：{', '.join(analysis['unknown_items'])}，需要确认。"
        elif no_valid_items:
            msg_content = "没能识别出具体菜品。"
        else:
            msg_content = "订单信息不明确。"

        return Command(
            goto="order_validator",
            update={
                "order_status": "解析存疑",
                "llm_analysis": analysis,
                "messages": [AIMessage(content=msg_content)]
            }
        )
    # 完全准备就绪后就去往支付节点
    return Command(
        goto="payment_processor",
        update={
            "order_items": new_order_items,
            "total_amount": total,
            "order_status": "待支付",
            "llm_analysis": analysis,
            "messages": [AIMessage(content=f"已生成订单：{[i['name'] for i in new_order_items]}，总价：{total}元")]
        }
    )


def order_validator(state: OrderState) -> Command[Literal["payment_processor", END]]:
    """
    订单验证节点：处理异常或信息缺失
    """
    print("[验证] 正在检查订单完整性...")
    analysis = state.get("llm_analysis", {})
    order_items = state.get("order_items", [])

    # 简单的验证逻辑：如果有未知商品或金额为0
    unknowns = analysis.get("unknown_items", [])

    if unknowns:
        error_msg = f"抱歉，我们暂时不提供：{', '.join(unknowns)}。请重新下单。"
        return Command(
            goto=END,
            update={
                "order_status": "验证失败",
                "messages": [AIMessage(content=error_msg)]
            }
        )

    if not order_items:
        return Command(
            goto=END,
            update={
                "order_status": "空订单",
                "messages": [AIMessage(content="未能识别任何有效菜品，流程结束。")]
            }
        )

    # 如果验证通过（比如虽然有小问题但可忽略）
    return Command(
        goto="payment_processor",
        update={"messages": [AIMessage(content="经二次验证，订单有效。")]}
    )


def payment_processor(state: OrderState) -> Command[Literal["delivery_scheduler", END]]:
    """
    支付节点
    """
    amount = state.get("total_amount", 0)
    print(f"[支付] 正在处理金额: {amount}元")

    # 模拟支付
    if amount > 1000:  # 假设大额风控
        return Command(
            goto=END,
            update={
                "payment_status": "拒绝",
                "messages": [AIMessage(content="金额过大，支付被系统拒绝。")]
            }
        )

    return Command(
        goto="delivery_scheduler",
        update={
            "payment_status": "成功",
            "messages": [AIMessage(content="支付成功！")]
        }
    )


def delivery_scheduler(state: OrderState) -> Command[Literal[END]]:
    """
    配送节点
    """
    address = state.get("delivery_address", "未填写地址")
    items = [i['name'] for i in state.get('order_items', [])]

    # 使用 LLM 生成最终通知
    prompt = f"""为客户生成一条外卖配送通知。
    菜品：{', '.join(items)}
    地址：{address}
    风格：热情、期待。"""

    msg = call_llm(prompt, "生成通知", temperature=0.7)
    print(f"[配送] {msg}")

    return Command(
        goto=END,
        update={
            "order_status": "配送中",
            "messages": [AIMessage(content=msg)]
        }
    )


# --- 5. 构建图 ---

builder = StateGraph(OrderState)
builder.add_node("order_receiver", order_receiver)
builder.add_node("order_validator", order_validator)
builder.add_node("payment_processor", payment_processor)
builder.add_node("delivery_scheduler", delivery_scheduler)

builder.add_edge(START, "order_receiver")

# 没有固定对应的边

graph = builder.compile()


# --- 6. 测试运行 ---

print("--- 测试用例 1: 纯自然语言输入 (自动解析) ---")
test_input = {
    "customer_name": "李四",
    "raw_order_text": "你好，我想要两个汉堡和一杯可乐，送到科技园A栋。",
    "delivery_address": "科技园A栋",  # 实际场景中这个也应该由LLM提取
    "messages": [],
    "order_items": []  # 故意留空，测试自动解析
}

try:
    final_state = graph.invoke(test_input)
    print("\n流程结束")
    print(f"最终状态: {final_state['order_status']}")
    print(f"最终金额: {final_state['total_amount']}")
    print(f"订单内容: {final_state['order_items']}")
except Exception as e:
    print(f"运行出错: {e}")

print("\n--- 测试用例 2: 包含不在菜单的商品 ---")
test_input_2 = {
    "customer_name": "王五",
    "raw_order_text": "来一份披萨和一杯可乐。",  # 披萨不在菜单里
    "messages": [],
    "order_items": []
}

try:
    final_state_2 = graph.invoke(test_input_2)
    print(f"\n最终状态: {final_state_2['order_status']}")
    # 应该打印出验证失败的消息
    print(f"系统回复: {final_state_2['messages'][-1].content}")
except Exception as e:
    print(f"运行出错: {e}")
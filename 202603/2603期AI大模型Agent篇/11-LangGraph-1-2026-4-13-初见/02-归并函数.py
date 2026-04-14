# from typing import TypedDict, Annotated
# from langgraph.graph import StateGraph
# from operator import add  # operator  python封装的add。delete
#
# class MyState(TypedDict):
#     names: Annotated[list[str], add]   # 回自动追加到列表的末尾
#
#
# # 多轮对话，没有当前的归并函数，你每次对话都会覆盖之前的内容。
#
# # 定义节点
# def A_node(state: MyState):
#     return {"names": ["张三"]}
#
# def B_node(state: MyState):
#     return {"names": ["李四"]}
#
# state_graph = StateGraph(state_schema=MyState)
#
# state_graph.add_node("A", A_node)
# state_graph.add_node("B", B_node)
# # 定义入口
# state_graph.set_entry_point("A")
# state_graph.set_entry_point("B")
#
# graph = state_graph.compile()
# result = graph.invoke({})
# print(result)
#

from typing import TypedDict
from langgraph.graph import StateGraph
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
import operator

# 如果后续做的对话类型的智能体，直接使用  MessagesState
# 定义状态结构   如果定义的是list[dict]，会覆盖之前的数据
class ChatState(MessagesState):
    messages: Annotated[list, add_messages]  # 每条消息是 {role, content}，会自动追加到列表末尾


# 节点函数：添加用户问题
def user_input_node(state: ChatState) -> dict:
    user_msg = {"role": "user", "content": "什么是LangGraph？"}
    return {"messages": [user_msg]}


# 节点函数：添加助手回复
def assistant_node(state: ChatState) -> dict:
    reply = {"role": "assistant", "content": "LangGraph 是一个有状态的图编排框架。"}
    return {"messages": [reply]}


# 构建状态图
builder = StateGraph(state_schema=ChatState)
builder.add_node("user_input", user_input_node)
builder.add_node("assistant_reply", assistant_node)
builder.set_entry_point("user_input")
builder.add_edge("user_input", "assistant_reply")

graph = builder.compile()

result = graph.invoke({"messages": []})
print(result["messages"])

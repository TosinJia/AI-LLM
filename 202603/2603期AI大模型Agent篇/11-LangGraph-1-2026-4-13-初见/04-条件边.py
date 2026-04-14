from typing import TypedDict
from langgraph.graph import StateGraph, END


class MyState(TypedDict):
    type: str
    result: str


def judge_node(state: MyState):
    """节点函数：可以做一些预处理"""
    return state  # 保持状态不变，只是路由


def route_condition(state: MyState):
    """条件函数：只负责路由决策"""
    # 根据自身的业务逻辑去制定流程的走向
    # 100业务
    if state["type"] == "a":
        return "A"
    elif state["type"] == "b":
        return "b"
    else:
        return "default"


def node_a(state):
    return {"result": "走了 A 分支"}


def node_b(state):
    return {"result": "走了 B 分支"}


def node_default(state):
    return {"result": "走了默认分支"}


# 构建图
graph = StateGraph(state_schema=MyState)
# 定义节点
graph.add_node("judge_node", judge_node)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_node("default", node_default)
# 定义开始节点
graph.set_entry_point("judge_node")

# 使用不同的函数作为条件函数     条件边的前提：不是复杂的业务，条件简单就只有几条分支
# 参数1：哪个节点结束后触发当前条件边执行
# 参数2：条件要根据某个函数去决定对应的流程的跳转
# 参数3：根据函数返回的内容去进行节点的跳转  {"函数返回值"："节点名称"}
graph.add_conditional_edges("judge_node", route_condition, {
    "A": "a",
    "b": "b",
    "default": "default"
})

# 添加结束边
graph.add_edge("a", END)
graph.add_edge("b", END)
graph.add_edge("default", END)

app = graph.compile()

# 测试
print("测试 A:", app.invoke({"type": "a", "result": ""}))
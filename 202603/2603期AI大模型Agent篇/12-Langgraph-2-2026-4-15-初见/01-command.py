from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal, TypedDict

# command：做动态工作流。
# 初始化状态
class MyState(TypedDict):
    type: str
    text: str
    result: str


# 创建节点
def node_a(state: MyState):
    return {"result": "目前走了节点A"}


def node_b(state: MyState):
    return {"result": "目前走了节点B"}


def node_default(state: MyState):
    return {"result": "目前走了默认节点"}


# 添加条件节点   Literal给当前返回值设定一个限制，只能返回Literal中列举的节点
def judge_node(state: MyState) -> Command[Literal["a", "b", "default"]]:
    if state["type"] == "a":
        # 如果你需要在路由过程中需要更新状态，才需要使用command
        return Command(update={"text": "走了A节点"}, goto="a")
    elif state["type"] == "b":
        return Command(update={"text": "走了B节点"}, goto="b")
    else:
        return Command(update={"text": "走了默认节点"}, goto="default")


# 构建图
state_graph = StateGraph(state_schema=MyState)
state_graph.add_node("a", node_a)
state_graph.add_node("b", node_b)
state_graph.add_node("default", node_default)
state_graph.add_node("judge_node", judge_node)

# 设置入口和边
state_graph.add_edge(START, "judge_node")
state_graph.add_edge("a", END)
state_graph.add_edge("b", END)
state_graph.add_edge("default", END)

graph = state_graph.compile()

print(graph.invoke({"type": "a"}))

from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Send

# 使用send 进行条件跳转（就类似于条件边）


class MyState(TypedDict):
    type: bool
    name: str


def node_a(state):
    if state["type"]:
        return Send("b", {"name": "张三"})
    else:
        return Send("c", {"name": "李四"})


def node_b(state):
    pass


def node_c(state):
    pass
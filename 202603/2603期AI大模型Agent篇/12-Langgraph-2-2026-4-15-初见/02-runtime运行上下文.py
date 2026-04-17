# from langgraph.graph import StateGraph
# from langgraph.runtime import Runtime
# from typing import TypedDict
# import dataclasses
#
# # 定义状态结构
# class MyState(TypedDict):
#     question: str
#     answer: str
#
# # 定义配置结构
# @dataclasses.dataclass(frozen=True)   # 建议加上@dataclasses.dataclass(frozen=True) 为了防止在图中去修改属性
# class MyContext(TypedDict):
#     language: str  # 配置中包含语言选项，比如 "en" 或 "zh"
#
# # 节点函数可以访问 runtime 参数  runtime 可以访问上下文和内存存储
# def step1(state: MyState, runtime: Runtime[MyContext]):
#     if runtime.context["language"] == "zh":
#         answer = "你好！"
#     else:
#         answer = "Hello!"
#     return {"answer": answer}
#
# # 构建图..
# graph = StateGraph(state_schema=MyState, context_schema=MyContext)
# graph.add_node("step1", step1)
# graph.set_entry_point("step1")
#
# # 编译
# app = graph.compile()
#
# # 执行时传入 config 参数（区分于 state）
# result = app.invoke({"question": "Hi"}, context={"language": "zh"})
# print(result)  # => {"question": "Hi", "answer": "你好！"}


from langgraph.graph import MessagesState
from langgraph.runtime import Runtime
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict


class MyContext(TypedDict):
    model: str


MODELS = {
    "anthropic": "anthropic:claude-3-5-haiku-latest",
    "openai": "openai:gpt-4.1-mini",
}


def call_model(state: MessagesState, runtime: Runtime[MyContext]):
    model = ""
    if runtime.context:
        model = runtime.context["model"]
        model = MODELS[model]
    return {"messages": {"role": "assistant", "content": model}}


builder = StateGraph(MessagesState, context_schema=MyContext)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

# 问题
input_message = {"role": "user", "content": "hi"}
# 没有配置时，使用默认值（Anthropic）
response_1 = graph.invoke({"messages": [input_message]})
# 切换成 openai
context = {"model": "openai"}
response_2 = graph.invoke({"messages": [input_message]}, context=context)

print(response_1)
print(response_2)
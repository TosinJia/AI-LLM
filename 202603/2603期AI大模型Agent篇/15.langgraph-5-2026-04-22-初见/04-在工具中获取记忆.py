import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

# 1. 初始化模型
llm = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_provider="openai",
    model='MiniMax-M2.1'
)


# 2. 定义状态 (State)
class CustomState(MessagesState):
    user_name: str


# 3. 定义工具   ToolRuntime就和Runtime是一样的
@tool
def get_user_name(runtime: ToolRuntime) -> str:
    """从状态中检索当前用户名。"""
    # 在 LangGraph 中，可以通过 InjectedState 注入整个状态
    return runtime.state.get("user_name", "未知用户")


@tool
def update_user_name(new_name: str, runtime: ToolRuntime) -> Command:
    """更新短期记忆中的用户名。"""
    print(f"--- 触发更新用户名工具: {new_name} ---")
    # Command 会在工具执行完后直接作用于 Graph 的 State   重点！！！！！
    return Command(
        update={
            "user_name": new_name,
            "messages": [
                ToolMessage(
                    content=f"姓名已经更改成： {new_name}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


tools = [get_user_name, update_user_name]
llm_with_tools = llm.bind_tools(tools)


# 4. 定义节点函数
def call_model(state: CustomState):
    """模型决策节点"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 5. 构建图 (Workflow)
workflow = StateGraph(CustomState)

# 添加处理节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

# 设置连线
workflow.add_edge(START, "agent")

# 动态决定：是去执行工具还是直接结束
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # 内置函数：判断消息中是否有 tool_calls
    {"tools": "tools", "__end__": END},
)

# 工具执行完后回到 agent，让模型根据工具结果说话
workflow.add_edge("tools", "agent")

# 6. 编译并运行
checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# --- 测试运行 ---
config = {"configurable": {"thread_id": "1"}}

print("\n--- 第一次对话 ---")
input_1 = {"messages": [{"role": "user", "content": "我的名字是初见"}]}
for event in app.stream(input_1, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

print("\n--- 查看当前 State 中的 user_name ---")
print(f"State Name: {app.get_state(config).values.get('user_name')}")

print("\n--- 第二次对话 ---")
input_2 = {"messages": [{"role": "user", "content": "我的名字是什么?"}]}
for event in app.stream(input_2, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
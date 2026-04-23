from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch # pip install langchain-tavily
from langchain.tools import tool
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='MiniMax-M2.1')


@tool
def tavily_search_tool(query: str) -> str:
    """这是一个搜索工具"""
    tool_instance = TavilySearch()
    return tool_instance.run(query)


# 执行工具的节点      function call   langgraph提供的执行工具的模块（调用工具节点）
tool_node = ToolNode([tavily_search_tool])
# 绑定工具到模型
model_with_tools = llm.bind_tools([tavily_search_tool])   # 模型只能做决策


def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(MessagesState)

# 定义节点和边
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")  # 模型因为绑定了工具，决定是否要使用工具，返回：tool message
builder.add_conditional_edges("call_model", should_continue, ["tools", END])
builder.add_edge("tools", "call_model")

graph = builder.compile()

print(graph.invoke({"messages": [{"role": "user", "content": "上海的天气?"}]}))

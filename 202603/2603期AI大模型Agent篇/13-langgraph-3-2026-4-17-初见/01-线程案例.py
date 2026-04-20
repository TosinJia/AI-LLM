import asyncio

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.chat_models import init_chat_model
# 导入内存检查点对象   短期记忆
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 初始化模型
llm = init_chat_model(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url=os.getenv('DASHSCOPE_BASE_URL'),
    model_provider="openai",
    model="MiniMax-M2.1"  # 在做智能体的时候，最好去选择模型效果好一点
)


# claude模型   opus4.7   需要实名


# 定义相关的node
async def process_message(state: MessagesState):
    response = await llm.ainvoke(state["messages"])
    return {"messages": response}


async def optimize_message(state: MessagesState):
    messages = (state["messages"] +
                [{"role": "system", "content": "请用幽默的形式回复用户"}])
    response = await llm.ainvoke(messages)
    return {"messages": response}


async def main():
    # 构建图
    builder = StateGraph(state_schema=MessagesState)
    builder.add_node(process_message)
    builder.add_node(optimize_message)
    builder.add_edge(START, "process_message")
    builder.add_edge("process_message", "optimize_message")
    builder.add_edge("optimize_message", END)

    # 初始化检查点
    checkpoint = InMemorySaver()   #  重要！！！！！！！！
    # 必须定义thread_id去区分每一次会话   12345  id是随机且不能重复的值   #  重要！！！！！！！！
    config = {"configurable": {"thread_id": "12345"}}
    graph = builder.compile(checkpointer=checkpoint)

    # 第一次对话
    input_message = {"messages": [HumanMessage(content="你好，我叫初见")]}
    result = await graph.ainvoke(input_message, config=config)
    print(result)

    # 第二次对话     能不能知道我是谁？
    input_message = {"messages": [HumanMessage(content="你好，请问我叫什么名字？")]}
    result = await graph.ainvoke(input_message, config=config)
    print(result)


if __name__ == '__main__':
    asyncio.run(main())

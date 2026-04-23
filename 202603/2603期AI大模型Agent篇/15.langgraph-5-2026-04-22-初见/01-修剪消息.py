from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='MiniMax-M2.1')


def call_llm(state: MessagesState):
    print("修剪前的消息：", state["messages"])
    # 修剪消息是在调用llm之前
    messages = trim_messages(
        state["messages"],
        strategy="last",  # 修剪策略：last从末尾，first从开头，middle从中间(last是保留最近的消息，常用)
        token_counter=count_tokens_approximately,  # 用来估算对话token数量
        max_tokens=500,  # 修剪后的消息总tokens数量不能超过100
        start_on="human",  # 修剪是从哪一种对话消息开始保留
        end_on=("ai", "tools", "human"),  # 修剪是从哪一种对话消息结尾
    )
    print("修剪后的消息：", messages)
    response = llm.invoke(messages)
    return {"messages": response}

checkpointer = InMemorySaver()  # 创建检查点
builder = StateGraph(MessagesState)  # 构建图
builder.add_node(call_llm)
builder.add_edge(START, "call_llm")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "我的名字叫初见"}, config)
graph.invoke({"messages": "帮我家的猫写一首诗"}, config)
graph.invoke({"messages": "现在对狗做一样的事情"}, config)
final_response = graph.invoke({"messages": "我的名字叫什么?"}, config)

print("最终消息：")
print(final_response["messages"])

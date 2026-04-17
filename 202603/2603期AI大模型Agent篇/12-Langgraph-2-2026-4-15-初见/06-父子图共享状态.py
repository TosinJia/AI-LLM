import os
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
# 初始化模型
llm = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="MiniMax-M2.1",
    model_provider="openai"
)


# 创建子图
def sub_plot(state: MessagesState):
    # 总结父图中模型生成的文档
    content = state["messages"][-1].content  # 获取最新一条消息
    summary_prompt = f"请用一句话总结以下内容：\n\n {content}"
    response = llm.invoke(summary_prompt)
    return {"messages": response}


# 初始化子图
summary_subgraph = (
    StateGraph(state_schema=MessagesState)
    .add_node("subplot", sub_plot)
    .add_edge(START, "subplot")
    .compile()
)


# 创建父图
def llm_answer_node(state: MessagesState):
    answer = llm.invoke(state["messages"])
    print("父图的中使用模型回复用户问题", answer)
    return {"messages": answer}


# 初始化父图
parent_graph = (
    StateGraph(MessagesState)
    .add_node("llm_answer", llm_answer_node)
    .add_node("summarize_subgraph", summary_subgraph)  # 将子图当作父图的一个节点
    .add_edge(START, "llm_answer")
    .add_edge("llm_answer", "summarize_subgraph")
    .add_edge("summarize_subgraph", END)
    .compile()
)

result = parent_graph.invoke(
    {"messages": [HumanMessage(content="langgraph是什么？")]}
)
print(result)
from IPython.display import Image, display

display(
    Image(
        parent_graph.get_graph().draw_mermaid_png(output_file_path="./父子图共享状态.png")
    )
)

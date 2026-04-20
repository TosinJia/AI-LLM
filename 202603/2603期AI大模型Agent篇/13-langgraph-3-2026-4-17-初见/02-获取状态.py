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


# 创建子图
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
    # print("父图的中使用模型回复用户问题", answer)
    return {"messages": answer}


checkpointer = InMemorySaver()
# 初始化父图
parent_graph = (
    StateGraph(MessagesState)
    .add_node("llm_answer", llm_answer_node)
    .add_node("summarize_subgraph", summary_subgraph)  # 将子图当作父图的一个节点
    .add_edge(START, "llm_answer")
    .add_edge("llm_answer", "summarize_subgraph")
    .add_edge("summarize_subgraph", END)
    .compile(checkpointer=checkpointer)
)
# 定义config
config = {"configurable": {"thread_id": "12345"}}
# 测试输入
input_state = {
    "messages": [{"role": "user", "content": "langgraph是什么？请用100字介绍"}],
}
result = parent_graph.invoke(input_state, config=config)
# print(result)

# 获取状态   get_state()   获取之前thread_id已经完成任务的所有保存的内容
# print(parent_graph.get_state(config))

print("================状态历史记录=================")
"""
可以通过调用 获取给定线程的图形执行的完整历史记录graph.get_state_history(config)。
这将返回与配置中提供的线程 ID 关联的对象列表StateSnapshot。
重要的是，检查点将按时间顺序排序，最新的检查点 /StateSnapshot将位于列表中的第一个。

注意：这里采用的是共享状态的子图，可以将子图的内容持久化，如果使用的是不同状态的就需要分别存储
"""

history = list(parent_graph.get_state_history(config))
for idx, snapshot in enumerate(history):
    print(f"Step {idx}:")
    print(f"  Checkpoint ID: {snapshot.config['configurable']['checkpoint_id']}")
    print(f"  Node: {snapshot.metadata.get('source')}")
    print(f"  Messages: {[m.content for m in snapshot.values['messages']]}")
    print("")

print("================重放机制=================")
# 任务执行完成之后，将中间的重要步骤给用户查看，用户就可以决定从某些步骤中重新开始执行
# 注意：必须传递这些内容 thread_id， checkpoint_id
# 通过thread_id定位到某次会话，通过checkpoint_id定位到对应的节点
# 重放是会在原有的基础上生成一个新的检查点分支

# 获取step2当前的checkpoint_id
step2_level_checkpoint = None
if history:
    step2_level_checkpoint = list(history)[1].config['configurable']['checkpoint_id']

# 重新创建一个config内容，其中指定checkpoint_id就能从指定的这个checkpoint_id节点开始重播
config = {"configurable": {"thread_id": "12345", "checkpoint_id": step2_level_checkpoint}}

new_result = parent_graph.invoke(None, config=config)
print("重播后的内容")
print(new_result)

# history = list(parent_graph.get_state_history(config))
# for idx, snapshot in enumerate(history):
#     print(f"Step {idx}:")
#     print(f"  Checkpoint ID: {snapshot.config['configurable']['checkpoint_id']}")
#     print(f"  Node: {snapshot.metadata.get('source')}")
#     print(f"  Messages: {[m.content for m in snapshot.values['messages']]}")
#     print("")
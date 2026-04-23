from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
import os

load_dotenv()
llm = init_chat_model(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_provider="openai",
        model='MiniMax-M2.1',
        temperature=0.7
    )


"""
    interrupt_before AI 已经干完活了（写了代码、写了邮件），你只是想在它造成后果（部署代码、发送邮件）之前检查一下。
    关键用作审核、拦截
"""

def writer_node(state: MessagesState):
    """节点 A: 负责写内容"""
    print("[Writer]: 正在撰写推文...")
    msg = llm.invoke([{"role": "user", "content": "写一条关于学习Python的幽默推文，50字以内"}])
    return {"messages": [msg]}


def publisher_node(state: MessagesState):
    """节点 B: 负责发布 (副作用)"""
    content = state["messages"][-1].content
    print(f"\n[Publisher]: 推文已发布到互联网: \n{content}")
    return {"messages": [AIMessage(content="发布成功")]}


# --- 构建图 ---

builder = StateGraph(MessagesState)
builder.add_node("writer", writer_node)
builder.add_node("publisher", publisher_node)

builder.add_edge(START, "writer")
builder.add_edge("writer", "publisher")
builder.add_edge("publisher", END)

# 核心配置：在进入 publisher 之前必须暂停！
checkpointer = MemorySaver()
# 在进入publisher这个节点之前先进行人工审核
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["publisher"])


# --- 运行逻辑 ---

def run_demo_1():
    print("\n=== 案例 1: interrupt_before (审核模式) ===")
    config = {"configurable": {"thread_id": "tweet_1"}}

    # 1. 启动：它会跑完 writer，然后在 publisher 门口停下
    print(">>> 启动任务...")
    graph.invoke({"messages": []}, config)

    # 2. 检查状态
    snapshot = graph.get_state(config)
    ai_draft = snapshot.values["messages"][-1].content

    print(f"\n[人类审核员]: 看到 AI 写了: {ai_draft}")
    user_input = input("是否批准? (y=批准 / 输入文字=修改并发布): ").strip()

    # 3. 处理决策
    if user_input.lower() == 'y':
        print("批准！放行...")
        # 传入 None 表示继续执行原计划
        graph.invoke(None, config)
    else:
        print("修改中...")
        # 修改状态：用人类的话替换 AI 的话
        new_msg = AIMessage(content=user_input)
        graph.update_state(config, {"messages": [new_msg]}, as_node="writer")

        print("修改完成，放行...")
        graph.invoke(None, config)


"""
    interrupt 函数 AI 正在执行一个复杂的长任务，跑到第 3 步发现缺个 API Key，或者缺个参数，它需要停下来找你要，拿到后继续跑第 4 步。
    关键用作填空
"""
# --- 节点定义 ---
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# 1. 定义 State：我们需要一个字段来在节点间传递 days
class TripState(TypedDict):
    messages: Annotated[list, add_messages]
    days: str  # 新增字段，用来存储天数

# --- 节点 1: 负责开场白 (副作用) ---
def start_node(state: TripState):
    # 这里放你只想执行一次的代码
    print("\n[AI]: 收到，正在为你规划去‘云南’的旅行...")
    # 这个节点不返回 messages，只负责打印和过渡
    return {}

# --- 节点 2: 负责提问 + 生成 ---
def planner_node(state: TripState):
    # 1. 中断逻辑
    # 第一次运行：在这里暂停
    # 恢复运行：直接从这里拿到值，继续往下走，不会回头去跑 start_node
    days = interrupt("请问你打算去几天？")

    print(f"[AI]: 收到，{days}天。正在生成行程...")

    # 2. 生成逻辑
    # (这里为了演示简单，直接返回文本，实际可用 LLM)
    content = f"这是为你生成的云南 {days} 天行程：大理 -> 丽江 -> 香格里拉..."
    return {"messages": [AIMessage(content=content)], "days": days}

# --- 构建图 ---
builder_2 = StateGraph(TripState)

builder_2.add_node("start", start_node)
builder_2.add_node("planner", planner_node)

# 连线： Start -> Planner -> End
builder_2.add_edge(START, "start")
builder_2.add_edge("start", "planner")
builder_2.add_edge("planner", END)

checkpointer_2 = MemorySaver()
graph_2 = builder_2.compile(checkpointer=checkpointer_2)


# --- 运行逻辑 ---

def run_demo_2():
    print("\n=== 案例 2: interrupt 函数 (填空模式) ===")
    config = {"configurable": {"thread_id": "trip_1"}}

    print(">>> 启动任务...")
    # 1. 第一次运行
    for chunk in graph_2.stream({"messages": []}, config, stream_mode="values", version="v2"):
        type = chunk["type"]
        ns = chunk["ns"]
        data = chunk["data"]
        # 目前是V2版本可以在流式输出中进行获取
        interrupts = chunk["interrupts"]
        if interrupts:

            print(f"\n[系统暂停] AI 询问: {interrupts[0].value}")

            # 3. 获取人类回答
            answer = input("回答: ")   # 114行day变量就是用户的输入

            print("恢复执行...")
            # 4. 恢复执行：使用 Command 将答案传回给节点内部的 days 变量   核心点
            res = graph_2.invoke(Command(resume=answer), config)
            print(res)

        # print(f"{chunk}")

    # # 2. 捕获中断
    # snapshot = graph_2.get_state(config)
    #
    # # 检查是否有中断任务
    # if snapshot.tasks and snapshot.tasks[0].interrupts:
    #     # 获取 interrupt("...") 里的问题
    #     question = snapshot.tasks[0].interrupts[0].value
    #     print(f"\n[系统暂停] AI 询问: {question}")
    #
    #     # 3. 获取人类回答
    #     answer = input("回答: ")   # 114行day变量就是用户的输入
    #
    #     print("恢复执行...")
    #     # 4. 恢复执行：使用 Command 将答案传回给节点内部的 days 变量
    #     res = graph_2.invoke(Command(resume=answer), config)
    #     print(res)


if __name__ == "__main__":
    # run_demo_1()
    run_demo_2()
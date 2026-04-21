from typing import Annotated
import operator
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Send
from typing import TypedDict, List

"""---------------------------values、updates、debug模式---------------------------"""


# 状态定义
class State(TypedDict):
    numbers: List[int]  # 输入的数字
    results: Annotated[list[int], operator.add]  # worker的结果
    final_sum: int  # 最终求和


# 1. Map阶段：分发数字
def split_numbers(state: State):
    """把数字分发给不同的worker"""
    numbers = state["numbers"]

    # 每个数字发给一个worker
    return [Send("worker", {"number": num}) for num in numbers]


# 2. Worker阶段：计算平方
def calculate_square(state: State):
    """每个worker计算一个数字的平方"""
    number = state["number"]
    square = number * number
    return {"results": [square]}


# 3. Reduce阶段：求和
def sum_results(state: State):
    """把所有结果加起来"""
    results = state.get("results", [])
    total = sum(results)
    return {"final_sum": total}


# 构建图
def create_simple_graph():
    graph = StateGraph(state_schema=State)

    # 添加节点
    graph.add_node("splitter", lambda s: s)  # 分发器
    graph.add_node("worker", calculate_square)  # 工作节点
    graph.add_node("summer", sum_results)  # 求和器

    # 连接节点
    graph.add_edge(START, "splitter")
    graph.add_conditional_edges("splitter", split_numbers, ["worker"])  # Map阶段
    graph.add_edge("worker", "summer")  # Worker完成后求和
    graph.add_edge("summer", END)

    return graph.compile()


"""---------------------------messages模式---------------------------"""
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm1 = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                       model_provider="openai",
                       model='MiniMax-M2.1')

class MyState(TypedDict):
    question: str
    results: str


def generate_answer(state: MyState):
    question = state["question"]
    answer = llm1.invoke([
        {"role": "user", "content": f"{question}"}
    ])
    return {"answer": answer.content}

def generate_answer1(state: MyState):
    answer = llm1.invoke([
        {"role": "user", "content": f"你好"}
    ])
    return {"answer": answer.content}


# 构建图
def create_llm_graph():
    graph = StateGraph(state_schema=MyState)

    # 添加节点
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("generate_answer1", generate_answer1)

    # 连接节点
    graph.add_edge(START, "generate_answer")
    graph.add_edge("generate_answer", "generate_answer1")
    graph.add_edge("generate_answer", END)

    return graph.compile()


"""---------------------------自定义模式---------------------------"""

from langgraph.config import get_stream_writer
import time


# 定义状态
class FileState(TypedDict):
    filename: str  # 文件名称
    content: str  # 文件内容
    word_count: int  # 内容数量
    processed: bool  # 是否处理完成


def read_file(state: FileState):
    """步骤1：读取文件"""
    writer = get_stream_writer()
    # 发送开始信息
    writer({"step": "读取文件", "status": "开始", "progress": 0})
    time.sleep(1)

    # 发送进度信息
    writer({"step": "读取文件", "status": "正在读取...", "progress": 50})
    time.sleep(1)

    # 模拟文件内容
    content = "这是一个示例文件，包含一些文本内容。"

    # 发送完成信息
    writer({
        "step": "读取文件",
        "status": "完成",
        "progress": 100,
        "data": {"size": len(content)}
    })

    return {"content": content}


def count_words(state: FileState):
    """步骤2：统计字数"""
    writer = get_stream_writer()
    writer({"step": "统计字数", "status": "开始", "progress": 0})
    time.sleep(0.5)

    writer({"step": "统计字数", "status": "正在分析...", "progress": 30})
    time.sleep(1)

    writer({"step": "统计字数", "status": "计算中...", "progress": 70})
    time.sleep(0.5)

    # 计算字数
    word_count = len(state["content"])

    writer({
        "step": "统计字数",
        "status": "完成",
        "progress": 100,
        "data": {"word_count": word_count}
    })

    return {"word_count": word_count}


def finalize_processing(state: FileState):
    """步骤3：完成处理"""
    writer = get_stream_writer()
    writer({"step": "完成处理", "status": "生成报告", "progress": 50})
    time.sleep(1)

    writer({
        "step": "完成处理",
        "status": "全部完成",
        "progress": 100,
        "data": {
            "filename": state["filename"],
            "total_chars": state["word_count"],
            "summary": f"文件 {state['filename']} 处理完成，共 {state['word_count']} 个字符"
        }
    })

    return {"processed": True}


# 构建图
def create_custom_graph():
    graph = (
        StateGraph(state_schema=FileState)
        .add_node("read_file", read_file)
        .add_node("count_words", count_words)
        .add_node("finalize", finalize_processing)
        .add_edge(START, "read_file")
        .add_edge("read_file", "count_words")
        .add_edge("count_words", "finalize")
        .compile()
    )
    return graph


"""---------------------------checkpoints、tasks模式---------------------------"""
# 需要提供检查点
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(api_key=os.getenv("DASHSCOPE_API_KEY"),
                      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                      model_provider="openai",
                      model='MiniMax-M2.1')


# 创建子图
def subplot(state: MessagesState) -> MessagesState:
    # 获取大模型回答的内容进行摘要总结
    answer = state["messages"][-1].content
    summary_prompt = f"请用一句话总结下面这句话：\n\n答：{answer}"
    response = llm.invoke(summary_prompt)
    return {"messages": [response]}


summary_subgraph = (
    StateGraph(state_schema=MessagesState)
    .add_node("subplot", subplot)
    .add_edge(START, "subplot")
    .add_edge("subplot", END)
    .compile()
)


# 创建父图
def llm_answer_node(state: MessagesState) -> MessagesState:
    # 使用大模型进行回答
    answer = llm.invoke(state["messages"])
    return {"messages": [answer]}


checkpointer = InMemorySaver()

# 构建图
def create_check_tasks_graph():
    parent_graph = (
        StateGraph(MessagesState)
        .add_node("llm_answer", llm_answer_node)
        .add_node("summarize_subgraph", summary_subgraph)
        .add_edge(START, "llm_answer")
        .add_edge("llm_answer", "summarize_subgraph")
        .compile(checkpointer=checkpointer)
    )
    return parent_graph


# 运行例子
def run_example():
    app = create_simple_graph()  # values、updates、debug模式案例
    app1 = create_llm_graph()  # messages模式
    app2 = create_custom_graph()  # 自定义模式
    app3 = create_check_tasks_graph()  # checkpoints、tasks模式

    # 测试数据
    initial_state = {
        "numbers": [1, 2, 3, 4, 5],
        "results": [],
        "final_sum": 0
    }

    # print("====================VALUES模式=====================")  常用
    # for result in app.stream(initial_state, stream_mode="values", version="v2"):  # 开启v2新版本的输出。注意点v要小写
    #     print(result)
    #
    # print("====================UPDATES模式=====================")
    # for result in app.stream(initial_state, stream_mode="updates", version="v2"):
    #     print(result)
    #
    # print("====================DEBUG模式=====================")
    # for result in app.stream(initial_state, stream_mode="debug", version="v2"):
    #     print(result)

    # print("====================MESSAGES模式=====================")   常用
    # for chunk in app1.stream({"question": "什么是状态图？"}, stream_mode="messages", version="v2"):
    #     if chunk["type"] == "messages":
    #         result, metadata = chunk["data"]
    #         if metadata["langgraph_node"] == "generate_answer":  # 可以从元数据中过滤不同节点中的llm回复
    #             print(result.content, end="", flush=True)
    #         # print(metadata)

    # print("====================CUSTOM模式=====================")
    # # 初始状态
    # initial_state1 = {
    #     "filename": "example.txt",
    #     "content": "",
    #     "word_count": 0,
    #     "processed": False
    # }
    # # 使用Custom模式运行
    # for chunk in app2.stream(initial_state1, stream_mode="custom", version="v2"):
    #     if chunk["type"] == "custom":
    #         data = chunk["data"]
    #         step = data.get("step", "")  # 当前步骤
    #         status = data.get("status", "")  # 目前状态
    #         progress = data.get("progress", 0)  # 完成进度
    #         data_result = data.get("data", {})  # 最终数据
    #
    #         # 显示进度
    #         progress_bar = "█" * (progress // 10) + "░" * (10 - progress // 10)
    #         print(f"\n[{step}] {status}")
    #         print(f"进度: [{progress_bar}] {progress}%")
    #
    #         # 显示额外数据
    #         if data_result:
    #             for key, value in data_result.items():
    #                 print(f"{key}: {value}")

    print("====================checkpoints、tasks模式=====================")

    config = {"configurable": {"thread_id": "1"}}
    # 测试输入
    input_state = {
        "messages": [{"role": "user", "content": "langgraph是什么？请用100字介绍"}],
    }

    for chunk in app3.stream(
            input_state,
            config,
            stream_mode="tasks",  # tasks  |  checkpoints
            subgraphs=True  # 如果要子图也进行流式输出，需要开启
    ):
        print(chunk)



if __name__ == "__main__":
    run_example()
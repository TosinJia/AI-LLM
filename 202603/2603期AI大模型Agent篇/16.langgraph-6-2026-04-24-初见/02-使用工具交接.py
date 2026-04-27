from typing_extensions import Literal

from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# 初始化大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model='kimi-k2.5',
    temperature=0.7
)


def make_handoff_tool(*, agent_name: str):
    """
    创建一个工具交接函数，用于在代理之间进行转接

    *, agent_name: str：*是做什么的 ？*就是强制让你在调用函数的传递参数的时候指定参数名  make_handoff_tool(agent_name="")

    Args:
        agent_name (str): 目标代理的名称

    Returns:
        tool: 返回一个可以执行代理转接的工具函数
    """
    # 根据目标代理名称动态生成工具名称
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
            runtime: ToolRuntime
    ):
        """请求另一个代理的帮助进行任务交接"""

        # 返回Command对象，用于导航到父图中的另一个代理节点
        return Command(
            # 导航到目标代理节点
            goto=agent_name,
            # 在父图中执行导航， 从一个子图去往父图的其他子图的时候必须加上Command.PARENT
            graph=Command.PARENT,
            # 更新状态：将完整的消息历史传递给目标代理，并添加工具消息
            # 这确保了聊天历史的完整性和有效性， 在子图跳转到其他的子图中时需要将当前所以的状态进行更新（runtime.state["messages"]）
            update={"messages": runtime.state["messages"] + [
                ToolMessage(name=tool_name, content=f"成功转接到 {agent_name} 代理，请开始进行运算",
                            tool_call_id=runtime.tool_call_id)]},
        )

    return handoff_to_agent


def make_agent(model, tools, system_prompt=None):
    """
    创建一个智能代理，能够使用工具并在需要时进行代理转接

    Args:
        model: 语言模型实例
        tools: 代理可用的工具列表
        system_prompt: 系统提示词，定义代理的角色和行为

    Returns:
        compiled_graph: 编译后的代理图
    """
    # 将工具绑定到模型上
    model_with_tools = model.bind_tools(tools)

    # 创建工具节点
    tool_node = ToolNode(tools)

    def call_model(state: MessagesState) -> Command[Literal["call_tools", END]]:
        """
        调用语言模型生成响应

        Args:
            state: 当前消息状态

        Returns:
            Command: 如果需要调用工具则转到call_tools，否则结束
        """
        messages = state["messages"]

        print("和模型交互之前：", messages)
        # 如果有系统提示词，将其添加到消息开头
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # 调用绑定了工具的模型
        response = model_with_tools.invoke(messages)

        # 检查模型是否决定使用工具
        if len(response.tool_calls) > 0:
            # 如果有工具调用，转到工具执行节点
            return Command(goto="call_tools", update={"messages": [response]})

        # 如果没有工具调用，直接返回响应消息
        return {"messages": [response]}

    # 构建代理的内部图结构
    graph = StateGraph(MessagesState)

    # 添加模型调用节点和工具调用节点
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", tool_node)

    # 设置图的边：从开始到模型调用，从工具调用回到模型调用
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        tools_condition,  # 内置函数：判断消息中是否有 tool_calls
        {"tools": "call_tools", "__end__": END},
    )

    graph.add_edge("call_tools", "call_model")

    # 编译并返回图
    return graph.compile()


def pretty_print_stream(chunk):
    """
    流式输出美化工具
    """
    # StreamPart 对象包含三个核心属性
    msg_type = chunk["type"]  # str: 'updates', 'metadata', 'values' 等
    ns = chunk["ns"]  # tuple: 命名空间
    data = chunk["data"]  # Any: 更新的具体内容
    for node_name, node_update in data.items():
        # 1. 打印节点标题（区分是谁在干活）
        print(f"\n正在运行节点: [{node_name}]")
        print("-" * 30)

        # 2. 检查是否有消息更新
        if "messages" in node_update:
            for msg in node_update["messages"]:
                # --- 核心提取逻辑 ---

                # 如果是 AI 说的话
                if msg.type == "ai":
                    if msg.content:
                        print(f"AI: {msg.content.strip()}")
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"[工具调用] 执行 {tc['name']}，参数: {tc['args']}")

                # 如果是工具返回的结果
                elif msg.type == "tool":
                    print(f"[工具结果] 得到: {msg.content}")

                # 如果是人类的输入
                elif msg.type == "human":
                    print(f"用户: {msg.content}")


# ============= 定义数学工具 =============

@tool
def add(a: int, b: int) -> int:
    """执行两个数字的加法运算"""
    result = a + b
    print(f"执行加法: {a} + {b} = {result}")
    return result


@tool
def multiply(a: int, b: int) -> int:
    """执行两个数字的乘法运算"""
    result = a * b
    print(f"执行乘法: {a} × {b} = {result}")
    return result


@tool
def subtract(a: int, b: int) -> int:
    """执行两个数字的减法运算"""
    result = a - b
    print(f"执行减法: {a} - {b} = {result}")
    return result


@tool
def divide(a: int, b: int) -> float:
    """执行两个数字的除法运算"""
    if b == 0:
        return "错误：不能除以零"
    result = a / b
    print(f"执行除法: {a} ÷ {b} = {result}")
    return result


# ============= 演示单个代理 =============

def demo_single_agent():
    """演示单个具有所有数学工具的代理"""
    print("=" * 60)
    print("演示：单个数学代理")
    print("=" * 60)

    # 创建一个拥有所有数学工具的代理
    math_agent = make_agent(
        llm,
        [add, multiply, subtract, divide],
        system_prompt="你是一个数学专家，可以执行各种数学运算。请一步步解决问题。"
    )

    print("问题: 计算 (3 + 5) × 12")
    print()

    # 运行代理并显示结果
    for chunk in math_agent.stream({"messages": [("user", "计算 (3 + 5) × 12")]}):
        pretty_print_stream(chunk)


# ============= 演示多代理协作 =============

def demo_multi_agent_collaboration():
    """演示多个专业代理之间的协作"""
    print("=" * 60)
    print("演示：多代理协作系统")
    print("=" * 60)

    # 创建加法专家代理    子图1
    addition_expert = make_agent(
        llm,
        [add, subtract, make_handoff_tool(agent_name="multiplication_expert")],
        system_prompt="""你是加法和减法专家。你精通加法和减法运算，必须使用工具去计算加法。
            当你完成加法或减法运算后，如果后续还需要乘法或除法运算，
            请立即使用 transfer_to_multiplication_expert 工具转接给乘法专家。
            不要尝试自己完成乘法运算。"""
    )

    # 创建乘法专家代理    子图2
    multiplication_expert = make_agent(
        llm,
        [multiply, divide, make_handoff_tool(agent_name="addition_expert")],
        system_prompt="""你是乘法和除法专家。你精通乘法和除法运算。
            当你接收到需要乘法运算的任务时，必须使用工具执行乘法运算。
            如果后续还需要加法或减法运算，请使用transfer_to_addition_expert工具转接给加法专家。
            当前任务：执行乘法运算并给出最终答案。"""
    )

    # 构建多代理协作图
    builder = StateGraph(MessagesState)

    # 添加两个专家代理节点
    builder.add_node("addition_expert", addition_expert)
    builder.add_node("multiplication_expert", multiplication_expert)

    # 设置入口点为加法专家
    builder.add_edge(START, "addition_expert")

    # 编译协作图
    collaboration_graph = builder.compile()

    print("问题: 计算 (3 + 5) × 12")
    print("加法专家将处理加法，然后转接给乘法专家处理乘法")
    print()

    # 运行协作图并显示子图中的所有更新
    for chunk in collaboration_graph.stream(
            {"messages": [HumanMessage(content="请计算 (3 + 5) × 12")]},
            subgraphs=True,  # 包含子图更新
            version="v2",
            stream_mode="updates"
    ):
        # 1.先加法专家进行(3+5)=8，接下来需要计算8*12，需要转接给乘法专家   2.乘法专家计算8*12得到结果
        pretty_print_stream(chunk)


# ============= 更复杂的协作示例 =============

def demo_complex_collaboration():
    """演示更复杂的多步骤协作"""
    print("=" * 60)
    print("演示：复杂多步协作")
    print("=" * 60)

    # 创建基础运算专家
    basic_math_expert = make_agent(
        llm,
        [add, subtract, make_handoff_tool(agent_name="advanced_math_expert")],
        system_prompt="""你是基础数学专家。你的唯一职责是执行“加法(add)”和“减法(subtract)”。
        执行逻辑规范：
        1. 观察算式，如果存在可以直接进行的加法或减法（尤其是括号内的），请立即调用工具计算。
        2. 严禁尝试口算，必须通过工具获得结果。
        3. 严禁执行乘法或除法。如果你发现当前步骤必须先进行乘除法才能继续，请立即转接给高级专家。
        4. 只要你刚刚完成了一步加/减法计算，请停下来观察剩下的算式：
           - 如果剩下的算式里还有你能算的加减法，继续算。
           - 如果剩下的部分只涉及乘除法，立即转接到 advanced_math_expert。
        不要道歉，不要解释，只负责计算或转接。"""
    )

    # 创建高级运算专家
    advanced_math_expert = make_agent(
        llm,
        [multiply, divide, make_handoff_tool(agent_name="basic_math_expert")],
        system_prompt="""你是高级数学专家。你的唯一职责是执行“乘法(multiply)”和“除法(divide)”。
        执行逻辑规范：
        1. 观察算式，如果你发现当前必须先执行加法或减法（例如括号内的内容尚未解出），请立即转接到 basic_math_expert。
        2. 如果当前步骤可以直接进行乘法或除法，请立即调用工具计算。
        3. 严禁尝试口算，必须通过工具获得结果。
        4. 只要你刚刚完成了一步乘/除法计算，请停下来观察剩下的算式：
           - 如果剩下的算式需要基础运算（加减），立即转接到 basic_math_expert。
           - 如果剩下的全是乘除，继续计算直到得出最终结果。
        你的目标是完成计算，但在遇到加减法时要坚决交接，不要自己通过“口算”来跳过步骤。"""
    )

    # 构建协作图
    builder = StateGraph(MessagesState)
    builder.add_node("basic_math_expert", basic_math_expert)
    builder.add_node("advanced_math_expert", advanced_math_expert)
    builder.add_edge(START, "basic_math_expert")

    complex_graph = builder.compile()

    print("复杂问题: 计算 ((10 + 5) × 3 - 8) ÷ 2")
    print("将需要多次代理转接来完成计算")
    print()

    for chunk in complex_graph.stream(
            {"messages": [HumanMessage(content="请逐步计算 ((10 + 5) × 3 - 8) ÷ 2")]},
            subgraphs=True,
            version="v2",
            stream_mode="updates"
    ):
        pretty_print_stream(chunk)


# ============= 主程序入口 =============

def main():
    """主程序，运行所有演示"""
    print("LangGraph工具交接案例演示")
    print("展示单代理和多代理协作的数学计算系统")
    print()

    try:
        # 演示1：单个代理
        # demo_single_agent()
        #
        # print("\n" + "-" * 20 + "\n")

        # 演示2：多代理协作
        # demo_multi_agent_collaboration()

        # print("\n" + "-" * 20 + "\n")
        #
        # # 演示3：复杂协作
        demo_complex_collaboration()

    except Exception as e:
        print(f"运行出错: {e}")


if __name__ == "__main__":
    main()

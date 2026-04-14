"""
LangGraph Map-Reduce 简单案例：数字求和
把一堆数字分给多个worker算平方，然后把结果加起来
"""
from typing import Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import TypedDict, List

# 什么时候使用send：
# 1.不确定当前需要执行多少个任务（循环）
# 2.实现并发/并行

# 必须在条件边去使用send       有什么情况不在条件边中去使用send（只是用send进行条件路由跳转）



# 任务：统计每个数字的平方和
# 场景，一个任务中需要同时想多个节点发送内容开始执行，最后统一汇总
# 开发任务：前端做html，后端做web开发、UI、-----》最终产品
# 状态定义
class State(TypedDict):
    numbers: List[int]        # 输入的数字
    results: Annotated[list[int], operator.add]        # worker的结果
    final_sum: int           # 最终求和

# 1. Map阶段：分发数字
def split_numbers(state: State):
    """把数字分发给不同的worker"""
    numbers = state["numbers"]
    print(f"分发数字: {numbers}")

    # 每个数字发给一个worker    send是可以发送临时状态的
    # 参数1：需要发送到的节点
    # 参数2：对应的状态内容
    return [Send("worker", {"number": num}) for num in numbers]  # 进行多个send发送

# 2. Worker阶段：计算平方
def calculate_square(state: State):
    """每个worker计算一个数字的平方"""
    number = state["number"]
    square = number * number
    print(f"Worker: {number}² = {square}")
    return {"results": [square]}

# 3. Reduce阶段：求和
def sum_results(state: State):
    """把所有结果加起来"""
    results = state.get("results", [])   # 如果worker节点还有没执行完的，会在这进行阻塞
    total = sum(results)
    print(f"求和: {results} = {total}")
    return {"final_sum": total}

# 构建图
def create_simple_graph():
    graph = StateGraph(State)

# lambda s: s   等同于下面的写法
#     def 匿名函数(s):
#         return s

    # 添加节点
    graph.add_node("splitter", lambda s: s)  # 分发器
    graph.add_node("worker", calculate_square)  # 工作节点
    graph.add_node("summer", sum_results)      # 求和器

    # 连接节点
    graph.add_edge(START, "splitter")
    graph.add_conditional_edges("splitter", split_numbers, ["worker"])  # Map阶段
    graph.add_edge("worker", "summer")  # Worker完成后求和
    graph.add_edge("summer", END)

    return graph.compile()

# 运行例子
def run_example():
    app = create_simple_graph()

    # 测试数据
    initial_state = {
        "numbers": [1, 2, 3, 4, 5],
        "results": [],
        "final_sum": 0
    }

    print("开始计算...")
    print("任务：计算每个数字的平方，然后求和")
    print()

    # 运行
    result = app.invoke(initial_state)
    print(result)

if __name__ == "__main__":
    run_example()
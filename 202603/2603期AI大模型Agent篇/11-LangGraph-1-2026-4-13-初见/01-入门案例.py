from typing import TypedDict
from langgraph.graph import StateGraph

# 定义输入state  类似于一个函数的入参
class InputState(TypedDict):
    question: str

# 定义输出state   定义工作流最终的输出状态内容，类似于函数的返回值
class OutputState(TypedDict):
    final_answer: str

# 定义全局state   1.TypedDict(推荐使用)    2.pydantic-BaseModel
# 实际上状态还是维护在全局state中，输入和输出只是往里面填充和获取
class MyState(TypedDict):
    question: str
    answer: str
    final_answer: str



# 搜索节点
def search_node(state: MyState):
    print(f"我进行了一次搜索{state['question']}")
    return {"answer":"搜索到了一个答案"}

# 节点工作节点
def llm_result(state: MyState):
    print(f"<UNK>{state['answer']}")
    print(f"对搜索到的内容，通过llm去进行总结回复")
    return {"final_answer": "这就是模型总结的答案"}

# 初始化状态图，维护的全局状态MyState
state_graph = StateGraph(state_schema=MyState,
                         input_schema=InputState,
                         output_schema=OutputState)

# 参数1：节点名称，参数2：对应的节点函数
state_graph.add_node("llm_result", llm_result)  # 在图中添加对应的节点
state_graph.add_node("search_node", search_node)
# 定义入口
state_graph.set_entry_point("search_node")
# 定义边（定义流程走向）
state_graph.add_edge("search_node", "llm_result")
# 构建图
graph = state_graph.compile()


# 执行图
result = graph.invoke({"question": "你是谁"})
print(result)
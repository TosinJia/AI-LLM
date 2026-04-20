from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from typing import Annotated
from operator import add


class TaskState(TypedDict):
    task_id: str  # 任务id
    title: str  # 标题
    assignee: str  # 接收任务的人
    priority: int  # 优先级
    comments: Annotated[list, add]  # 评论
    status: str  # 状态


def create_task(state: TaskState):
    """创建任务"""
    return {
        "status": "进行中",
        "comments": [f"任务 '{state['title']}' 已创建，分配给 {state['assignee']}"]
    }


def update_task(state: TaskState):
    """更新任务"""

    return {
        "status": "已更新",
        "comments": ["任务状态已更新"]
    }


def plan_task(state: TaskState):
    """准备任务完成"""
    return {
        "status": "已准备",
        "comments": [f"任务 '{state['title']}' 已准备"]
    }


# 创建任务管理流程
workflow = StateGraph(state_schema=TaskState)
workflow.add_node("create", create_task)
workflow.add_node("update", update_task)
workflow.add_node("complete", plan_task)
workflow.add_edge(START, "create")
workflow.add_edge("create", "update")
workflow.add_edge("update", "complete")
workflow.add_edge("complete", END)

checkpointer = InMemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 创建任务
config = {"configurable": {"thread_id": "task_001"}}
result = app.invoke({
    "task_id": "T001",
    "title": "开发注册功能",
    "assignee": "张三",
    "priority": 1,
    "comments": [],
    "status": "待分配"
}, config)

print("=== 任务创建完成 ===")
print(f"状态: {result['status']}")
print("评论:", result['comments'])

# 演示 as_node 参数, 或者指定某个具体检查点的config去进行更新
# 本质是创建了一个新的检查点，再次调用invoke方法会从新的检查点继续执行剩下流程
history = list(app.get_state_history(config))
# 获取complete节点的上一个update节点
before_complete = next(s for s in history if s.next == ("complete",))
print("\n=== 使用 as_node 参数 ===")

# 对update节点状态的修改，但实际上，是重新创建了一个检查点，"自动通知：任务优先级提升"存入state中
# 如果是通过before_complete.config获取的config内容，as_node可以省略。
app.update_state(before_complete.config, {
    "status": "已更新",
    "comments": ["自动通知：任务优先级提升"],
    "priority": 3
})
# 执行一次流程   如果修改完状态后，想让图从update节点执行起来，需要手动调用invoke方法
final_result = app.invoke(None, config)
# # 查看最终状态
final_state = app.get_state(config)
print("最终状态:", final_result["status"])
print("完整评论:", final_result["comments"])

# # 手动更新状态 - 添加评论
# print("\n=== 手动添加评论 ===")
# app.update_state(config, {
#     "comments": ["项目经理：请在周五前完成"],
#     "priority": 2
# })
#
# # 查看更新后的状态
# updated_state = app.get_state(config)
# print(f"优先级: {updated_state.values['priority']}")
# print("所有评论:", updated_state.values['comments'])
#
# # 继续更新 - 添加更多评论
# print("\n=== 添加更多评论 ===")
# app.update_state(config, {
#     "comments": ["张三：已完成开发"],
#     "status": "开发完成"
# })
#
# # 查看最终状态
# final_state = app.get_state(config)
# print(f"最终状态: {final_state.values['status']}")
# print("完整评论历史:", final_state.values['comments'])
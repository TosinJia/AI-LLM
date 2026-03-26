from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.utils.workflow import draw_all_possible_flows


class MyWorkflow(Workflow):
    # 在当前这个类中用@step装饰器装饰的函数他就是一个步骤（节点）
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")


async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    # await关键字：用于等待异步操作完成，只能在async函数内使用。
    result = await w.run()

    draw_all_possible_flows(w, "my_workflow.html")
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
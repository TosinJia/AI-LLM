from llama_index.core.workflow import InputRequiredEvent, HumanResponseEvent
from llama_index.core.workflow import step, StopEvent, StartEvent, Workflow
import asyncio


class HumanInTheLoopWorkflow(Workflow):
    @step
    async def step1(self, ev: StartEvent) -> InputRequiredEvent:
        # 提示用户输入预算信息
        return InputRequiredEvent(prefix="请问你的预算是多少？ ")

    @step
    async def step2(self, ev: HumanResponseEvent) -> StopEvent:
        # 在这接收到人类输入的内容，进行处理
        res = ev.response

        return StopEvent(result=f"根据你的预算：{res}，即将为你生成一个规划")


async def main():
    handler = HumanInTheLoopWorkflow().run()

    # 获取一个工作流处理器，就是一个异步生成器，通过迭代这个生成器，可以实时的捕获工作流的执行状态
    async for event in handler.stream_events():
        # 如果获取的是InputRequiredEvent对象，那么就可以让人类进行输入
        if isinstance(event, InputRequiredEvent):
            # 获取人类的问题
            response = input(event.prefix)
            handler.ctx.send_event(HumanResponseEvent(response=response))

    final_result = await handler
    print(final_result)


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
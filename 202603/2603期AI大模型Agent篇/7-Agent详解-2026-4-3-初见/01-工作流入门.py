from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step
)

from 加载模型 import get_llm

llm, e = get_llm()


# 自定义工作流
class JokeFlow(Workflow):

    # 创建工作流入口节点，在入参中加上StartEvent
    @step
    async def generate_joke(self, ev: StartEvent) -> StopEvent:
        topic = ev.topic
        result = await llm.acomplete(f"帮我生成一个关于{topic}的小故事，字数在100")
        return StopEvent(result)


async def main():
    # 实例化当前工作流
    w = JokeFlow()
    result = await w.run(topic="狼来了")
    print(result)


if __name__ == '__main__':
    import asyncio
    # python 中开启异步最简单的方式
    asyncio.run(main())

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)
from 加载模型 import get_llm


class JokeEvent(Event):
    """
        定义工作流事件:事件是用户定义的 pydantic 对象。您可以控制其属性和任何其他辅助方法。
    """
    joke: str


class JokeFlow(Workflow):
    """
        设置工作流类：工作流通过子类继承Workflow

    """

    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm

    @step
    async def generate_joke(self, ctx: Context, ev: StartEvent) -> JokeEvent:
        """
            工作流入口
            StartEvent：表示向何处发送初始工作流输入
                它可以保存任意属性。这里，我们使用 访问了主题 ev.topic，如果不存在该属性，则会引发错误。
                您也可以使用ev.get("topic")来处理属性可能不存在的情况，而不会引发错误。
        """
        topic = ev.topic

        prompt = f"帮我生成一个关于 {topic}的小故事，字数在100字左右."
        response = await self.llm.acomplete(prompt)
        # 存储一个k-v形式的数据
        await ctx.store.set("response", response)
        return JokeEvent(joke=str(response))

    @step
    async def critique_joke(self, ctx: Context, ev: JokeEvent) -> StopEvent:
        """
            工作流出口点:当工作流遇到了StopEvent他会立刻停止并返回内容
        """
        joke = ev.joke
        # 获取对应的值
        print(await ctx.store.get("response"))
        prompt = f"对下面的故事进行全面的分析: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


async def main():
    # 加载大模型和嵌入模型
    llm, embed_model = get_llm()
    w = JokeFlow(llm, timeout=60, verbose=False)
    # 导入WorkflowServer将工作流当作一个服务
    from workflows.server import WorkflowServer

    server = WorkflowServer()
    server.add_workflow("my_workflow", w)
    await server.serve("127.0.0.1", 8080)


if __name__ == '__main__':
    import asyncio

    # 因为w.run是异步的，所以我们需要使用异步的形式去启动程序
    asyncio.run(main())
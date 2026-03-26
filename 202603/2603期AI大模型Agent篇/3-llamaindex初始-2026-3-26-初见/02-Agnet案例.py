import asyncio
from llama_index.core.agent.workflow import FunctionAgent
# pip install llama-index-llms-deepseek
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.dashscope import DashScope
import os
from dotenv import load_dotenv
'''
QWEN模型对llama index中的agent支持不好，所以采用DeepSeek
'''
# 加载 API 配置
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
api_base_url = os.getenv("DEEPSEEK_BASE_URL")

# 选择模型
model = "deepseek-chat"
llm = DeepSeek(model=model, api_key=api_key, api_base_url=api_base_url, temperature=0.1)
# model = "qwen3-max"
# api_key = os.getenv("DASHSCOPE_API_KEY")
# api_base_url = os.getenv("DASHSCOPE_BASE_URL")
#
# # 初始化千问模型(设置成默认)
# llm = DashScope(model_name=model, api_key=api_key, api_base_url=api_base_url)


# 定义一个简单的计算器工具
def multiply(a: float, b: float) -> float:
    """两个数相乘并返回乘积"""
    return a * b


def add(a: float, b: float) -> float:
    """将两个数相加并返回和"""
    return a + b


workflow = FunctionAgent(
    tools=[multiply, add],
    llm=llm,
    system_prompt="你是一个可以使用工具执行基本数学运算的代理。请用中文回答",
)


async def main():
    response = await workflow.run(user_msg="请计算20+(2*4)?")
    print("=== 最终响应 ===")
    print(response)

    # 运行代理
    # from llama_index.core.agent.workflow import AgentStream
    #
    # handler = workflow.run("请用中文计算：：20+(2*4)?", )
    #
    # async for ev in handler.stream_events():
    #     if isinstance(ev, AgentStream):
    #         print(f"{ev.delta}", end="", flush=True)
    #
    # response = await handler
    # print(response)

# 运行代理
if __name__ == "__main__":
    asyncio.run(main())
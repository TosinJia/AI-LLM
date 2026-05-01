import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient  # pip install langchain-mcp-adapters
from langchain.agents import create_agent   # pip install langchain
from langchain_openai import ChatOpenAI     # pip install langchain-openai
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="kimi-k2.5",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)


async def main():
    client = MultiServerMCPClient(
        {
            "fast-health-calculator": {
                "transport": "stdio",
                "command": "python",
                "args": [r"01-first_mcp_server.py"],
            },

        }
    )

    tools = await client.get_tools()
    print(tools)
    agent = create_agent(
        llm,
        tools
    )
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "请计算8+56=?"}]}
    )
    bmi_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "我的体重是75KG，身高是1.80m，请计算我的bmi？"}]}
    )
    print(math_response)
    print(bmi_response)

    # 获取提示词
    prompt = await client.get_prompt(prompt_name="analyze_my_health", server_name="fast-health-calculator",
                                     arguments={"name": "Alice", "weight": "60", "height": "1.65"})
    print(prompt)

    # 获取资源
    resources = await client.get_resources(server_name="fast-health-calculator", uris=["resource://health-guidelines"])
    print(resources[0].as_string())


if __name__ == "__main__":
    asyncio.run(main())
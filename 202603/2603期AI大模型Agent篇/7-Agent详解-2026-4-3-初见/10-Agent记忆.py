from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch  # 推荐使用标准的社区版导入
from langgraph.checkpoint.memory import InMemorySaver  # 短期记忆
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 1. 配置 LLM (通义千问)
model = "qwen3.5-flash-2026-02-23"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=api_base_url,
    temperature=0.1
)
# 2. 配置工具
# TavilySearchResults 是 LangChain 社区标准的搜索工具封装
tools = [TavilySearch(max_results=1)]

# 3. 创建 Agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个超级智能助手，能帮助用户解决问题",
    checkpointer=InMemorySaver(),
)

# 4. 执行任务
config = {"configurable": {"thread_id": "1"}}
query1 = "请问现任的美国总统是谁？他的年龄的平方是多少? 请用中文告诉我这两个问题的答案"
query2 = "请问我上一个问题问了什么？"
try:
    result = agent.invoke({"messages": [{"role": "user", "content": query1}]}, config)
    print("\n====== 第一次回答 ======")
    print(result["messages"][-1].content)
    result = agent.invoke({"messages": [{"role": "user", "content": query2}]}, config)
    print("\n====== 第二次回答 ======")
    print(result["messages"][-1].content)
except Exception as e:
    print(f"发生错误: {e}")
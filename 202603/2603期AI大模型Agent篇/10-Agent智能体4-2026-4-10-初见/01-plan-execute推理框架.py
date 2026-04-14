from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner # pip install langchain-experimental
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import numexpr
import math
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
# 初始化模型
model = "MiniMax-M2.1"
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")
# 初始化千问模型
llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

# 创建工具
search = TavilySearch()


# 创建一个数学计算工具
@tool
def calculator(expression: str) -> str:
    """支持多个表达式的数学计算器（使用 numexpr）"""
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        expressions = [expr.strip() for expr in expression.split(";") if expr.strip()]
        results = []

        for expr in expressions:
            res = numexpr.evaluate(expr, global_dict={}, local_dict=local_dict)
            results.append(f"{expr} = {res}")

        return "\n".join(results)
    except Exception as e:
        return f"计算错误: {str(e)}"


# 定义工具列表
tools = [
    search,
    calculator
]

# 初始化规划区+执行器
planner = load_chat_planner(llm)  # 规划
executor = load_agent_executor(llm, tools, verbose=True)  # 执行器

# plan-execute代理
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
# 1.换算汇率2.搜索酒店价格
result = agent.invoke(
    "我有5000人民币的预算，想去日本旅行5天。请帮我计算一下，按照当前汇率，这些钱在东京能住几晚中等价位的酒店？")

print(result)

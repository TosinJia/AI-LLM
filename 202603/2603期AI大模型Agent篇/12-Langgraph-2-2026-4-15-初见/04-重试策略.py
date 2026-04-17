import sqlite3
from langchain.chat_models import init_chat_model # pip install langchain # pip install langchain-core
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.types import RetryPolicy
from langchain_community.utilities import SQLDatabase # pip install langchain-community
from langchain.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from dotenv import load_dotenv # pip install python-dotenv

load_dotenv()

db = SQLDatabase.from_uri("sqlite:///:memory:")
model = init_chat_model("deepseek-chat") # pip install langchain-deepseek


def query_database(state: MessagesState, runtime: Runtime):
    print(f"正在尝试第 {runtime.execution_info.node_attempt} 次查询...")
    # 手动抛出一个异常来强制触发重试  runtime.execution_info.node_attempt获取当前节点的重试次数
    if runtime.execution_info.node_attempt < 3:
        print("模拟数据库连接失败...")
        raise sqlite3.OperationalError("Database connection lost")  # 进行重试

    query_result = db.run("SELECT 1;")  # 模拟成功
    return {"messages": [AIMessage(content=str(query_result))]}


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Define a new graph
builder = StateGraph(MessagesState)
builder.add_node(
    "query_database",
    query_database,
    retry_policy=RetryPolicy(retry_on=[sqlite3.OperationalError, sqlite3.IntegrityError]),  # 可以自己设定需要触发的异常类
)
builder.add_node("model", call_model, retry_policy=RetryPolicy(max_attempts=5))  # 重试次数
builder.add_edge(START, "model")
builder.add_edge("model", "query_database")
builder.add_edge("query_database", END)
graph = builder.compile()

response = graph.invoke({"messages": [HumanMessage(content="你好呀？")]})
print(response)

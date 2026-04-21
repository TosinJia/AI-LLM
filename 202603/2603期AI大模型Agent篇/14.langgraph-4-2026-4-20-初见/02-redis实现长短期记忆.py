import uuid
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis import RedisSaver  # 短期记忆 # pip install -U langgraph-checkpoint-redis
from langgraph.store.redis import RedisStore  # 长期记忆
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.runtime import Runtime
from dataclasses import dataclass
import redis
from langgraph.store.base import BaseStore

load_dotenv()

# --- 1. 初始化模型 ---
llm = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_provider="openai",
    model='MiniMax-M2.1'
)


@dataclass
class Context:
    user_id: str


# 定义节点
def call_model(state: MessagesState, runtime: Runtime[Context]):
    # 获取用户id
    user_id = runtime.context.user_id
    # 创建命名空间
    namespace = ("memories", user_id)
    # 获取用户最新的问题
    last_message = state["messages"][-1].content
    # 通过runtime去进行搜索长期记忆
    memories = runtime.store.search(namespace, query=last_message)

    # 将检索到的长期记忆填充到提示词中
    user_info = "\n".join([d.value["data"] for d in memories])
    system_prompt = (f"你是一个乐于助人的助手\n\n"
                     f"用户信息：{user_info}\n\n"
                     f"如果用户信息不为空才根据已知的用户信息进行回答")

    # 调用llm进行回复
    result = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])

    # 怎么存长期记忆?    长期记忆，一般存储用户相关的信息（用户的习惯、姓名等）  根据公司业务来，  存储用户的数据是为了之后做其他业务
    if "记住" in last_message:
        # 简单提取“记住”后面的内容（实际生产可用LLM提取）
        memory_content = last_message.replace("记住", "").strip("：: ")
        runtime.store.put(namespace, str(uuid.uuid4()), {"data": memory_content})
        print(f"--- [系统日志] 已存入长期记忆: {memory_content} ---")

    return {"messages": result}


# 定义图
DB_URI = "redis://localhost:6379"

with RedisStore.from_conn_string(DB_URI) as store, \
        RedisSaver.from_conn_string(DB_URI) as checkpointer:
    # 初始化连接对象
    store.setup()
    checkpointer.setup()

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    graph = builder.compile(checkpointer=checkpointer, store=store)

    # --- 4. 交互循环 ---
    current_thread_id = "1"
    current_user_id = "user_v1"

    print("=== LangGraph 交互系统 ===")
    print("指令说明: 输入 'switch' 切换会话, 'exit' 退出程序")

    while True:
        prompt = f"\n[当前线程: {current_thread_id}] 用户: "
        user_input = input(prompt).strip()

        if user_input.lower() == 'exit':
            break

        if user_input.lower() == 'switch':
            new_id = input("请输入新的 Thread ID (例如 1, 2, 3): ")
            current_thread_id = new_id
            print(f"--- 已切换到线程 {current_thread_id} ---")
            continue

        if not user_input:
            continue

        # 构建配置
        config = {
            "configurable": {
                "thread_id": current_thread_id,
                "user_id": current_user_id,
            }
        }

        # 执行流式输出（使用 values 模式）
        # 注意：由于我们要手动输入，每次流只传入当前这一条消息
        for chunk in graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                context=Context(user_id=current_user_id),
                stream_mode="messages",
                version="v2"
        ):
            if chunk["type"] == "messages":
                result, metadata = chunk["data"]
                print(result.content, end="", flush=True)

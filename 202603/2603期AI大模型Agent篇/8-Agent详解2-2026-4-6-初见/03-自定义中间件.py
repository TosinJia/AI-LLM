from langchain.agents import create_agent, AgentState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from langchain.agents.middleware import (
    before_agent,
    before_model,
    after_model,
    after_agent,
    wrap_model_call,
    wrap_tool_call,
    ModelRequest,
    ModelResponse,
)
from dataclasses import dataclass
from pydantic.dataclasses import dataclass
from typing import Callable
import json
import re
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

model = "MiniMax-M2.1"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")
# 初始化qwen模型
llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=api_base_url,
    temperature=0.1
)
# dataclass底层基于pydantic
@dataclass(frozen=True)   # 自动创建init、eq、repr,frozen=True 的时候初始化属性值之后就不能修改
class Context:
    user_id: str  # 用户id
    user_permissions: str  # 用户权限

# 创建json格式验证
def repair_json_string(raw_str: str) -> str:
    # 1. 去掉 Markdown 的代码块标签
    raw_str = re.sub(r"```json\s*|```", "", raw_str).strip()

    # 2. 修复最常见的：对象或数组末尾多余的逗号
    # 匹配: , 后面跟着 } 或 ]
    raw_str = re.sub(r",\s*([}\]])", r"\1", raw_str)

    # 3. 简单的引号补全（针对属性名漏掉引号的情况）
    # 匹配: {后面或逗号后面 没写引号的 key
    raw_str = re.sub(r"([{,]\s*)([a-zA-Z0-9_]+)(\s*:)", r'\1"\2"\3', raw_str)

    return raw_str


# 定义before_agent自定义中间件   在智能体执行之前
# 1.根据用户权限限制用户能访问的工具或者一些资源2.对于用户的问题进行校验
# AgentState获取用户的输入
@before_agent(can_jump_to="end")
def manage_human_message_before_agent(state: AgentState, runtime: Runtime[Context]):
    user_massage = ""
    # 找到用户最新的一个提问
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            user_massage = message.content
            break

    # 定义敏感词表
    sensitive_words = ["TM", "TMD", "CNM", "挂了", "垃圾"]
    if any(word in user_massage.upper() for word in sensitive_words):
        # 判断用户问题中是否又敏感词，如果又直接结束Agent
        return {
            "messages": [AIMessage(content="检测到不当言论，请文明交流")],
            "jump_to": "end"  # 当前对话直接就结束

        }
    # 进行用户权限的控制
    # 获取配置信息
    user_id  =runtime.context.user_id
    user_permissions = runtime.context.user_permissions
    if user_permissions == "Vip":
        # 可以使用所有工具
        print("可以使用所有工具")
    else:
        # 只能使用部分工具
        print("只能使用部分工具")

    return None

# 在模型调用之前：1.动态提示词  2.上下文压缩
@before_model
def inject_user_context(state: AgentState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    # 模拟从数据库获取用户偏好
    user_preference = "用户喜欢幽默的角色"

    # 创建一个新的系统消息，注入用户信息
    context_message = SystemMessage(content=f"用户偏好：{user_preference}。请根据此偏好回答问题。")

    return {
        "messages": [context_message] + state["messages"]
    }

@after_model
def fix_json_structure(state: AgentState, runtime: Runtime[Context]):
    # 1. 获取模型最后一条回复
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return
    # 这里我们模拟模型返回错误的json
    # raw_content = last_message.content
    raw_content = """
        ```json
        {
          "user_id": "123",
          "action": "send_package",
          "items": ["book", "pen"],
        }
    """
    print(f"开始进行json格式修复:{raw_content}")
    try:
        # 尝试直接解析，如果成功说明不需要修复
        json.loads(raw_content)
    except json.JSONDecodeError:
        # 2. 如果解析失败，执行修复逻辑
        fixed_content = repair_json_string(raw_content)
        print(f"json格式修复完成：{fixed_content}")

        try:
            # 再次验证修复结果
            json.loads(fixed_content)
            # 3. 【关键】写回消息对象
            last_message.content = fixed_content
            # 也可以记录一个标记位说明发生过修正
            last_message.additional_kwargs["is_fixed"] = True
        except Exception:
            # 如果修复后还是不行，可以抛出异常触发重试，或记录错误
            pass

    return {"messages": state["messages"]}  # 在中间件中直接返回messages：直接替换之前的内容


@wrap_model_call
def smart_model_wrapper(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    user_preference = "用户喜欢二次元"
    context_msg = SystemMessage(content=f"用户偏好：{user_preference}。请根据此偏好回答问题。")
    # 通过override去覆盖一个新请求(只是一个临时指令，并不会添加到历史对话中)
    new_request = request.override(
        system_message=context_msg
    )
    # 检查redis缓存中是否又相同的请求，有就直接返回
    # 调用真正执行模型的内容
    response = handler(new_request)  # 真正调用模型的地方（重点）
    # 模拟 after_model 的逻辑：结构化修正 (可选)
    # 如果handler(new_request)报错了，可以切换到备用模型
    return response


agent = create_agent(
    model=llm,
    middleware=[manage_human_message_before_agent, inject_user_context, fix_json_structure],
)
result = agent.invoke({"messages": [HumanMessage(content="你好我是初见")]},
                      context=Context(user_id="1", user_permissions="Vip"))
print(result)

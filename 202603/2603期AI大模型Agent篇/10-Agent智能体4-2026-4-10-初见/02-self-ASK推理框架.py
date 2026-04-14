from typing import Dict, Any, Callable, Optional
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest, hook_config
from langchain.agents.middleware.types import StateT
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from dotenv import load_dotenv
import os

from langgraph.typing import ContextT

# 加载环境变量
load_dotenv()
# 初始化模型
model = "deepseek-chat"
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
# 初始化千问模型
llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)

# 设定模型结束词
stop_words = [
    "\nIntermediate answer:",
    "Intermediate answer:",
    "\nIntermediate answer：",  # 防御中文冒号
    "Intermediate answer："

]
llm_with_stop = llm.bind(stop=stop_words)
# self-ask提示词定义
SELF_ASK_SYSTEM_PROMPT = """
### 角色设定
你是一个专业的“搜索链（Self-Ask）”推理专家。你的唯一职责是根据用户的问题，通过**分步拆解**的方式引导搜索。

### 核心限制 (绝对禁止)
1. **严禁越权**：禁止生成任何以 "Intermediate answer:" 开头的行。这部分内容必须由外部搜索工具提供。
2. **严禁多步**：每次仅允许输出 **一个** "Follow up:" 问题。输出完成后必须立即停止，不得继续生成。
3. **严禁幻觉**：如果模型已知答案，但在 Self-Ask 流程中，仍需按照拆解步骤确认信息。
4. **严禁合并**：如果问题简单，但是还是需要进行问题的拆解，不能合并多个问题

### 运行流程
- **判断**：首先判断问题是否需要拆解。
- **输出**：若需要，输出 "Are follow up questions needed here: Yes" 并换行提供第一个 "Follow up:"。
- **暂停**：在Follow up:后提出问题，然后立即切断生成。
- **工具调用**： 直接调用Intermediate_Answer搜索工具去进行问题的搜索。
- **接力**：当外部工具提供 "Intermediate Answer:" 后，你再根据已知信息决定是提出下一个 "Follow up:" 还是输出最终答案。

### 格式规范
Question: [用户输入]
Are follow up questions needed here: [Yes/No]
Follow up: [仅限一个具体的搜索问题]
Intermediate answer: (等待Intermediate_Answer工具搜索完成)
...
So the final answer is: [基于中间答案汇总的结论]

### 示例
Question: 马斯克的第二家公司的现任 CEO 是谁？
Are follow up questions needed here: Yes
Follow up: 马斯克创办或参与的第二家公司是什么？
Intermediate answer: Zip2 (第一家) 之后是 X.com (后来的 PayPal)。 (此时必须调用 intermediate_answer 工具，输入: "马斯克创办或参与的第二家公司是什么？")
Follow up: X.com (PayPal) 的现任 CEO 是谁？
Intermediate answer: Dan Schulman (注：此处以实际搜索结果为准)。
So the final answer is: Dan Schulman。

### 待处理任务
Question: {input}
"""

# 定义工具
search_tool = TavilySearch(name="Intermediate_Answer",
                           description="当需要回答 'Follow up' 中的子问题的时候，必须调用此工具。输入应该是子问题的文本")


# 创建self-ask中间件
class SelfAskMiddleware(AgentMiddleware):
    """
        该中间件负责拦截模型的文本输出，如果发现'Follow up'模式，则引导去进行搜索
    """

    @hook_config(can_jump_to=["model", "end"])
    def after_model(
            self, state: StateT, runtime: Runtime
    ) -> dict[str, Any] | None:
        """
            检测模型回复，分三种情况
            1.包含'Follow up'  ->记录子问题，进行搜索工具使用
            2.包含 So the final answer is  ->推理完成了，跳转到结束
            3.两个都不包含 ->需要重试
        """
        # 获取模型的回答
        last_message = state["messages"][-1] if state["messages"] else None
        content = last_message.content
        if not isinstance(last_message, AIMessage):
            return None
        print(f"模型的回复：{content}")

        # 1.包含'Follow up'  ->记录子问题，进行搜索工具使用
        if "Follow up" in content:
            # 获取Follow up所在行的数据
            final_line = next((line for line in content.splitlines() if "Follow up" in line), "")
            print(f"检测到子问题：{final_line}")
            return None
        # 2.包含 So the final answer is  ->推理完成了，跳转到结束
        if "So the final answer is" in content:
            final_line = next((line for line in content.splitlines() if "So the final answer is" in line), "")
            print(f"智能体推理完成：{final_line}")
            return {
                "jump_to": "end"
            }
        #  3.两个都不包含 ->需要重试
        print(f"模型并没有按照指定格式进行输出，注入纠正提示词")
        ai_message = AIMessage(content="请严格按照 Self-Ask 格式继续推理。\n"
                                       "如果还需要查询信息，请写：Follow up: <子问题>\n"
                                       "如果已有足够信息，请写：So the final answer is: <答案>")

        return {
            "messages": state["messages"] + [ai_message],
            "jump_to": "model"
        }

    def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """
            将工具返回的内容包装成  Intermediate answer: 格式
        """
        # 获取工具的输出
        tool_result = handler(request)
        format_result = f"Intermediate answer: {tool_result}"
        print(f"工具的输出：{format_result}")

        return ToolMessage(content=format_result, tool_call_id=request.tool_call["id"])


# 创建Agent
agent = create_agent(model=llm_with_stop,
                     tools=[search_tool],
                     system_prompt=SELF_ASK_SYSTEM_PROMPT,
                     middleware=[SelfAskMiddleware()])
messages = {"messages": [HumanMessage(content="NBA 历史上总得分第一的球员，他职业生涯效力的第一支球队主场在哪个城市？")]}
response = agent.invoke(messages)
print(response)

from typing import Any, Callable
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()
# 初始化模型
model = "qwen3.5-plus"
api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL")
# 初始化千问模型
llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


# 定义思考提示词
THINKING_PROMPT = """
### 任务目标
你是一个全能的专家型助手。请解决以下问题：{question}

### 指令
在给出最终答案之前，你必须先9在 <thinking> 标签内进行深度的内部推理。

### 思考框架 (Thinking Framework)
请务必按照以下四个维度进行拆解：
1. **核心解析**：该问题的本质是什么？有哪些隐藏约束？
2. **逻辑推演**：逐步推导解决该问题的路径，列出关键的判断点。
3. **潜在陷阱**：如果不仔细，最容易在哪个地方出错？（例如：单位、逻辑漏洞、语气不当等）。
4. **结论预演**：在输出前，先简要确定最终结论的核心要点。

### 约束
- 所有的推理过程必须包裹在 <thinking>...</thinking> 标签内。
- 严禁在思考阶段直接结束，必须在标签外给出清晰、完整的最终答案。

"""
# 定义反思提示词
REFLECTION_PROMPT = """
### 任务背景
用户提出了问题：{question}
你之前的初始答案是：{initial_answer}

### 任务指令
请对上述初始答案进行严苛的自我审查，并在 <reflection> 标签内记录你的反思。

### 审查清单 (Review Checklist)
请针对以下维度进行“找茬”：
1. **准确性检查**：答案中的事实、计算、逻辑推导是否 100% 正确？
2. **完整性检查**：是否遗漏了用户问题中的任何子需求或背景条件？
3. **简洁与清晰度**：是否存在啰嗦的废话？回答的结构是否易于理解？
4. **改进空间**：如果有机会做得更好，你应该如何调整表述或内容？

### 输出要求
- 反思过程必须包裹在 <reflection>...</reflection> 标签内。
- 在标签之后，请整合反思意见，输出一个经过全面优化的最终答案。
"""


# 创建自定义思考反思中间件
@wrap_model_call
def thinking_reflection_wrap_model_call(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]):
    """进行思考和反思的代码编写"""
    # --- 1. 思考阶段 (Thinking) ---
    # 修改系统提示词，强制模型输出 <thinking> 标签
    thinking_request = request.override(
        system_message=SystemMessage(content=THINKING_PROMPT)
    )
    # 第一次调用模型：获取【思考 + 初版答案】
    initial_response = handler(thinking_request)
    initial_content = initial_response.result[-1].content
    print("思考阶段：", initial_content)

    # --- 2. 反思阶段 (Reflection) ---
    # 构造反思的 Prompt
    # 我们可以把初版答案喂回给模型，让它自我检查
    reflection_input = REFLECTION_PROMPT.format(
        question=request.messages[0].content,  # 获取用户提问
        initial_answer=initial_content
    )

    # 第二次调用模型：进行反思并改进   （重点）  测试，可以用好一点模型
    # 注意：这里我们通常会创建一个新的 Request，或者修改 Message 历史
    reflection_request = request.override(
        messages=[HumanMessage(content=reflection_input)],
        system_message=SystemMessage(content=REFLECTION_PROMPT)
    )

    final_response = handler(reflection_request)
    final_response.result[-1].response_metadata = {
        "original_thinking": initial_content
    }
    return final_response


agent = create_agent(model=llm,
                     system_prompt="你是一个乐于助人的助手",
                     middleware=[thinking_reflection_wrap_model_call]
                     )

response = agent.invoke({"messages": [HumanMessage(
    content="我要在 4 平米的阳台上实现：1. 洗衣机和烘干机叠放；2. 一个洗手池；3. 养 5 盆花；4. 还要放一个折叠躺椅看书。请给出具体的空间布局方案。")]})
print("最终答案：", response)
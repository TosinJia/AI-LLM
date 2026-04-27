from typing import Literal
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# ==============================================================================
# 数据定义 (模拟数据库)
# ==============================================================================

# 知识库: 技术问题的解决方案映射
# Key: 问题关键词 (小写), Value: 解决方案
KNOWLEDGE_BASE = {
    "login": "请清除浏览器缓存并重新登录，或重置密码。",
    "payment": "请检查银行卡余额，确认交易状态，或联系银行。",
    "bug": "我们已记录此问题，技术团队将在24小时内处理。",
    "network": "请检查网络连接，或尝试切换网络环境。",
    "performance": "建议清理缓存、重启应用或检查系统资源使用情况。"
}

# 产品列表: 产品ID -> 产品信息
# 包含名称、价格、功能列表
PRODUCTS = {
    "basic": {"name": "基础版", "price": 99, "features": ["基础功能", "邮件支持"]},
    "pro": {"name": "专业版", "price": 299, "features": ["高级功能", "优先支持", "API访问"]},
    "enterprise": {"name": "企业版", "price": 999, "features": ["企业功能", "专属客服", "定制开发"]}
}

# 用户数据库: 用户ID -> 用户信息
# 包含套餐、状态、支持级别、余额
USER_DATABASE = {
    "user123": {"plan": "pro", "status": "active", "support_level": "premium", "balance": 500},
    "user456": {"plan": "basic", "status": "active", "support_level": "standard", "balance": 100}
}

# ==============================================================================
# LLM 初始化
# ==============================================================================

# 创建 Qwen 模型实例
# 使用阿里云 DashScope API 兼容 OpenAI 接口
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="kimi-k2.5",
    temperature=0.1  # 低温度，保持回答确定性
)

# ==============================================================================
# 工具定义 (使用 @tool 装饰器)
# ==============================================================================

# ----------------------------------------------------------------------
# 技术支持工具 (tech_agent 使用)
# ----------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索技术知识库，查找解决方案

    Args:
        query: 用户的问题描述

    Returns:
        str: 从知识库中找到的解决方案，或提示创建工单
    """
    print(f"[Tech] 搜索知识库: {query}")

    # 将查询转为小写进行匹配
    query_lower = query.lower()
    results = []

    # 遍历知识库，查找匹配的问题
    for issue, solution in KNOWLEDGE_BASE.items():
        if issue in query_lower:
            results.append(f"{issue}: {solution}")

    # 如果找到匹配项，返回所有解决方案
    if results:
        return "找到以下解决方案:\n" + "\n".join(results)

    # 未找到匹配，返回创建工单提示
    return "未在知识库中找到相关解决方案，建议创建技术工单进行人工处理。"

# ----------------------------------------------------------------------
# 技术支持工具: 创建工单 (tech_agent 使用)
# ----------------------------------------------------------------------

@tool
def create_support_ticket(issue_description: str, priority: str = "normal") -> str:
    """
    创建技术支持工单

    Args:
        issue_description: 问题描述
        priority: 优先级 (normal/high/urgent)

    Returns:
        str: 工单创建成功信息，包含工单ID
    """
    import uuid

    # 生成唯一工单ID: TICKET-XXXXXXXX 格式
    ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"
    print(f"[Tech] 创建工单: {ticket_id}")

    return f"已创建支持工单: {ticket_id}\n问题描述: {issue_description}\n优先级: {priority}\n我们的技术团队将在24小时内处理您的问题。"

# ----------------------------------------------------------------------
# 销售工具: 获取产品信息 (sales_agent 使用)
# ----------------------------------------------------------------------

@tool
def get_product_info(product_query: str = "") -> str:
    """
    获取产品信息和价格

    Args:
        product_query: 产品查询词（可选），支持产品ID或名称

    Returns:
        str: 产品信息列表，包含价格和功能
    """
    print(f"[Sales] 查询产品: {product_query or '全部'}")

    # 如果没有指定查询词，返回所有产品
    if not product_query:
        result = "我们的产品线包括:\n\n"
        for key, product in PRODUCTS.items():
            result += f"**{product['name']}** - ¥{product['price']}/月\n"
            result += f"功能: {', '.join(product['features'])}\n\n"
        return result

    # 根据查询词查找匹配的产品
    query_lower = product_query.lower()
    for key, product in PRODUCTS.items():
        if key in query_lower or product["name"] in query_lower:
            return (
                f"**{product['name']}**\n"
                f"价格: ¥{product['price']}/月\n"
                f"功能: {', '.join(product['features'])}"
            )

    return f"未找到关于'{product_query}'的产品信息。请查看我们的完整产品列表。"

# ----------------------------------------------------------------------
# 销售工具: 计算升级费用 (sales_agent 使用)
# ----------------------------------------------------------------------

@tool
def calculate_upgrade_cost(current_plan: str, target_plan: str) -> str:
    """
    计算升级费用

    Args:
        current_plan: 当前套餐ID
        target_plan: 目标套餐ID

    Returns:
        str: 升级费用计算结果，包含新增功能列表
    """
    print(f"[Sales] 计算升级: {current_plan} -> {target_plan}")

    # 验证套餐ID有效性
    if current_plan not in PRODUCTS or target_plan not in PRODUCTS:
        return "无效的套餐类型。请检查套餐名称。"

    # 获取当前和目标套餐的价格
    current_price = PRODUCTS[current_plan]["price"]
    target_price = PRODUCTS[target_plan]["price"]

    # 如果目标价格不高于当前价格，无需升级费用
    if target_price <= current_price:
        return (
            f"目标套餐 ({PRODUCTS[target_plan]['name']}) "
            f"价格不高于当前套餐 ({PRODUCTS[current_plan]['name']})，无需升级费用。"
        )

    # 计算升级费用
    upgrade_cost = target_price - current_price

    # 计算新增功能
    new_features = set(PRODUCTS[target_plan]["features"]) - set(PRODUCTS[current_plan]["features"])

    return (
        f"升级费用计算:\n"
        f"当前套餐: {PRODUCTS[current_plan]['name']} (¥{current_price}/月)\n"
        f"目标套餐: {PRODUCTS[target_plan]['name']} (¥{target_price}/月)\n"
        f"升级费用: ¥{upgrade_cost}/月\n\n"
        f"新增功能: {', '.join(new_features)}"
    )

# ----------------------------------------------------------------------
# 管理工具: 查询账户信息 (admin_agent 使用)
# ----------------------------------------------------------------------

@tool
def get_user_account_info(user_id: str) -> str:
    """
    查询用户账户信息

    Args:
        user_id: 用户ID

    Returns:
        str: 用户账户详细信息
    """
    print(f"[Admin] 查询账户: {user_id}")

    # 验证用户ID是否提供
    if not user_id:
        return "请提供您的用户ID以查询账户信息。"

    # 从数据库查找用户
    if user_id in USER_DATABASE:
        user_info = USER_DATABASE[user_id]
        return (
            f"账户信息:\n"
            f"用户ID: {user_id}\n"
            f"当前套餐: {user_info['plan']}\n"
            f"账户状态: {user_info['status']}\n"
            f"支持级别: {user_info['support_level']}\n"
            f"账户余额: ¥{user_info['balance']}"
        )

    return f"未找到用户ID '{user_id}' 的账户信息。"

# ----------------------------------------------------------------------
# 管理工具: 处理退款请求 (admin_agent 使用)
# ----------------------------------------------------------------------

@tool
def process_refund_request(user_id: str, reason: str) -> str:
    """
    处理退款请求

    Args:
        user_id: 用户ID
        reason: 退款原因

    Returns:
        str: 退款申请结果
    """
    print(f"[Admin] 处理退款: {user_id}")

    # 验证用户ID有效性
    if not user_id or user_id not in USER_DATABASE:
        return "请提供有效的用户ID以处理退款请求。"

    user_info = USER_DATABASE[user_id]

    # 检查账户状态
    if user_info["status"] != "active":
        return "只有活跃账户才能申请退款。"

    # 计算退款金额（按月费计算）
    refund_amount = PRODUCTS[user_info["plan"]]["price"]

    return (
        f"退款申请已提交:\n"
        f"用户ID: {user_id}\n"
        f"退款原因: {reason}\n"
        f"退款金额: ¥{refund_amount}\n"
        f"处理时间: 3-5个工作日\n"
        f"退款将原路返回到您的支付账户。"
    )

# ==============================================================================
# 创建子图 (每个子Agent一个子图)
# ==============================================================================

# ----------------------------------------------------------------------
# 子图1: 技术支持 Agent
# 负责处理: 报错、bug、故障、登录问题、网络问题等
# ----------------------------------------------------------------------

def create_tech_agent_subgraph():
    """
    创建技术支持Agent子图

    子图结构:
        START -> tech_model -> (tools_condition) -> tech_tools -> tech_model -> END
                                    |
                                    v
                                   END (无工具调用时)

    工具:
        - search_knowledge_base: 搜索知识库
        - create_support_ticket: 创建工单

    Returns:
        Compiled graph: 编译后的子图，可被主图调用
    """

    # 定义该子图使用的工具列表
    tech_tools = [search_knowledge_base, create_support_ticket]

    # 定义 LLM 调用节点
    # 功能: 调用 LLM，让 LLM 决定是否需要调用工具
    def tech_model_node(state: MessagesState):
        """
        技术Agent的模型节点

        Args:
            state: 包含消息历史的状态

        Returns:
            dict: 更新后的状态，包含 LLM 响应
        """
        print("[Tech] LLM 决策...")

        # 系统提示词：指导 LLM 如何使用工具
        system_prompt = """你是一个技术支持助手。

        工具使用规则：
        1. 首先使用 search_knowledge_base 搜索知识库，查找解决方案
        2. 如果知识库找到了答案，直接返回给用户
        3. 如果知识库找不到答案（返回"未在知识库中找到..."），则使用 create_support_ticket 创建工单

        回答要求：
        - 如果知识库有答案，简明扼要地告诉用户
        - 如果知识库没有答案，创建工单并告知用户工单ID"""
        # 使用系统提示词 + 用户消息
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        ai_message = llm.bind_tools(tech_tools).invoke(messages)
        return {"messages": [ai_message]}

    # 创建状态图
    builder = StateGraph(MessagesState)

    # 添加节点:
    # 1. tech_model: LLM 决策节点
    # 2. tech_tools: 工具执行节点 (由 ToolNode 自动处理工具调用)
    builder.add_node("tech_model", tech_model_node)
    builder.add_node("tech_tools", ToolNode(tech_tools))

    # 添加边:
    # 1. START -> tech_model: 起点到 LLM 节点
    builder.add_edge(START, "tech_model")

    # 2. tech_model -> 条件边: LLM 决定是调用工具还是结束
    #    tools_condition 会检查 LLM 输出是否有 tool_calls
    #    - 如果有: 路由到 tech_tools
    #    - 如果没有: 路由到 END
    builder.add_conditional_edges(
        "tech_model",
        tools_condition,
        {"tools": "tech_tools", END: END}
    )

    # 3. tech_tools -> tech_model: 工具执行完后返回 LLM 形成循环
    builder.add_edge("tech_tools", "tech_model")

    return builder.compile()

# ----------------------------------------------------------------------
# 子图2: 销售 Agent
# 负责处理: 价格咨询、套餐升级、产品信息、购买咨询等
# ----------------------------------------------------------------------

def create_sales_agent_subgraph():
    """
    创建销售Agent子图

    工具:
        - get_product_info: 获取产品信息
        - calculate_upgrade_cost: 计算升级费用

    子图结构与 Tech Agent 相同
    """
    sales_tools = [get_product_info, calculate_upgrade_cost]

    def sales_model_node(state: MessagesState):
        print("[Sales] LLM 决策...")
        ai_message = llm.bind_tools(sales_tools).invoke(state["messages"])
        return {"messages": [ai_message]}

    builder = StateGraph(MessagesState)
    builder.add_node("sales_model", sales_model_node)
    builder.add_node("sales_tools", ToolNode(sales_tools))

    builder.add_edge(START, "sales_model")
    builder.add_conditional_edges(
        "sales_model",
        tools_condition,
        {"tools": "sales_tools", END: END}
    )
    builder.add_edge("sales_tools", "sales_model")

    return builder.compile()

# ----------------------------------------------------------------------
# 子图3: 客户管理 Agent
# 负责处理: 余额查询、账户信息、退款申请等
# ----------------------------------------------------------------------

def create_admin_agent_subgraph():
    """
    创建客户管理Agent子图

    工具:
        - get_user_account_info: 查询账户信息
        - process_refund_request: 处理退款

    子图结构与 Tech Agent 相同
    """
    admin_tools = [get_user_account_info, process_refund_request]

    def admin_model_node(state: MessagesState):
        print("[Admin] LLM 决策...")
        ai_message = llm.bind_tools(admin_tools).invoke(state["messages"])
        return {"messages": [ai_message]}

    builder = StateGraph(MessagesState)
    builder.add_node("admin_model", admin_model_node)
    builder.add_node("admin_tools", ToolNode(admin_tools))

    builder.add_edge(START, "admin_model")
    builder.add_conditional_edges(
        "admin_model",
        tools_condition,
        {"tools": "admin_tools", END: END}
    )
    builder.add_edge("admin_tools", "admin_model")

    return builder.compile()

# ==============================================================================
# 创建主管图 (支持循环协调)
# ==============================================================================

# ----------------------------------------------------------------------
# 主管节点: 负责任务分配和协调
# 使用 LLM 做意图识别，支持多任务循环协调
# ----------------------------------------------------------------------

def create_supervisor_graph():
    """
    创建主管Graph - 协调所有子Agent的任务

    复杂问题示例: "我想了解专业版的价格，另外查一下我的余额"
    - 涉及: 销售问题(价格) + 管理问题(余额)
    - 需要循环调用: supervisor -> sales -> supervisor -> admin -> supervisor -> END

    主管状态 (SupervisorState):
        - messages: 消息列表 (从 MessagesState 继承)
        - pending_tasks: 待处理任务队列 ['tech', 'sales', 'admin']
        - completed_tasks: 已完成任务列表
        - current_agent: 当前正在执行的 Agent

    Returns:
        Compiled graph: 编译后的主管图
    """

    # 先创建三个子图
    tech_subgraph = create_tech_agent_subgraph()
    sales_subgraph = create_sales_agent_subgraph()
    admin_subgraph = create_admin_agent_subgraph()

    # 定义主管状态类型
    # 继承 MessagesState，获得 messages 通道和 add_messages reducer
    class SupervisorState(MessagesState):
        """
        主管状态: 包含消息和任务追踪信息

        Attributes:
            current_agent: 当前执行的 Agent 名称
            pending_tasks: 待处理的任务队列 (关键！用于循环协调)
            completed_tasks: 已完成的任务列表
            next_agent: 下一个要执行的 Agent 名称（由 supervisor_node 设置）
        """
        current_agent: str
        pending_tasks: list[str]
        completed_tasks: list[str]
        next_agent: str | None

    def supervisor_node(state: SupervisorState) -> dict:
        """
        主管节点 - 负责任务识别和分配

        工作流程:
            1. 如果有待处理任务(PendingTasks)，取出第一个任务执行
            2. 如果没有待处理任务（首次），使用 LLM 做意图识别
            3. 将所有识别的任务填入 pending_tasks（除第一个外）
            4. 返回第一个任务名称

        关键改进: 使用 LLM 做意图识别，不再用关键词匹配

        Args:
            state: 当前状态

        Returns:
            dict: 状态更新，包含 pending_tasks 和 next_agent
        """
        pending = state.get("pending_tasks", [])  # 获取待执行的Agent列表
        completed = state.get("completed_tasks", [])  # 已完成的Agent列表
        messages = state["messages"]  # 获取消息
        last_msg = messages[-1] if messages else None  # 消息列表中最后一条消息

        # ========== 情况1: 还有待处理任务，从队列取第一个执行 ==========
        if pending:
            next_task = pending[0]  # 取出队首任务
            remaining = pending[1:]  # 剩余任务
            print(f"\n[Supervisor] 取出待处理任务: {next_task}, 剩余: {remaining}")
            return {
                "pending_tasks": remaining,
                "next_agent": next_task  # 标记下一个要执行的 Agent
            }

        # ========== 情况2: 没有待处理任务，使用提示词做 LLM 意图识别 ==========
        if isinstance(last_msg, HumanMessage):
            print(f"\n[Supervisor] 使用 LLM 进行意图识别...")

            # 意图识别提示词
            routing_prompt = f"""你是意图识别专家。根据用户消息，决定调用哪个 Agent 来处理。

            Agent 类型：
            - tech_agent: 技术支持（报错、bug、故障、登录问题、网络问题、性能问题、技术咨询等）
            - sales_agent: 销售服务（价格咨询、套餐升级、产品信息、购买咨询、升级费用等）
            - admin_agent: 账户管理（查询余额、账户信息、退款申请、账户状态等）

            注意：一个问题可能需要多个 Agent 处理。

            用户消息: {last_msg.content}

            请分析用户消息，返回需要调用的 Agent 列表（最多3个，按处理顺序排列）。
            只返回 Agent 名称，用逗号分隔。例如：tech_agent,sales_agent"""

            # 调用 LLM 获取意图识别结果
            response = llm.invoke(routing_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 解析 LLM 返回的 Agent 列表
            new_tasks = []
            for agent in ["tech_agent", "sales_agent", "admin_agent"]:
                if agent in response_text and agent not in completed:
                    new_tasks.append(agent)

            # 如果 LLM 返回了有效任务
            if new_tasks:
                print(f"[Supervisor] LLM 识别到任务: {new_tasks}")
                next_task = new_tasks[0]
                remaining = new_tasks[1:]
                return {
                    "pending_tasks": remaining,
                    "next_agent": next_task
                }

        # ========== 情况3: 没有新任务 ==========
        print("\n[Supervisor] 所有任务已完成，结束对话")
        return {"pending_tasks": [], "next_agent": None}

    def route_by_pending(state: SupervisorState) -> Literal["tech_agent", "sales_agent", "admin_agent", END]:
        """
        根据 pending_tasks 状态路由到对应的 Agent

        工作流程:
            1. 检查 next_agent 字段（由 supervisor_node 设置）
            2. 返回对应的 Agent 节点或 END

        Args:
            state: 当前状态

        Returns:
            str: 下一个节点的名称
        """
        next_agent = state.get("next_agent")
        if next_agent == "tech_agent":
            return "tech_agent"
        elif next_agent == "sales_agent":
            return "sales_agent"
        elif next_agent == "admin_agent":
            return "admin_agent"
        else:
            return END

    def call_tech_agent(state: SupervisorState) -> SupervisorState:
        """
        调用技术子Agent

        执行流程:
            1. 调用 tech_subgraph 子图
            2. 获取子图执行结果
            3. 返回更新后的状态，让 supervisor 决定下一步

        Args:
            state: 当前状态

        Returns:
            SupervisorState: 更新后的状态
        """
        print("[Supervisor] 调用 Tech 子图...")

        pending = state.get("pending_tasks", [])
        completed = state.get("completed_tasks", [])

        result = tech_subgraph.invoke({"messages": state["messages"]})

        return {
            "messages": result["messages"],
            "pending_tasks": pending,  # 保持 pending 不变，让 supervisor 取出下一个
            "completed_tasks": completed + ["tech"],
            "current_agent": "tech"
        }

    def call_sales_agent(state: SupervisorState) -> SupervisorState:
        """
        调用销售子Agent

        执行流程与 call_tech_agent 相同
        """
        print("[Supervisor] 调用 Sales 子图...")

        pending = state.get("pending_tasks", [])
        completed = state.get("completed_tasks", [])

        result = sales_subgraph.invoke({"messages": state["messages"]})

        return {
            "messages": result["messages"],
            "pending_tasks": pending,
            "completed_tasks": completed + ["sales"],
            "current_agent": "sales"
        }

    def call_admin_agent(state: SupervisorState) -> SupervisorState:
        """
        调用客户管理子Agent

        执行流程与 call_tech_agent 相同
        """
        print("[Supervisor] 调用 Admin 子图...")

        pending = state.get("pending_tasks", [])
        completed = state.get("completed_tasks", [])

        result = admin_subgraph.invoke({"messages": state["messages"]})

        return {
            "messages": result["messages"],
            "pending_tasks": pending,
            "completed_tasks": completed + ["admin"],
            "current_agent": "admin"
        }

    # 创建主管图
    builder = StateGraph(SupervisorState)

    # 添加所有节点
    builder.add_node("supervisor", supervisor_node)  # 主管节点
    builder.add_node("tech_agent", call_tech_agent)  # 技术子Agent
    builder.add_node("sales_agent", call_sales_agent)  # 销售子Agent
    builder.add_node("admin_agent", call_admin_agent)  # 管理子Agent

    # 添加边
    builder.add_edge(START, "supervisor")  # 起点到主管

    # supervisor_node 更新状态后，由 route_by_pending 根据 next_agent 路由
    builder.add_conditional_edges(
        "supervisor",
        route_by_pending,
        {
            "tech_agent": "tech_agent",
            "sales_agent": "sales_agent",
            "admin_agent": "admin_agent",
            END: END
        }
    )

    # ========== 子Agent完成后，如果有待处理任务则返回supervisor继续 ==========
    # 关键：pending_tasks 在 call_xxx_agent 返回时已更新
    builder.add_edge("tech_agent", "supervisor")
    builder.add_edge("sales_agent", "supervisor")
    builder.add_edge("admin_agent", "supervisor")

    return builder.compile()

# ==============================================================================
# 测试运行
# ==============================================================================

def run_examples():
    """
    运行测试示例

    测试场景:
        1. 技术问题: 登录 + 报错
        2. 销售问题: 产品价格咨询
        3. 管理问题: 账户余额查询
        4. 混合问题(循环协调): 价格 + 余额查询 - 需要多次循环协调
    """
    print("=" * 70)
    print("主管架构多智能体系统 (支持循环协调多任务)")
    print("=" * 70)

    # 编译主管图
    graph = create_supervisor_graph()

    # 定义测试用例
    test_cases = [
        # {"message": "我的应用登录有问题，显示error 500"},
        # {"message": "我想了解专业版的价格"},
        # {"message": "查询一下我的余额，用户ID是user123"},
        {"message": "我想了解专业版的价格，另外查一下我的余额，用户ID是user123"},  # 复杂多任务
    ]

    # 遍历测试用例
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"测试案例 {i}")
        print(f"{'=' * 70}")
        print(f"用户: {test_case['message']}")
        print("-" * 50)

        # 构建初始状态
        initial_state = {
            "messages": [HumanMessage(content=test_case["message"])],
            "current_agent": "supervisor",
            "pending_tasks": [],
            "completed_tasks": [],
            "next_agent": None
        }

        try:
            # 执行图
            result = graph.invoke(initial_state)

            # 打印结果
            print("\nAI 回复:")
            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"  {msg.content}")

            # 打印已完成的任务
            print(f"\n已完成任务: {result.get('completed_tasks', [])}")

        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()

# ==============================================================================
# 程序入口
# ==============================================================================

if __name__ == "__main__":
    run_examples()

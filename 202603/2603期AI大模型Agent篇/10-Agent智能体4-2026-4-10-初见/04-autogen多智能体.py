import autogen # pip install pyautogen==0.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
from autogen_config import (
    LLM_CONFIG, AGENT_CONFIG, CHAT_CONFIG, SAMPLE_TASKS,
    TERMINATION_CONFIG
)
import re


# ========================= 第一步：定义智能体（Agent） =========================

def create_agents():
    """创建所有智能体"""

    # 1.1 创建用户代理（User Agent）
    user_proxy = autogen.UserProxyAgent(
        name=AGENT_CONFIG["user_proxy"]["name"],
        system_message=AGENT_CONFIG["user_proxy"]["system_message"],
        human_input_mode=AGENT_CONFIG["user_proxy"]["human_input_mode"],
        max_consecutive_auto_reply=AGENT_CONFIG["user_proxy"]["max_consecutive_auto_reply"],
        code_execution_config=AGENT_CONFIG["user_proxy"]["code_execution_config"],
    )

    # 1.2 创建助手代理（Assistant Agent）
    assistant_agent = autogen.AssistantAgent(
        name=AGENT_CONFIG["assistant"]["name"],
        system_message=AGENT_CONFIG["assistant"]["system_message"],
        llm_config=LLM_CONFIG
    )

    # 1.3 创建监督代理（Monitor Agent）
    monitor_agent = autogen.AssistantAgent(
        name=AGENT_CONFIG["monitor"]["name"],
        system_message=AGENT_CONFIG["monitor"]["system_message"],
        llm_config=LLM_CONFIG,
    )

    # 1.4 创建测试代理（Tester Agent）
    tester_agent = autogen.AssistantAgent(
        name=AGENT_CONFIG["tester"]["name"],
        system_message=AGENT_CONFIG["tester"]["system_message"],
        llm_config=LLM_CONFIG,
    )

    return user_proxy, assistant_agent, monitor_agent, tester_agent


# ========================= 第二步：设计对话策略与任务流程 =========================

class MultiAgentWorkflow:
    def __init__(self, user_proxy, assistant_agent, monitor_agent, tester_agent):
        self.user_proxy = user_proxy
        self.assistant_agent = assistant_agent
        self.monitor_agent = monitor_agent
        self.tester_agent = tester_agent

    def create_group_chat(self):
        """创建群组对话，定义智能体交互规则"""

        # 定义智能体参与列表
        agents = [
            self.user_proxy,
            self.assistant_agent,
            self.monitor_agent,
            self.tester_agent
        ]

        # 设置对话流程规则
        def custom_speaker_selection(last_speaker, group_chat):
            """
            优化后的自定义发言人选择逻辑（状态机模式）
            """
            # 1. 获取对话历史
            messages = group_chat.messages

            # 初始状态：如果没有消息，由 UserProxy 发起任务
            if not messages:
                return self.user_proxy

            # 2. 获取最后一条消息的文本内容并进行标准化处理
            last_message = messages[-1]
            # 使用 .get() 避免 KeyErrors，转小写并去除空格方便后续匹配
            content = last_message.get("content", "").strip()

            # 3. 基于“最后一位发言人”的身份决定“下一位发言人”

            # --- 场景 A: 用户发话 ---
            if last_speaker == self.user_proxy:
                # 用户通常提出需求，下一步固定交给开发助手
                if len(messages) == 1:
                    return self.assistant_agent
                return self.tester_agent

            # --- 场景 B: 开发助手 (Assistant) 发话 ---
            elif last_speaker == self.assistant_agent:
                # 使用正则或字符串检查是否包含代码块
                # 防止助手只说话不写代码，如果没有代码块则打回重写
                if "```python" in content:
                    return self.monitor_agent
                else:
                    print("📢 系统提示：检测到助手未提供代码块，要求其重新生成。")
                    return self.assistant_agent

            # --- 场景 C: 监督代理 (Monitor) 发话 ---
            elif last_speaker == self.monitor_agent:
                # 使用正则匹配标签，提高容错率（支持空格、大小写等）
                pass_match = re.search(r"\[\s*通过\s*]", content)
                fix_match = re.search(r"\[\s*修改\s*]", content)

                if pass_match:
                    print("✅ 审查通过 -> 转交给用户将文件写入本地。")
                    return self.user_proxy
                elif fix_match:
                    print("🔄 审查建议修改 -> 打回给开发助手。")
                    return self.assistant_agent
                else:
                    # 兜底逻辑：如果监督员没给出明确结论，通常默认其指出有问题，打回助手
                    print("⚠️ 审查结论模糊，默认打回助手进行确认。")
                    return self.assistant_agent

            # --- 场景 D: 测试代理 (Tester) 发话 ---
            elif last_speaker == self.tester_agent:
                # 在 AutoGen 中，如果需要结束，通常让 Tester 输出 TERMINATE
                # 然后在 GroupChatManager 的 is_termination_msg 中捕获它。
                # 如果流程需要循环回用户（例如请求新任务），则返回 UserProxy。
                if "TERMINATE" in content.upper():
                    return None  # 返回 None 会触发 Manager 检查是否终止

                # 如果测试发现 Bug，其实也可以在这里加逻辑返回给 Assistant
                if "FAILED" in content.upper() or "错误" in content:
                    print("❌ 测试失败 -> 打回给开发助手修复。")
                    return self.assistant_agent

                return self.user_proxy

            # 4. 最终兜底：如果逻辑跑出预期，默认交还给用户或助手，防止程序卡死
            return self.assistant_agent

        def is_termination_msg(message: dict):
            """
            message 是一个字典，包含 'content', 'name' 等字段
            """
            content = message.get("content")
            if content is None:
                return False

            # 检查关键字
            content = content.lower()
            return any(keyword.lower() in content for keyword in TERMINATION_CONFIG["keywords"])

        # 创建群组对话
        group_chat = autogen.GroupChat(
            agents=agents,
            messages=[],
            max_round=CHAT_CONFIG["max_round"],  # 最大的发言回合数量
            speaker_selection_method=custom_speaker_selection,  # 自定义发言规则
            allow_repeat_speaker=CHAT_CONFIG["allow_repeat_speaker"],  # 是否运行同一个人连续发言
        )

        # 创建群组对话管理器
        manager = autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=LLM_CONFIG,
            system_message=CHAT_CONFIG["manager_system_message"],
            is_termination_msg=is_termination_msg
        )

        return manager


# ========================= 第三步：运行多智能体对话（执行协同任务） =========================

def run_multi_agent_task(task_description: str):
    """启动多智能体协同任务"""

    print("🚀 启动多智能体协同任务...")
    print(f"📋 任务描述: {task_description}")
    print("-" * 50)

    # 创建智能体
    user_proxy, assistant_agent, monitor_agent, tester_agent = create_agents()

    # 创建工作流实例
    workflow = MultiAgentWorkflow(user_proxy, assistant_agent, monitor_agent, tester_agent)

    # 创建群组对话管理器
    manager = workflow.create_group_chat()

    try:
        # 启动对话循环
        result = user_proxy.initiate_chat(
            manager,
            message=f"""
            新任务请求：{task_description}

            请按照以下流程协同完成：
            1. 编程助手：分析需求并编写代码
            2. 代码审查员：审查代码质量和安全性
            3. 测试工程师：编写测试用例并验证功能
            4. 当测试工程师完成之后，请明确给出完成的提示

            请开始执行任务。
            """,
        )

        print("\n" + "=" * 50)
        print("✅ 多智能体协同任务执行完成！")
        print("=" * 50)

        return result

    except Exception as e:
        print(f"任务执行过程中出现错误: {str(e)}")
        return None


# ========================= 示例使用 =========================

if __name__ == "__main__":

    # 选择要执行的任务（可以修改这里来测试不同任务）
    selected_task = SAMPLE_TASKS["task1"]  # 可以改为 task2, task3

    print("🤖 AutoGen多智能体协同系统启动")
    print("=" * 60)

    # 执行任务
    result = run_multi_agent_task(selected_task)

    if result:
        print(f"\n📊 任务执行摘要:")
        print(f"- 总对话轮数: {len(result.chat_history) if hasattr(result, 'chat_history') else '未知'}")
        print(f"- 任务状态: 已完成")
    else:
        print("\n任务执行失败，请检查配置和网络连接")

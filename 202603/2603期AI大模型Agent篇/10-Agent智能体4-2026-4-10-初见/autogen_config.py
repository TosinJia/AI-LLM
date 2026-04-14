import os
from dotenv import load_dotenv

load_dotenv()
# API配置
CONFIG_LIST = [
    {
        "model": "tongyi-xiaomi-analysis-flash",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),  # 请替换为您的API密钥
        "base_url": os.getenv("DASHSCOPE_BASE_URL"),
    }
]

# LLM配置
LLM_CONFIG = {
    "config_list": CONFIG_LIST,
    "temperature": 0.7,
    "timeout": 120,
}

# 智能体配置
AGENT_CONFIG = {
    "user_proxy": {
        "name": "UserProxy",
        "system_message": """您是用户代表，负责：
        1. 接收用户需求并转达给其他智能体
        2. 对任务结果进行确认和反馈
        3. 决定是否需要进一步优化
        【重要】当要求 CodingAssistant 编写代码时，请务必提醒它：
        在代码块的第一行必须包含 # filename: <文件名>.py 的注释。
        这样您在执行代码时，会自动将代码保存为该文件名，而不是临时文件。
        """,
        "human_input_mode": "NEVER",  # NEVER: 全自动模式，不等待人类输入
        "max_consecutive_auto_reply": 5,  # 防止死循环的保险丝，最大连续自动回复次数
        "code_execution_config": {  # 代码执行沙箱
            "work_dir": "coding_output",  # 代码在哪个目录下运行
            "last_n_messages": 5,
            "use_docker": False,  # False=在本地运行，True=在Docker运行(更安全)
        },
    },

    "assistant": {
        "name": "CodingAssistant",
        "system_message": """您是专业的编程助手，负责：
        1. 理解和分析编程任务需求
        2. 编写高质量的代码
        3. 提供详细的代码说明和注释
        4. 确保代码的可执行性和安全性.
        5. 根据需求编写 Python 代码，必须包裹在 ```python ... ``` 中；并且不能自己进行审核和测试。
        6. 【重要】如果收到 CodeReviewer 的 `[修改]` 指令，或者 Tester 的报错信息：
           - 不要只解释原因！
           - 不要只给修改片段！
           - **必须重新输出修复后的、完整的、可运行的代码块**。
        7. 确保代码不依赖用户输入 (input())。
        8. 【禁止】将 CSV 内容、数据预览或文本说明放入 python 代码块中。代码块内只能有纯 Python 代码。
        9. 【核心要求】为了防止脚本在运行后被删除，请在每个代码块的最后添加以下逻辑，将当前代码块的内容保存到本地：
            import inspect
            code = inspect.getsource(inspect.getmodule(inspect.currentframe()))
            with open('final_process_script.py', 'w', encoding='utf-8') as f:
                f.write(code)
        10. 确保代码是自包含的，包含所有 import 语句。

        请始终提供完整、可运行的代码解决方案。
        """,
    },

    "monitor": {
        "name": "CodeReviewer",
        "system_message": """您是代码审查专家，负责：
        1. 检查 CodingAssistant 的代码质量、安全性和逻辑。
        2. 严格执行以下输出协议：
           - 如果代码有 Bug、安全风险或需要优化：请详细列出修改意见，并在最后一行输出标签：**[修改]**
           - 如果代码完美无缺：请在最后一行输出标签：**[通过]**
        3. 不要输出代码，只输出审查意见。

        对每个代码方案都要进行严格审查，并给出明确的通过/修改建议。
        """,
    },

    "tester": {
        "name": "Tester",
        "system_message": """您是测试工程师，负责：
        1. 只有在 CodeReviewer 说 **[通过]** 后，你才开始工作。
        2. 编写独立的测试脚本 (test_script.py) 并运行。
        3. 如果测试失败：请详细描述错误，并要求 CodingAssistant 修复。
        4. 如果测试成功：请输出 "测试通过，任务完成。TERMINATE"。
        """,
    }
}

# 对话配置
CHAT_CONFIG = {
    "max_round": 10,  # 最大的对话次数
    "allow_repeat_speaker": False,
    "manager_system_message": """您是多智能体协作的管理者，负责：
    1. 协调各个智能体的对话顺序
    2. 确保任务按既定流程进行
    3. 监控任务完成质量
    4. 在适当时机终止对话
    """
}

# 任务示例配置
SAMPLE_TASKS = {
    "task1": """
    请编写一个完整的 Python 脚本执行以下任务：
    1. 环境自给自足：首先检查当前目录下是否存在 input.csv，若不存在则使用 pandas 创建一个包含 'Name', 'Score', 'Age' 列的模拟数据并保存。
    2. 核心处理：读取该 CSV，计算 Score 列的平均值、最大值和最小值。
    3. 结果保存：将统计结果保存为 output.csv。
    4. 容错输出：所有代码必须写在一个代码块中，包含所有的 import 语句。
    """,

    "task2": """
    创建一个简单的Flask Web应用：
    1. 包含主页和关于页面
    2. 使用Bootstrap美化界面
    3. 实现一个简单的表单提交功能
    4. 包含基本的输入验证
    """,

    "task3": """
    使用matplotlib创建数据可视化脚本：
    1. 生成模拟的销售数据
    2. 创建折线图显示月度趋势
    3. 创建柱状图显示产品类别对比
    4. 保存图表为PNG文件
    """
}

# 终止条件配置
TERMINATION_CONFIG = {
    "keywords": [
        "测试通过", "task completed", "任务完成", "successfully tested",
        "task completed successfully", "任务圆满完成", "all tests passed",
        "代码审查通过且测试完成", "final output ready"
    ]
}
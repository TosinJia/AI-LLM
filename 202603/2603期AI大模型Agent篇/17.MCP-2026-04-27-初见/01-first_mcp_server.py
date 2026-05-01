from fastmcp import FastMCP # pip install fastmcp
from typing import Annotated
from pydantic import Field

mcp = FastMCP(name="first_mcp_server")


# 定义工具
@mcp.tool()
async def add(a: int, b: int) -> int:
    """
       计算两数之和的工具
    :param a: 参数1
    :param b: 参数2
    :return:  返回两个参数之和
    """
    return a + b


@mcp.tool()
async def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    计算 BMI 指数并返回健康建议。
    Args:
        weight_kg: 体重 (公斤)
        height_m: 身高 (米)
    """
    if height_m <= 0:
        return "错误：身高必须大于 0"

    bmi = weight_kg / (height_m ** 2)
    result = f"BMI 指数: {bmi:.2f}"

    if bmi < 18.5:
        return f"{result} (偏瘦)"
    elif bmi < 24.9:
        return f"{result} (正常)"
    else:
        return f"{result} (偏胖)"


# 模拟一份静态数据
HEALTH_GUIDELINES = """
1. 每天保持 8 小时睡眠。
2. 多吃蔬菜水果，少吃糖。
3. 每周至少运动 150 分钟。
"""


# 定义资源：使用自定义的 URI 协议头 resource：代表分类，health-guidelines具体路径
@mcp.resource("resource://health-guidelines", mime_type="text/plain")
async def get_health_guidelines() -> str:
    """获取通用的健康生活指南"""
    return HEALTH_GUIDELINES


# 定义提示词模板
@mcp.prompt()
async def analyze_my_health(
        name: Annotated[str, Field(description="用户的姓名或昵称")],
        weight: Annotated[float, Field(description="体重，单位：公斤 (kg)")],
        height: Annotated[float, Field(description="身高，单位：米 (m)")]
) -> str:
    """创建一个让 AI 分析个人健康状况的提示词"""
    return f"""
    请扮演一位专业的健康顾问。
    用户 {name} 的体重是 {weight}kg，身高是 {height}m。

    请执行以下步骤：
    1. 使用 'calculate_bmi' 工具计算他的 BMI。
    2. 读取 'health://guidelines' 资源，结合指南给出建议。
    """


# ====================================
# 方法2：使用 Low-Level API (底层原理)
# 这是传统写法。你需要手动构建 JSON Schema，手动解析参数，手动分发路由。虽然繁琐，但能让你看清协议底层到底在传什么。
# 为什么需要学习底层写法呢？
# 仅当你需要在运行时动态生成工具（而不是写死函数）或精细控制协议生命周期（如自定义鉴权、复杂订阅）时，才需要底层写法提供的极致控制权。
# ====================================
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from pydantic import AnyUrl

# 1. 初始化服务器
server = Server("low-level-calculator")


# 2. 手动定义工具列表 (Schema)
# 必须显式写出 JSON 结构，非常容易写错
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="add",
            description="计算两个数字的和",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数字"},
                    "b": {"type": "number", "description": "第二个数字"},
                },
                "required": ["a", "b"],
            },
        ),
        types.Tool(
            name="calculate_bmi",
            description="计算 BMI 指数",
            inputSchema={
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number", "description": "体重(kg)"},
                    "height_m": {"type": "number", "description": "身高(m)"},
                },
                "required": ["weight_kg", "height_m"],
            },
        ),
    ]


# 3. 手动处理调用逻辑 (路由)
@server.call_tool()
async def handle_call_tool(
        name: str,
        arguments: dict | None
) -> list[types.TextContent]:
    if name == "add":
        # 需要手动从字典中提取参数
        a = arguments.get("a")
        b = arguments.get("b")
        result = a + b
        return [types.TextContent(type="text", text=str(result))]

    elif name == "calculate_bmi":
        weight = arguments.get("weight_kg")
        height = arguments.get("height_m")

        if height <= 0:
            return [types.TextContent(type="text", text="错误：身高必须大于 0")]

        bmi = weight / (height ** 2)
        return [types.TextContent(type="text", text=f"BMI: {bmi:.2f}")]

    else:
        raise ValueError(f"未知工具: {name}")


# 手动定义Resources - 手动处理 URI 路由
@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="resource://health-guidelines",
            name="健康指南",
            description="通用的健康生活建议文本",
            mimeType="text/plain"
        )
    ]


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    # 必须手动判断 URI 是否匹配
    if uri.unicode_string() == "resource://health-guidelines":
        return HEALTH_GUIDELINES
    raise ValueError(f"未找到资1111源: {uri} type:{type(uri)}")


# 手动定义Prompts (提示词模板) - 手动构建消息结构
@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="analyze_my_health",
            description="分析用户的健康状况",
            arguments=[
                types.PromptArgument(name="name", description="用户姓名", required=True),
                types.PromptArgument(name="weight", description="体重(kg)", required=True),
                types.PromptArgument(name="height", description="身高(m)", required=True),
            ]
        )
    ]


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    if name == "analyze_my_health":
        user_name = arguments.get("name")
        w = arguments.get("weight")
        h = arguments.get("height")

        # 返回标准的消息结构
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"我是 {user_name}，体重{w}，身高{h}。请帮我计算BMI并根据健康指南给出建议。"
                    )
                )
            ]
        )
    raise ValueError(f"未知提示词: {name}")


# 4. 启动循环
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    # mcp.run()

    asyncio.run(main())

import asyncio
import sys
from fastmcp import Client

# ==========================================
# 配置部分
# ==========================================
# 这里指定要启动的 Server 脚本路径
SERVER_SCRIPT = "01-first_mcp_server.py"    # client调用server
SERVER_SCRIPT = "03-mcp_traffic.py"         # client通过traffic调用server

# 获取当前 python 解释器路径   skills + cli    shell命令
python_path = sys.executable
client = Client(SERVER_SCRIPT)

async def main():
    async with client:
        await client.ping()
        print("连接成功！会话已就绪。\n")

        # ==========================================
        # 场景 A: 使用工具 (Tools) - AI 的双手
        # ==========================================
        print("--- 测试工具调用 (Tools) ---")

        # A1. 列出可用工具
        tools = await client.list_tools()
        print(f"发现 {len(tools)} 个工具: {[t.name for t in tools]}")

        # A2. 调用 add 工具
        print("\n>> 调用 add(a=10, b=5.5)...")
        result_add = await client.call_tool("add", arguments={"a": 10, "b": 5})
        print(f"计算结果: {result_add.content[0].text}")

        # A3. 调用 calculate_bmi 工具
        print("\n>> 调用 calculate_bmi(weight=70, height=1.75)...")
        result_bmi = await client.call_tool("calculate_bmi", arguments={"weight_kg": 70, "height_m": 1.75})
        print(f"BMI 结果: {result_bmi.content[0].text}")

        # ==========================================
        # 场景 B: 读取资源 (Resources) - AI 的眼睛
        # ==========================================
        print("\n--- 测试资源读取 (Resources) ---")

        # B1. 列出可用资源
        resources = await client.list_resources()
        print(f"发现资源: {[r.uri for r in resources]}")

        # B2. 读取具体资源内容
        target_uri = "health://guidelines"
        print(f"\n>> 读取资源内容: {target_uri}")
        try:
            # read_resource 返回的是一个列表，因为一个 URI 可能包含多个数据块
            res_content = await client.read_resource(target_uri)
            text = res_content[0].text
            print(f"资源内容预览:\n{text.strip()}")
        except Exception as e:
            print(f"读取失败: {e}")

        # ==========================================
        # 场景 C: 获取提示词 (Prompts) - AI 的剧本
        # ==========================================
        print("\n--- 测试提示词模板 (Prompts) ---")

        # C1. 列出可用提示词
        prompts = await client.list_prompts()
        print(f"发现提示词: {[p.name for p in prompts]}")

        # C2. 获取填充后的提示词
        prompt_name = "analyze_my_health"
        print(f"\n>> 获取提示词: {prompt_name} (参数: Alice, 60kg, 1.65m)")

        prompt_result = await client.get_prompt(
            prompt_name,
            arguments={"name": "Alice", "weight": "60", "height": "1.65"}
        )

        # 打印生成的剧本内容
        message = prompt_result.messages[0]
        print(f"生成的 Prompt 角色: {message.role}")
        print(f"生成的 Prompt 内容:\n{message.content.text}")

        print("\n所有测试完成，断开连接。")


if __name__ == "__main__":
    asyncio.run(main())
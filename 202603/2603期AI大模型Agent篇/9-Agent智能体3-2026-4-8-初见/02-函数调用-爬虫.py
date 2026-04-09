from openai import OpenAI
import os
from dotenv import load_dotenv
from get_train_number_info import Crawl
from datetime import datetime
import json

load_dotenv()
model = "qwen3-max"  # 模型需要选择好一点的，不然会识别不到对应的任务
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")

client = OpenAI(api_key=api_key, base_url=api_base_url)


# 获取当前日期
def check_date():
    today = datetime.now().date()
    return today


# 定义函数库
# 函数库对象必须是一个字典，一个键值对代表一个函数，其中Key是代表函数名称的字符串，而value表示对应的函数。
function_repository = {
    "check_train_number_info": Crawl().main,
    "check_date": check_date
}


# 大模型执行
def get_llm_response(messages, model):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "check_train_number_info",
                    "description": "根据给定的日期查询对应的车票信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_": {
                                "type": "string",
                                "description": "日期",
                            },
                            "start": {
                                "type": "string",
                                "description": "出发站",
                            },
                            "end": {
                                "type": "string",
                                "description": "终点站",
                            }

                        },

                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_date",
                    "description": "返回当前的日期",
                    "parameters": {
                        "type": "object",
                        "properties": {
                        }
                    }
                }
            }
        ]
    )
    return response.choices[0].message


prompt = "查询后天长沙到上海的票"

messages = [
    {"role": "system", "content": "你是一个超级地图助手，你可以找到任何地址"},
    {"role": "user", "content": prompt}
]
response = get_llm_response(messages, model)

messages.append(response)  # 把大模型的回复加入到对话中
print("=====大模型回复1=====")
print(response)

# 如果返回的是函数调用结果，则打印出来
while response.tool_calls is not None:
    for tool_call in response.tool_calls:
        args = json.loads(tool_call.function.arguments)
        print("参数：", args)

        # 执行本地函数
        function_response = function_repository[tool_call.function.name](**args)

        print("=====函数返回=====")
        print(function_response)

        messages.append({
            "tool_call_id": tool_call.id,  # 用于标识函数调用的 ID
            "role": "tool",
            "name": tool_call.function.name,
            "content": str(function_response)  # 数值result 必须转成字符串
        })
    print("messages:", messages)
    response = get_llm_response(messages, model)
    print("=====大模型回复2=====")
    print(response)
    messages.append(response)  # 把大模型的回复加入到对话中

print("=====最终回复=====")
print(response.content)

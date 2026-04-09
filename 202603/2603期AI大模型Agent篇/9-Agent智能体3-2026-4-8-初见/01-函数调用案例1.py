import pandas as pd # pip install pandas
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import numpy as np

# 加载环境变量
load_dotenv()
# 初始化模型
model_name = "MiniMax-M2.1"
api_key = os.getenv("DASHSCOPE_API_KEY")
api_base_url = os.getenv("DASHSCOPE_BASE_URL")
client = OpenAI(api_key=api_key, base_url=api_base_url)

# 准备数据
df_employees = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Hank'],
    'Age': [25, 30, 35, 28, 32, 45, 29, 40],
    'Salary': [50000.0, 75000.5, 95000.75, 62000.0, 88000.25, 120000.0, 55000.0, 105000.0],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'Finance', 'HR', 'IT'],
    'IsMarried': [True, False, True, False, True, True, False, True],
    'YearsExperience': [3, 5, 8, 4, 7, 15, 4, 12]
})


# 获取数据的 Schema 信息，用于告诉 LLM 数据长什么样
def get_data_schema():
    return f"""
    数据集包含以下列：
    - Name (str): 员工姓名
    - Age (int): 年龄
    - Salary (float): 年薪
    - Department (str): 部门 (包含: {', '.join(df_employees['Department'].unique())})
    - IsMarried (bool): 婚姻状况
    - YearsExperience (int): 工作年限
    数据总行数: {len(df_employees)}
    """


# 1.准备函数（工具）
def calculate_salary_statistics():
    """计算薪资的统计信息"""
    try:
        # 直接使用全局 df，或者从数据库查询
        stats = {
            "average": round(df_employees['Salary'].mean(), 2),
            "median": round(df_employees['Salary'].median(), 2),
            "max": round(df_employees['Salary'].max(), 2),
            "min": round(df_employees['Salary'].min(), 2)
        }
        return json.dumps(stats)
    except Exception as e:
        return json.dumps({"error": str(e)})


def analyze_by_department():
    """按部门统计分析"""
    try:
        dept_stats = df_employees.groupby('Department').agg({
            'Name': 'count',
            'Salary': 'mean',
            'Age': 'mean'
        }).round(2)

        result = dept_stats.rename(columns={'Name': 'count', 'Salary': 'avg_salary', 'Age': 'avg_age'}).to_dict(
            orient='index')
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def find_employees_by_criteria(min_salary=None, max_age=None, department=None):
    """根据条件筛选员工"""
    try:
        df = df_employees.copy()
        if min_salary:
            df = df[df['Salary'] >= min_salary]
        if max_age:
            df = df[df['Age'] <= max_age]
        if department:
            df = df[df['Department'] == department]

        result = df[['Name', 'Department', 'Salary', 'Age']].to_dict(orient='records')
        return json.dumps({"count": len(result), "data": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def analyze_experience_salary_correlation():
    """分析经验与薪资相关性"""
    try:
        corr = df_employees['YearsExperience'].corr(df_employees['Salary'])
        return json.dumps({"correlation_coefficient": round(corr, 4)})
    except Exception as e:
        return json.dumps({"error": str(e)})


# 2.函数映射表
function_mapping = {
    'calculate_salary_statistics': calculate_salary_statistics,
    'analyze_by_department': analyze_by_department,
    'find_employees_by_criteria': find_employees_by_criteria,
    'analyze_experience_salary_correlation': analyze_experience_salary_correlation
}
# 3.定义tools(定义工具的描述信息，能让模型进行选择)
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_salary_statistics",
            "description": "计算全公司员工薪资的统计指标（平均值、中位数、最大最小）",
            "parameters": {"type": "object", "properties": {}, "required": []}  # 无参数
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_by_department",
            "description": "按部门进行分组统计（人数、平均薪资、平均年龄）",
            "parameters": {"type": "object", "properties": {}, "required": []}  # 无参数
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_employees_by_criteria",
            "description": "筛选员工。如果不指定条件，则不要传参。",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_salary": {"type": "number", "description": "最低薪资"},
                    "max_age": {"type": "integer", "description": "最大年龄"},
                    "department": {"type": "string", "description": "部门名称"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_experience_salary_correlation",
            "description": "计算工作年限与薪资的相关系数",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


# 4.执行函数调用的核心逻辑
def run(query):
    print("用户问题：", query)

    # 定义提示词
    messages = [{
        "role": "system",
        "content": f"你是一个数据分析专家，当前对应的表格数据如下 \n {get_data_schema} \n"
                   f"请根据用户的问题选择合适的工具"},
        {"role": "user", "content": query}
    ]

    # 第一次调用模型
    response = client.chat.completions.create(
        model=model_name,  # 模型名称
        messages=messages,  # 消息列表
        tools=tools,  # 工具列表
        tool_choice="auto"  # 工具选择模式
    )
    # 模型返回的就是是否要调用工具（做出决策）
    response_msg = response.choices[0].message
    # 将模型的输出添加到messages中
    messages.append(response_msg)

    print("第一次调用模型返回的结果：", response_msg)
    # 获取模型本次决策需要调用的模型
    tool_calls = response_msg.tool_calls

    # 判断模型是否要调用工具
    if tool_calls:
        print(f"模型本地决定调用{len(tool_calls)}个工具")

        # 因为模型会决策出使用多个工具去回答用户的问题，通过循环去执行多个工具
        for tool_call in tool_calls:
            # 获取需要调用工具的名称
            fn_name = tool_call.function.name
            # 获取需要调用工具的参数
            fn_args = json.loads(tool_call.function.arguments)

            print(f"执行的工具：{fn_name}，工具的参数：{fn_args}")

            # 判断当前模型返回的函数是否存在
            if fn_name in function_mapping:
                # 调用对应的函数，获取函数的结果
                fn_result = function_mapping[fn_name](**fn_args)
                print(f"{fn_name}函数的结果是：{fn_result}")
                # 需要往messages中添加tools调用的消息
                messages.append({
                    "role": "tool",  # 代表的是工具消息
                    "tool_call_id": tool_call.id,  # 必须要传递
                    "name": fn_name, "content": fn_result,
                })
            else:
                print(f"函数未定义:{fn_name}")
        # 将用户问题、函数调用的结果一起打包给模型进行最终的回复
        result = client.chat.completions.create(
            model=model_name,  # 模型名称
            messages=messages,  # 消息列表
            tools=tools,  # 工具列表
            tool_choice="auto"  # 工具选择模式
        )
        print(f"最终回复：{result}")
    else:
        print("模型决策之后决定不调用工具")

if __name__ == '__main__':
    run("帮我找一下工资大于8万的IT部门员工？在帮我算一下全公司的薪资相关性？")

    # 如果现在有1000工具，你该怎么去调用？
    #   1.RAG先去进行工具的过滤，在1000中找到最相似的5个工具，中间件，在生成工具的描述的时候，生成类似的问题
    # 用本地模型跑代码：8B的模型会把tools的能力蒸馏掉
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from 加载模型 import get_llm

# 加载模型
llm, embed_model = get_llm()


class EnterpriseKnowledgeBase:
    def __init__(self):
        self.router_engine = None
        self.setup_indexes()

    def setup_indexes(self):
        """设置各种索引"""
        # 1. 加载不同类型的文档
        # 技术文档
        tech_docs = SimpleDirectoryReader(input_files=["/home/tosinjia/LLM/files/java_basics_manual.md"]).load_data()
        # 产品文档
        product_docs = SimpleDirectoryReader(
            input_files=["/home/tosinjia/LLM/files/DeepSeek15天指导手册——从入门到精通.pdf"]).load_data()

        # 2. 创建向量索引
        tech_index = VectorStoreIndex.from_documents(tech_docs)
        product_index = VectorStoreIndex.from_documents(product_docs)

        # 3. 配置查询引擎
        tech_engine = tech_index.as_query_engine(  # 专门查询Java相关的内容
            similarity_top_k=3,
            response_mode="compact"
        )
        product_engine = product_index.as_query_engine(  # 专门查询deepseek知识的
            similarity_top_k=5,
            response_mode="tree_summarize"
        )

        # 4. 创建工具集
        query_tools = [
            QueryEngineTool(
                query_engine=tech_engine,
                metadata=ToolMetadata(
                    name="technical_docs",
                    description=(
                        "java基础技术文档库，包含基本语法、数据类型和变量、运算符、"
                        "控制结构、面向对象。适合回答编程、架构、异常处理等"
                    )
                )
            ),
            QueryEngineTool(
                query_engine=product_engine,
                metadata=ToolMetadata(
                    name="product_manual",
                    description=(
                        "包含对应deepseek的简单使用，基础对话篇"
                        "新⼿必学的10个魔法指令、效率⻜跃篇、场景实战篇等"
                    )
                )
            )
        ]

        # 5. 创建路由查询引擎
        self.router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=query_tools,
            verbose=True  # 开启详细日志
        )

    def query(self, question: str):
        """执行查询"""
        if not self.router_engine:
            raise ValueError("知识库未初始化")

        print(f"查询问题: {question}")
        print("-" * 50)

        response = self.router_engine.query(question)

        print(f"回答: {response}")
        print("=" * 50)

        return response


# 使用示例
def main():
    # 初始化知识库
    kb = EnterpriseKnowledgeBase()

    # 测试不同类型的查询
    test_queries = [
        "java的循环语句有哪些",
        "Java的特点？",
        "deepseek中的有效提问的五个⻩⾦法则？",
    ]

    for query in test_queries:
        kb.query(query)
        print()


if __name__ == "__main__":
    main()
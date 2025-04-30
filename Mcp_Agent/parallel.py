import asyncio

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# from mcp_agent.workflows.parallel.fan_in import FanIn
# from mcp_agent.workflows.parallel.fan_out import FanOut
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM
from rich import print
# To illustrate a parallel workflow, we will build a student assignment grader,``
# which will use a fan-out agent to grade the assignment in parallel using multiple agents,
# and a fan-in agent to aggregate the results and provide a final grade.


query = "双创基地有哪些公司在找后端工程师？"

app = MCPApp(name="mcp_parallel_workflow")


async def example_usage():
    async with app.run() as short_story_grader:
        logger = short_story_grader.logger

        file_searcher = Agent(
            name="file_searcher",
            instruction="""
            你是一个文件搜索专家，请根据用户的问题，使用filesystem工具在
            /Users/jing/Desktop/coding.../Agent/wechat-agent/Mcp_Agent/docs
            读取文件内容，并返回可以回答相关问题的文件内容。
            1.list全部文件
            2.根据文件名称判断该文件是否可以回答用户的问题。
            3.如果可以回答，则读取文件找到对应的内容，并返回。
            4.如果没有一个文件的文件名可以回答用户的问题，随机读取并总结一个文件的内容返回。
            不要输出markdown格式
            """,
            server_names=["filesystem"],
        )

        content_generator = Agent(
            name="content_generator",
            instruction="""根据用户的问题，直接给出对应回答。
            不要输出markdown格式""",
        )

        filter_agent = Agent(
            name="filter_agent",
            instruction="""
            任务：
            根据用户输入的query,评估content_generator和file_searcher的回答，
            如果file_searcher的回答更符合用户的问题，则返回file_searcher的回答，
            否则返回content_generator的回答。
            必须遵守的要求：
            1. 不要输出markdown格式
            2. 不要输出任何解释,直接给出回答。（我们不应该让用户看到任何的思考过程！！）
            """,
    
        )

        parallel = ParallelLLM(
            fan_in_agent=filter_agent,
            fan_out_agents=[file_searcher, content_generator],
            llm_factory=OpenAIAugmentedLLM,
        )

        result = await parallel.generate_str(
            message=f"query: {query}",
        )

        logger.info(f"{result}")


if __name__ == "__main__":
    import time

    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
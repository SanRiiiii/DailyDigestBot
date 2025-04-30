import asyncio
import os

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

app = MCPApp(name="news_agent")

async def example_usage():
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        # This agent can read the filesystem or fetch URLs
        news_agent = Agent(
            name="news_agent",
            instruction="""You can read local files or fetch URLs or search the web.
                Return the requested information when asked.""",
            server_names=["filesystem", "baidu-search"], # MCP servers this Agent can use
        )

        async with news_agent:
            # Automatically initializes the MCP servers and adds their tools for LLM use
            # tools = await finder_agent.list_tools()
            # logger.info(f"Tools available:", data=tools)

            # Attach an OpenAI LLM to the agent (defaults to GPT-4o)
            llm = await news_agent.attach_llm(OpenAIAugmentedLLM)

            # This will perform a file lookup and read using the filesystem server
            # result = await llm.generate_str(
            # message="查看data_1.csv和data_2.csv文件，找出寰宇智能的营业范围"
            # )
            # logger.info(f"寰宇智能的营业范围: {result}")

            # # Uses the fetch server to fetch the content from URL
            # result = await llm.generate_str(
            #     message="Print the first two paragraphs from https://36kr.com/p/3245190310444675"
            # )
            # logger.info(f"Blog intro: {result}")

            result = await llm.generate_str(
                message="Search the web for the latest (2025-04-10) AI news on the website called 36kr and fetch the content of the first article"
                
            )
            logger.info(f"36kr news: {result}")

            # Multi-turn interactions by default
            result = await llm.generate_str("Summarize that in a 128-char tweet")
            logger.info(f"Tweet: {result}")

if __name__ == "__main__":
    asyncio.run(example_usage())

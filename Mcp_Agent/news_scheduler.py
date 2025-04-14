import schedule
import time
import asyncio
from datetime import datetime
from Mcp_Agent.agent import Agent

async def send_news():
    """Send news at scheduled time"""
    print(f"Starting news delivery at {datetime.now()}")
    agent = Agent()
    await agent.run("请帮我生成今天的新闻摘要，包含国内和国际重要新闻，每个类别选择3-5条最重要的新闻，并给出简要分析。")
    print(f"News delivery completed at {datetime.now()}")

def run_scheduler():
    """Run the scheduler"""
    # Schedule the job to run every day at 9:00 AM
    schedule.every().day.at("09:00").do(lambda: asyncio.run(send_news()))
    
    print("News scheduler started. Will send news every day at 9:00 AM.")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    run_scheduler() 
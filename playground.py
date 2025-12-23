import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api=os.getenv("PHI_API_KEY")


web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the latest financial news and market sentiment.",
    model=Groq(id="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY")),
    tools=[DuckDuckGo()],
    instructions=["Provide concise summaries and always cite your sources."],
    show_tool_calls=True,  # Corrected parameter name
    markdown=True,
)

# 2. Financial Data Agent (for specific, structured data)
finance_agent = Agent(
    name="Finance AI Agent",
    role="Retrieve and display structured financial data for a given stock ticker.",
    model=Groq(id="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY")),
    tools=[
        # Use the robust custom tool for news
        # Use the other pre-built tools directly
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)
    ],
    instructions=["Use Markdown tables to display all financial data clearly."],
    show_tool_calls=True,  # Corrected parameter name
    markdown=True,
)

app=Playground(agents=[web_search_agent, finance_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)


import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables from a .env file
load_dotenv()

# --- Custom Robust News Tool ---
# This wrapper function fixes the 'num_stories' type error.
def get_robust_company_news(company_ticker: str, num_stories: any = 5):
    try:
        # Safely convert the input to an integer
        story_count = int(num_stories)
    except (ValueError, TypeError):
        # If conversion fails, default to a reasonable number and log a warning
        print(f"Warning: Invalid 'num_stories' format ('{num_stories}'). Defaulting to 5 stories.")
        story_count = 5
    
    # Instantiate the original tool and call its method with the clean integer
    yfinance_tool = YFinanceTools(company_news=True)
    return yfinance_tool.get_company_news(company_ticker=company_ticker, num_stories=story_count)


# --- Agent Definitions ---

# 1. Web Search Agent (for broad, up-to-date information)
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the latest financial news and market sentiment.",
    model=Groq(id="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY")),
    tools=[DuckDuckGo()],
    instructions=["Provide concise summaries and always cite your sources."],
    show_tool_calls=True,  
    markdown=True,
)

# 2. Financial Data Agent (for specific, structured data)
finance_agent = Agent(
    name="Finance AI Agent",
    role="Retrieve and display structured financial data for a given stock ticker.",
    model=Groq(id="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY")),
    tools=[
        # Use the robust custom tool for news
        get_robust_company_news,
        # Use the other pre-built tools directly
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)
    ],
    instructions=["Use Markdown tables to display all financial data clearly."],
    show_tool_calls=True, 
    markdown=True,
)

# 3. Multi-Agent Coordinator (to manage the team)
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=Groq(id="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY")),
    instructions=["Delegate tasks to the appropriate agent to gather and present the requested financial information.", "Synthesize the final report.", "Always include sources."],
    show_tool_calls=True, 
    markdown=True,
)

# --- Run the Agent ---
# The user prompt now specifies the number of news stories, which the robust tool can handle.
multi_ai_agent.print_response(
    "Summarize analyst recommendations and share the latest 3 news stories for NVDA.",
    stream=True
)


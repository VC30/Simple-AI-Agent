from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")


##web search agent
web_search_agent=Agent(
    name="Web Search Agent",
    role="search the web for the information",
    model=Groq(id="llama-3.2-1b-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

##Financial Agent
Finance_Agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.2-1b-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True)
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)


multi_ai_agent=Agent(
    model=Groq(id="llama-3.1-70b-versatile"),
    team=[web_search_agent,Finance_Agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)
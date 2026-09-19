from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

web_search = TavilySearch(
    api_key=tavily_api_key,
    max_results=5,
    topic="general",
    include_answer=True,
    include_raw_content=False,
)

print("Tavily search tool ready.")
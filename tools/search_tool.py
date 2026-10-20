
import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch


# Load environment variables from .env
load_dotenv()

# Initialize the Tavily web search tool
web_search = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
)


def search_web(query: str) -> str:
    """
    Search the web for information related to the given query.

    This tool allows the Researcher Agent to retrieve relevant
    web search results for further research and analysis.
    """
    results = web_search.invoke({"query": query})
    return str(results)

 
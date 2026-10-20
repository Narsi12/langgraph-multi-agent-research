
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

from state import ResearchState
from tools.search_tool import search_web

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


def researcher_agent(state: ResearchState) -> dict:
    """
    Research the given topic using web search and an LLM.

    The agent retrieves the topic from the shared state,
    searches the web, summarizes the results, and returns
    the research findings along with message history updates.
    """

    topic = state["topic"]

    # Step 1: Search the web
    search_results = search_web(
        f"Research {topic}. Find reliable sources, key facts, "
        "recent developments, and practical examples."
    )

    # Step 2: Summarize the search results using the LLM
    response = llm.invoke([
        HumanMessage(
            content=f"""
            You are an expert research assistant.

            Research topic:
            {topic}

            Web search results:
            {search_results}

            Instructions:
            - Summarize the important findings.
            - Include source URLs when available.
            - Identify facts, examples, and recent developments.
            - Do not invent facts or references.
            - Mention limitations in the available information.
            """
        )
    ])

    research_data = response.content

    # Step 3: Return updates to the shared state
    return {
    "research_data": research_data,
    "messages": [
        AIMessage(
            content=f"Research completed for topic: {topic}"
        )
    ],
}

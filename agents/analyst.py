
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

from state import ResearchState

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


def analyst_agent(state: ResearchState) -> dict:
    """
    Analyze research findings and identify insights,
    trends, benefits, risks, and limitations.
    """

    topic = state["topic"]
    research_data = state["research_data"]

    # Step 1: Send research findings to the LLM
    response = llm.invoke([
        HumanMessage(
            content=f"""
            You are an expert research analyst.

            Research topic:
            {topic}

            Research findings:
            {research_data}

            Analyze the findings and provide:

            1. Key insights
            2. Trends and patterns
            3. Benefits and limitations
            4. Risks and challenges
            5. Evidence gaps
            6. Practical recommendations

            Do not invent facts or statistics.
            Clearly identify unsupported claims.
            """
        )
    ])

    analysis = response.content

    # Step 2: Return updates to the shared state
    return {
        "analysis": analysis,
        "messages": [
            AIMessage(
                content=f"Analysis completed for topic: {topic}"
            )
        ],
    }

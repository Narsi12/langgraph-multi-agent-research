
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

from state import ResearchState

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


def writer_agent(state: ResearchState) -> dict:
    """
    Generate a structured research report using
    the research findings and analysis.
    """

    topic = state["topic"]
    research_data = state["research_data"]
    analysis = state["analysis"]
    feedback = state.get("revision_feedback", "")

    response = llm.invoke([
        HumanMessage(
            content=f"""
            You are a professional research report writer.

            Topic:
            {topic}

            Research findings:
            {research_data}

            Analysis:
            {analysis}

            Human review feedback:
            {feedback or "No previous feedback."}

            Write a complete research report with these sections:

            1. Title
            2. Executive Summary
            3. Introduction
            4. Research Findings
            5. Analysis and Insights
            6. Risks and Limitations
            7. Recommendations
            8. Conclusion
            9. Sources

            Requirements:
            - Use the provided research and analysis.
            - Preserve source URLs when available.
            - Do not invent facts, statistics, or citations.
            - Address all human review feedback.
            - Return the complete report.
            """
        )
    ])

    draft = response.content

    return {
        "draft": draft,
        "final_report": "",
        "revision_feedback": "",
        "messages": [
            AIMessage(
                content=f"Research report drafted for topic: {topic}"
            )
        ],
    }

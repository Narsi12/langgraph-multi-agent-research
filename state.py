
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """
    The add_messages reducer is used to maintain conversation history
    in the LangGraph state.

    It appends new messages to the existing message list instead of
    replacing the entire list. It also supports updating existing
    messages when their IDs match.

    This helps agents maintain a shared message history throughout
    the workflow, making it easier to track agent interactions,
    LLM responses, and other messages.
    """
    # Conversation and agent message history
    messages: Annotated[list[AnyMessage], add_messages]

    # User's research topic
    topic: str

    # Researcher's output
    research_data: str

    # Analyst's output
    analysis: str

    # Writer's output
    draft: str
    final_report: str

    # Supervisor routing
    next_agent: str

    # Human-in-the-loop
    review_decision: str
    revision_feedback: str

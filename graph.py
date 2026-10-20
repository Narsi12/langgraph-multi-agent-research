
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from state import ResearchState

from agents.supervisor import supervisor_agent
from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.writer import writer_agent


def human_review_agent(state: ResearchState) -> dict:
    """
    Pause the workflow and request human approval
    or revision feedback for the generated report.
    """

    review = interrupt({
        "message": "Review the generated report",
        "topic": state["topic"],
        "draft": state["draft"],
    })

    if isinstance(review, dict):
        decision = review.get("decision", "").strip().lower()
        feedback = review.get("feedback", "").strip()
    else:
        decision = str(review).strip().lower()
        feedback = ""

    if decision == "approve":
        return {
            "review_decision": "approve",
            "final_report": state["draft"],
            "revision_feedback": "",
        }

    return {
        "review_decision": "revise",
        "revision_feedback": (
            feedback or "Improve the report based on the review."
        ),
        "final_report": "",
    }


def route_supervisor(state: ResearchState) -> str:
    """Route execution to the agent selected by the supervisor."""

    return state["next_agent"]


def route_human_review(state: ResearchState) -> str:
    """Route according to the human review decision."""

    if state["review_decision"] == "approve":
        return "finalize"

    return "writer"


def finalize_report(state: ResearchState) -> dict:
    """Finalize the approved report."""

    return {
        "final_report": state["draft"]
    }


def build_graph():
    # Create the graph builder
    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("analyst", analyst_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("human_review", human_review_agent)
    builder.add_node("finalize", finalize_report)

    # Start with the supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor decides the next step
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "human_review": "human_review",
        },
    )

    # Return control to supervisor after research and analysis
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")

    # Writer sends the draft for human review
    builder.add_edge("writer", "human_review")

    # Human approves or requests revisions
    builder.add_conditional_edges(
        "human_review",
        route_human_review,
        {
            "finalize": "finalize",
            "writer": "writer",
        },
    )

    builder.add_edge("finalize", END)

    # Save graph checkpoints in application memory
    checkpointer = InMemorySaver()

    # Compile the graph with checkpointing enabled
    return builder.compile(checkpointer=checkpointer)


# Create the compiled graph
graph = build_graph()

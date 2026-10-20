
from state import ResearchState


def supervisor_agent(state: ResearchState) -> dict:
    """
    Decide which agent should execute next based on
    the current state of the research workflow.
    """

    # Step 1: Check whether research is completed
    if not state.get("research_data"):
        next_agent = "researcher"

    # Step 2: Check whether analysis is completed
    elif not state.get("analysis"):
        next_agent = "analyst"

    # Step 3: Check whether report draft is available
    elif not state.get("draft"):
        next_agent = "writer"

    # Step 4: All tasks are completed
    else:
        next_agent = "human_review"

    print(f"[Supervisor] Routing to: {next_agent}")

    return {
        "next_agent": next_agent
    }

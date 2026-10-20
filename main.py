
import uuid

from langgraph.types import Command

from graph import graph


def main():
    """
    Start the multi-agent research workflow
    and handle human review until approval.
    """

    # Step 1: Get research topic from the user
    topic = input(
        "Enter your research topic: "
    ).strip()

    if not topic:
        print("Research topic cannot be empty.")
        return

    # Step 2: Create a unique ID for this workflow
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Step 3: Initialize the shared state
    initial_state = {
        "messages": [],
        "topic": topic,
        "research_data": "",
        "analysis": "",
        "draft": "",
        "final_report": "",
        "next_agent": "",
        "review_decision": "",
        "revision_feedback": "",
    }

    print("\nStarting research workflow...")

    # Step 4: Run the graph
    result = graph.invoke(
        initial_state,
        config=config,
    )

    # Step 5: Handle Human-in-the-Loop interruptions
    while result.get("__interrupt__"):

        interrupt_info = result["__interrupt__"][0]
        review_data = interrupt_info.value

        print("\n" + "=" * 60)
        print("HUMAN REVIEW REQUIRED")
        print("=" * 60)

        print(f"\nTopic: {review_data['topic']}")
        print("\nGenerated report:\n")
        print(review_data["draft"])

        print("\nChoose an option:")
        print("1. Approve report")
        print("2. Request revisions")

        choice = input("\nEnter 1 or 2: ").strip()

        if choice == "1":
            resume_value = {
                "decision": "approve",
                "feedback": "",
            }

        elif choice == "2":
            feedback = input(
                "What changes do you want? "
            ).strip()

            resume_value = {
                "decision": "revise",
                "feedback": feedback,
            }

        else:
            print("Invalid choice. Please enter 1 or 2.")
            continue

        # Step 6: Resume the paused graph
        result = graph.invoke(
            Command(resume=resume_value),
            config=config,
        )

    # Step 7: Display the approved report
    print("\n" + "=" * 60)
    print("FINAL APPROVED REPORT")
    print("=" * 60)

    print(result["final_report"])


if __name__ == "__main__":
    main()

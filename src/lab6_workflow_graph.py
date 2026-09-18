"""
LAB 6 - A workflow agent built with LangGraph.

Labs 3-5 used create_agent, which builds ONE loop: think -> maybe call a tool ->
think again. That is the right shape when you do not know the steps in advance.

Often you DO know the steps. You want: classify the request, route it to the
right handler, review the draft, and send it back for revision if it is weak.
That is a WORKFLOW, and you draw it as a graph:

        START -> classify -> (route) -> academic_desk
                                     -> hostel_desk      -> review -> END
                                     -> technical_desk        |
                                              ^               |
                                              +--- revise ----+

Three ideas to take away:
  STATE  a dictionary that every node reads from and writes to.
  NODE   a plain Python function: state in, partial state out.
  EDGE   who runs next. A conditional edge lets the graph branch or loop.

Run:  python lab6_workflow_graph.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from config import banner, get_llm

llm = get_llm()
MAX_REVISIONS = 2


# ---------------------------------------------------------------------------
# 1. THE STATE
# ---------------------------------------------------------------------------
# A TypedDict is just a dictionary with declared key names. LangGraph merges
# whatever a node returns into this dictionary.
class TicketState(TypedDict):
    question: str       # what the student asked
    category: str       # academic | hostel | technical
    draft: str          # the answer being written
    verdict: str        # PASS | REVISE
    feedback: str       # why the reviewer wants a revision
    revisions: int      # how many times we have looped


# ---------------------------------------------------------------------------
# 2. THE NODES
# ---------------------------------------------------------------------------
def classify(state: TicketState) -> dict:
    """Decide which desk should handle this ticket."""
    reply = llm.invoke(
        "Classify this student query into exactly one word: "
        "academic, hostel, or technical.\n"
        "academic = marks, attendance, exams, subjects, projects\n"
        "hostel   = rooms, mess, timings, warden\n"
        "technical= wifi, laptop, software, portal login\n\n"
        f"Query: {state['question']}\n"
        "Answer with one word only."
    ).text.strip().lower()

    # Never trust free text. Normalise it against the values you allow.
    for known in ("academic", "hostel", "technical"):
        if known in reply:
            category = known
            break
    else:
        category = "academic"  # safe default

    print(f"  [classify] -> {category}")
    return {"category": category, "revisions": 0}


def _desk(state: TicketState, persona: str) -> dict:
    revision_note = ""
    if state.get("feedback"):
        revision_note = (
            f"\nYour previous draft was rejected. Reviewer feedback: {state['feedback']}\n"
            f"Previous draft: {state['draft']}\n"
            "Write an improved version."
        )
    draft = llm.invoke(
        f"{persona}\nAnswer the student in under 70 words. Be specific and practical."
        f"\n\nStudent query: {state['question']}{revision_note}"
    ).text.strip()
    return {"draft": draft}


def academic_desk(state: TicketState) -> dict:
    print("  [route] academic desk")
    return _desk(state, "You are the academic section clerk of an engineering college.")


def hostel_desk(state: TicketState) -> dict:
    print("  [route] hostel desk")
    return _desk(state, "You are the hostel warden's office assistant.")


def technical_desk(state: TicketState) -> dict:
    print("  [route] technical desk")
    return _desk(state, "You are the campus IT help desk engineer.")


def review(state: TicketState) -> dict:
    """A second model call that grades the draft. This is the 'reflection' step."""
    reply = llm.invoke(
        "You are a quality reviewer. Judge whether the reply below actually answers "
        "the student's question with a concrete next step.\n"
        "Reply in this exact format:\n"
        "PASS\n"
        "or\n"
        "REVISE: <one sentence saying what is missing>\n\n"
        f"Question: {state['question']}\n"
        f"Reply: {state['draft']}"
    ).text.strip()

    if reply.upper().startswith("PASS"):
        print("  [review] PASS")
        return {"verdict": "PASS", "feedback": ""}

    feedback = reply.split(":", 1)[-1].strip()
    print(f"  [review] REVISE - {feedback[:70]}")
    return {
        "verdict": "REVISE",
        "feedback": feedback,
        "revisions": state["revisions"] + 1,
    }


# ---------------------------------------------------------------------------
# 3. THE ROUTING FUNCTIONS (these decide edges, they do not change state)
# ---------------------------------------------------------------------------
def route_to_desk(state: TicketState) -> str:
    return state["category"]


def route_after_review(state: TicketState) -> str:
    if state["verdict"] == "PASS":
        return "done"
    if state["revisions"] >= MAX_REVISIONS:
        # ALWAYS give a loop an escape hatch, or it will run forever and burn
        # your whole API quota in one afternoon.
        print(f"  [review] revision limit ({MAX_REVISIONS}) reached - accepting draft")
        return "done"
    return state["category"]  # send it back to the same desk


# ---------------------------------------------------------------------------
# 4. WIRE THE GRAPH
# ---------------------------------------------------------------------------
builder = StateGraph(TicketState)

builder.add_node("classify", classify)
builder.add_node("academic", academic_desk)
builder.add_node("hostel", hostel_desk)
builder.add_node("technical", technical_desk)
builder.add_node("review", review)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    route_to_desk,
    {"academic": "academic", "hostel": "hostel", "technical": "technical"},
)

for desk in ("academic", "hostel", "technical"):
    builder.add_edge(desk, "review")

builder.add_conditional_edges(
    "review",
    route_after_review,
    {"academic": "academic", "hostel": "hostel", "technical": "technical", "done": END},
)

workflow = builder.compile()


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    banner("The graph you just built")
    # Paste this text into https://mermaid.live to see the picture.
    print(workflow.get_graph().draw_mermaid())

    for question in [
        "I was absent for two lab sessions because of fever. What do I do now?",
        "The mess food timing clashes with my evening class. Can I get a packed dinner?",
        "I cannot log in to the exam portal, it says invalid credentials.",
    ]:
        banner(f"TICKET: {question}")
        final = workflow.invoke({"question": question, "revisions": 0})
        print(f"\nDESK    : {final['category']}")
        print(f"VERDICT : {final['verdict']}  (after {final['revisions']} revision/s)")
        print(f"REPLY   : {final['draft']}")

    print(
        "\n" + "-" * 70 + "\n"
        "EXERCISE 1: add a fourth desk, 'fees', and route billing questions to it.\n"
        "EXERCISE 2: set MAX_REVISIONS = 0 and compare the answer quality.\n"
        "EXERCISE 3: print the state after every node using workflow.stream(...)\n"
        "            instead of workflow.invoke(...).\n"
        "EXERCISE 4 (stretch): make review() return a Pydantic model using\n"
        "            llm.with_structured_output() instead of parsing text."
    )

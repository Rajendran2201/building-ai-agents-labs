"""
LAB 7 - Multi-agent collaboration (the supervisor pattern).

Why more than one agent? Because a single agent with twelve tools and a
600-word system prompt starts choosing badly. Splitting the work gives each
agent a short prompt, few tools and one job - and each one gets better at it.

The supervisor pattern:

              +-------------------------------+
              |          SUPERVISOR           |  reads the conversation,
              |  "who should work next?"      |  names the next worker
              +---+-------------+---------+---+
                  |             |         |
             researcher      analyst    writer
                  |             |         |
                  +------> back to supervisor <------+
                                |
                             FINISH

Every worker reports back to the supervisor. The supervisor is the only node
that decides what happens next, so there is exactly one place to debug.

Run:  python lab7_multi_agent.py
"""

import ast
import operator
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import banner, get_llm

llm = get_llm()
WORKERS = ["researcher", "analyst", "writer"]
MAX_STEPS = 8  # hard stop, so a confused supervisor cannot loop forever


# ---------------------------------------------------------------------------
# TOOLS (the same safe calculator and lookups from Lab 4)
# ---------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _ev(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError("only arithmetic is allowed")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use for every calculation."""
    try:
        return str(_ev(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


@tool
def search_wikipedia(topic: str) -> str:
    """Look up a factual topic on Wikipedia and return a short summary."""
    import wikipedia

    try:
        return wikipedia.summary(topic, sentences=5, auto_suggest=True)
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia lookup failed: {exc}"


@tool
def search_web(query: str) -> str:
    """Search the live web for recent information and return results with URLs."""
    from ddgs import DDGS

    try:
        with DDGS() as engine:
            results = list(engine.text(query, max_results=4))
    except Exception as exc:  # noqa: BLE001
        return f"Web search failed: {exc}"
    return "\n\n".join(
        f"{r.get('title','')}\n{r.get('body','')}\n{r.get('href','')}" for r in results
    ) or "No results."


# ---------------------------------------------------------------------------
# THE THREE WORKERS - each is a complete agent from Lab 4, just smaller
# ---------------------------------------------------------------------------
researcher = create_agent(
    model=llm,
    tools=[search_wikipedia, search_web],
    system_prompt=(
        "You are a researcher. Gather facts using your tools and report them as "
        "short bullet points. Include a number or a date wherever one exists. "
        "Never write prose and never draw conclusions - that is someone else's job."
    ),
)

analyst = create_agent(
    model=llm,
    tools=[calculate],
    system_prompt=(
        "You are a quantitative analyst. Take the facts already gathered in the "
        "conversation and compute what matters: totals, differences, percentages, "
        "growth rates. Use the calculate tool for every number. Show each figure "
        "with the expression that produced it. Do not invent data that is not "
        "already in the conversation."
    ),
)

writer = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "You are a technical writer. Turn the research and analysis already in "
        "the conversation into the final deliverable the user asked for. Use only "
        "facts present in the conversation. Structure it with short headings. "
        "Finish with a one-line 'Bottom line:'."
    ),
)


# ---------------------------------------------------------------------------
# THE SHARED STATE
# ---------------------------------------------------------------------------
class TeamState(TypedDict):
    # add_messages is a REDUCER. Annotated[list, add_messages] tells LangGraph
    # "when a node returns messages, APPEND them" instead of the default
    # behaviour, which is to overwrite. Without it, each worker would wipe out
    # what the previous worker said.
    messages: Annotated[list, add_messages]
    next: str
    steps: int


# ---------------------------------------------------------------------------
# THE SUPERVISOR NODE
# ---------------------------------------------------------------------------
SUPERVISOR_PROMPT = """You are the supervisor of a three-person team.

researcher - finds facts using Wikipedia and web search. Use FIRST for any
             question that needs external information.
analyst    - does arithmetic on facts that are already in the conversation.
             Only useful once the researcher has reported numbers.
writer     - produces the final document. Use LAST, exactly once.

Read the conversation so far and reply with EXACTLY ONE word: researcher,
analyst, writer, or FINISH.

Rules:
- Do not send work to the writer until the facts needed are in the conversation.
- Reply FINISH only after the writer has produced the deliverable.
- Never repeat a worker that has just reported unless something is clearly missing."""


def supervisor(state: TeamState) -> dict:
    if state["steps"] >= MAX_STEPS:
        print("  [supervisor] step limit reached -> FINISH")
        return {"next": "FINISH", "steps": state["steps"] + 1}

    transcript = "\n\n".join(
        f"{getattr(m, 'name', None) or m.__class__.__name__}: {m.text[:600]}"
        for m in state["messages"]
    )
    reply = llm.invoke(
        f"{SUPERVISOR_PROMPT}\n\nCONVERSATION SO FAR:\n{transcript}\n\nNext worker:"
    ).text.strip().lower()

    choice = next((w for w in WORKERS if w in reply), "FINISH")
    print(f"  [supervisor] -> {choice}")
    return {"next": choice, "steps": state["steps"] + 1}


# ---------------------------------------------------------------------------
# WORKER NODES - run the sub-agent, then hand its report back to the team
# ---------------------------------------------------------------------------
def _run_worker(state: TeamState, agent, name: str) -> dict:
    print(f"  [{name}] working...")
    result = agent.invoke({"messages": state["messages"]})
    report = result["messages"][-1].text
    # name= is what lets the supervisor tell the workers apart in the transcript.
    return {"messages": [AIMessage(content=report, name=name)]}


def researcher_node(state: TeamState) -> dict:
    return _run_worker(state, researcher, "researcher")


def analyst_node(state: TeamState) -> dict:
    return _run_worker(state, analyst, "analyst")


def writer_node(state: TeamState) -> dict:
    return _run_worker(state, writer, "writer")


# ---------------------------------------------------------------------------
# WIRE THE TEAM
# ---------------------------------------------------------------------------
def route(state: TeamState) -> Literal["researcher", "analyst", "writer", "__end__"]:
    return END if state["next"] == "FINISH" else state["next"]


team = StateGraph(TeamState)
team.add_node("supervisor", supervisor)
team.add_node("researcher", researcher_node)
team.add_node("analyst", analyst_node)
team.add_node("writer", writer_node)

team.add_edge(START, "supervisor")
team.add_conditional_edges(
    "supervisor",
    route,
    {"researcher": "researcher", "analyst": "analyst", "writer": "writer", END: END},
)
# every worker reports back to the supervisor
for worker in WORKERS:
    team.add_edge(worker, "supervisor")

crew = team.compile()


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    banner("Team structure")
    print(crew.get_graph().draw_mermaid())

    task = (
        "Prepare a one-page technical brief for second-year students on India's "
        "installed solar power capacity. Include the latest capacity figure, how "
        "much it grew over the last five years in percentage terms, and what that "
        "means for engineering job prospects."
    )

    banner("TASK")
    print(task)
    print("\nExecution trace:")

    final = crew.invoke(
        {"messages": [HumanMessage(content=task)], "next": "", "steps": 0}
    )

    banner("FINAL DELIVERABLE")
    print(final["messages"][-1].text)

    banner("WHO SAID WHAT")
    for message in final["messages"]:
        who = getattr(message, "name", None) or message.__class__.__name__
        print(f"\n### {who}\n{message.text[:400]}")

    print(
        "\n" + "-" * 70 + "\n"
        "EXERCISE 1: remove the analyst from WORKERS and from the graph. Does the\n"
        "            supervisor cope? What happens to the percentage figure?\n"
        "EXERCISE 2: set MAX_STEPS = 2 and explain the output in your record.\n"
        "EXERCISE 3: add a 'reviewer' worker that must approve the writer's draft\n"
        "            before the supervisor is allowed to FINISH.\n"
        "EXERCISE 4: replace the supervisor's free-text reply with\n"
        "            llm.with_structured_output() and compare reliability over\n"
        "            ten runs."
    )

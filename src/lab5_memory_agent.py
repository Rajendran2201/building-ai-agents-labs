"""
LAB 5 - Memory.

In Lab 1 Part C you proved that a model forgets everything between calls.
Here you fix that, and you learn that "memory" is really two different things:

    SHORT-TERM  the current conversation.  Handled by a CHECKPOINTER + thread_id.
                Lives as long as the program runs. Automatic once configured.

    LONG-TERM   facts that should survive after the program exits.
                Handled by TOOLS that read and write storage. Never automatic -
                the agent must decide something is worth remembering.

Run:  python lab5_memory_agent.py
"""

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from config import banner, get_llm

MEMORY_FILE = Path(__file__).parent / "long_term_memory.json"


# ===========================================================================
# PART A - SHORT-TERM MEMORY (checkpointer + thread_id)
# ===========================================================================
# A checkpointer saves the state of the conversation after every step, filed
# under a thread_id that you choose. InMemorySaver keeps it in RAM.
chat_agent = create_agent(
    model=get_llm(),
    tools=[],  # no tools yet - we are only testing memory
    system_prompt="You are a friendly academic advisor. Keep replies under 40 words.",
    checkpointer=InMemorySaver(),
)


def chat(text: str, thread_id: str) -> str:
    """Every call with the same thread_id continues the same conversation."""
    result = chat_agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].text.strip()


def part_a() -> None:
    banner("PART A - Short-term memory")

    print("--- thread 'student-A' ---")
    intro = "My name is Divya and I am in the 5th semester, ECE branch."
    print(f"You : {intro}")
    print("Bot :", chat(intro, "student-A"))
    print("\nYou : Which semester am I in?")
    print("Bot :", chat("Which semester am I in?", "student-A"))

    print("\n--- thread 'student-B' (a different student, same agent object) ---")
    print("You : Which semester am I in?")
    print("Bot :", chat("Which semester am I in?", "student-B"))

    print(
        "\nOBSERVE: thread 'student-A' remembers. Thread 'student-B' does not,\n"
        "because it is a separate conversation. In a real application the\n"
        "thread_id would be the student's roll number or a chat session id.\n"
        "Getting this wrong is how chatbots leak one user's data to another."
    )

    # You can read the stored state back yourself - useful when debugging.
    state = chat_agent.get_state({"configurable": {"thread_id": "student-A"}})
    print(f"\nMessages stored under 'student-A': {len(state.values['messages'])}")


# ===========================================================================
# PART B - LONG-TERM MEMORY (tools that write to a file)
# ===========================================================================
def _load() -> dict:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {}


def _save(data: dict) -> None:
    MEMORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


@tool
def remember_fact(key: str, value: str) -> str:
    """Save a durable fact about the student so it is still known in future
    sessions. Use a short lowercase key such as "branch", "semester" or
    "project_topic". Only save things the student states about themselves."""
    data = _load()
    data[key] = value
    _save(data)
    return f"Saved {key} = {value}"


@tool
def recall_facts() -> str:
    """Return everything currently known about the student from previous
    sessions. Call this at the start of a conversation."""
    data = _load()
    if not data:
        return "Nothing stored yet."
    return "\n".join(f"{k}: {v}" for k, v in data.items())


memory_agent = create_agent(
    model=get_llm(),
    tools=[remember_fact, recall_facts],
    system_prompt=(
        "You are an academic advisor with a long-term memory.\n"
        "- Call recall_facts at the start of a conversation.\n"
        "- When the student states a durable fact about themselves (branch, "
        "semester, project topic, career goal), call remember_fact.\n"
        "- Do not save passing remarks or anything about other people.\n"
        "Keep replies under 40 words."
    ),
    checkpointer=InMemorySaver(),
)


def advise(text: str, thread_id: str = "session-1") -> None:
    print(f"\nYou : {text}")
    result = memory_agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    for message in result["messages"]:
        if message.__class__.__name__ == "AIMessage" and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"      [tool] {call['name']}({call['args']})")
    print("Bot :", result["messages"][-1].text.strip())


def part_b() -> None:
    banner("PART B - Long-term memory")

    advise("Hi, I am doing my final year project on solar panel fault detection.")
    advise("What did I say my project was about?")

    print(f"\nContents of {MEMORY_FILE.name} right now:")
    print(json.dumps(_load(), indent=2))
    print(
        "\nRun this script again. Part B will recall the project topic on the\n"
        "FIRST turn, because it came from the file, not from the conversation."
    )


# ===========================================================================
# PART C - WHAT HAPPENS WHEN A CONVERSATION GETS TOO LONG
# ===========================================================================
# Every turn is re-sent to the model, so a long conversation eventually exceeds
# the context window and costs more every time. SummarizationMiddleware watches
# the conversation and, once it passes a threshold, replaces the older messages
# with a summary of them.
summarising_agent = create_agent(
    model=get_llm(),
    tools=[],
    system_prompt="You are a helpful tutor. Keep replies under 30 words.",
    middleware=[
        SummarizationMiddleware(
            model=get_llm(),
            trigger=("messages", 6),   # summarise once there are more than 6 messages
            keep=("messages", 4),      # keep the last 4 messages word for word
        )
    ],
    checkpointer=InMemorySaver(),
)


def part_c() -> None:
    banner("PART C - Summarising old turns")

    topics = [
        "Explain what a vector is, in one line.",
        "Now explain a matrix, in one line.",
        "Now explain a tensor, in one line.",
        "Now explain an eigenvalue, in one line.",
        "List, in order, every topic I have asked about so far.",
    ]
    for turn in topics:
        print(f"\nYou : {turn}")
        out = summarising_agent.invoke(
            {"messages": [{"role": "user", "content": turn}]},
            config={"configurable": {"thread_id": "long-chat"}},
        )
        print("Bot :", out["messages"][-1].text.strip())

    state = summarising_agent.get_state({"configurable": {"thread_id": "long-chat"}})
    print(
        f"\nMessages retained after 5 turns: {len(state.values['messages'])}"
        "\n(Without the middleware this list would keep growing forever.)"
    )


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()

    print(
        "\n" + "-" * 70 + "\n"
        "EXERCISE 1: give Part A the same thread_id for both students and re-run.\n"
        "            Explain, in the record notebook, what goes wrong and why.\n"
        "EXERCISE 2: delete long_term_memory.json and run Part B again.\n"
        "EXERCISE 3: in Part C, raise trigger to ('messages', 50). Does the agent\n"
        "            answer the last question better or worse? Why?\n"
        "EXERCISE 4: make remember_fact refuse to overwrite an existing key\n"
        "            without the student confirming."
    )

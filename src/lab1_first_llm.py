"""
LAB 1 - Talking to a model, and seeing why a model alone is NOT an agent.

Learning goals
  1. Send a prompt and read a response.
  2. See the difference between a system message and a human message.
  3. Watch the model fail at a task it physically cannot do, and understand that
     this gap is the reason agents exist.

Run:  python lab1_first_llm.py
"""

from langchain_core.messages import HumanMessage, SystemMessage

from config import banner, get_llm

llm = get_llm()


# ---------------------------------------------------------------------------
# PART A - the simplest possible call
# ---------------------------------------------------------------------------
banner("PART A - One prompt, one answer")

response = llm.invoke("Explain what an AI agent is in exactly two sentences.")
print(response.text)


# ---------------------------------------------------------------------------
# PART B - a system message changes the model's behaviour, not its knowledge
# ---------------------------------------------------------------------------
banner("PART B - Adding a system message")

messages = [
    SystemMessage(
        "You are a strict examiner for a B.E. Computer Science course. "
        "Answer in bullet points. Never use more than 40 words."
    ),
    HumanMessage("What is the difference between a chatbot and an agent?"),
]
print(llm.invoke(messages).text)


# ---------------------------------------------------------------------------
# PART C - conversation is NOT automatic
# ---------------------------------------------------------------------------
banner("PART C - The model has no memory of its own")

print("Turn 1 ->", llm.invoke("My roll number is 21CS045.").text)
print("\nTurn 2 ->", llm.invoke("What is my roll number?").text)
print(
    "\nOBSERVE: turn 2 has no idea. Each .invoke() is a brand new request.\n"
    "Memory is something YOU add (Lab 5), not something the model has."
)


# ---------------------------------------------------------------------------
# PART D - the capability gap that motivates tools
# ---------------------------------------------------------------------------
banner("PART D - Three things a plain model cannot do reliably")

for question in [
    "What is 84937 multiplied by 2946?",          # arithmetic: often wrong
    "What is today's date?",                       # live data: cannot know
    "What is our college's exam timetable?",       # private data: never saw it
]:
    print(f"\nQ: {question}")
    print(f"A: {llm.invoke(question).text.strip()}")

print(
    "\n" + "-" * 70 + "\n"
    "CONCLUSION\n"
    "A language model predicts text. It cannot calculate, cannot look things up,\n"
    "and has never seen your private documents.\n"
    "An AGENT = a model + tools it may call + a loop that decides when to call\n"
    "them. The rest of this module builds exactly that, one piece at a time."
)

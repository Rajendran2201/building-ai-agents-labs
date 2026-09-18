"""
LAB 3 - Your first real agent: knowledge retrieval as a TOOL.

The difference from Lab 2 in one line:
    Lab 2 - you searched, then asked the model to summarise.
    Lab 3 - the model decides whether to search, what to search for, and when
            it has searched enough.

Run Lab 2 first (it creates handbook_index.json).

Run:  python lab3_retrieval_agent.py
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore

from config import banner, get_embeddings, get_llm

INDEX_FILE = Path(__file__).parent / "handbook_index.json"

if not INDEX_FILE.exists():
    raise SystemExit("handbook_index.json not found. Run: python lab2_rag_pipeline.py")

vector_store = InMemoryVectorStore.load(str(INDEX_FILE), get_embeddings())


# ---------------------------------------------------------------------------
# STEP 1 - Wrap the retriever in a tool
# ---------------------------------------------------------------------------
# The @tool decorator turns an ordinary Python function into something the model
# can call. THREE things become part of the model's prompt automatically:
#   1. the function name          -> search_handbook
#   2. the docstring              -> tells the model WHEN to use it
#   3. the type hints             -> tells the model what arguments to send
# Write the docstring for the model, not for your classmate. It is the only
# instruction the model gets about this tool.

@tool
def search_handbook(query: str) -> str:
    """Search the college student handbook for rules on attendance, internal
    assessment, laboratory courses, the library, the hostel, project work,
    examination malpractice and grievances. Use this for ANY question about
    college rules. Pass a short natural-language query."""
    hits = vector_store.similarity_search(query, k=3)
    if not hits:
        return "No matching section found in the handbook."
    return "\n\n---\n\n".join(
        f"[section from {doc.metadata.get('source', 'handbook')}]\n{doc.page_content}"
        for doc in hits
    )


# ---------------------------------------------------------------------------
# STEP 2 - Build the agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the student help desk for Sri Venkatesa Institute of Technology.

Rules you must follow:
- Always call search_handbook before answering a question about college rules.
- Answer using only what the tool returns. Do not use general knowledge.
- If the handbook does not cover it, say so plainly and suggest the student ask
  the class advisor.
- Quote the specific figure (marks, days, rupees, percentage) whenever there is one.
- Keep answers under 80 words."""

agent = create_agent(
    model=get_llm(),
    tools=[search_handbook],
    system_prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# STEP 3 - Run it, and watch every step
# ---------------------------------------------------------------------------
def ask(question: str) -> None:
    banner(f"Q: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    # result["messages"] is the full transcript: what you asked, what the model
    # decided, what the tool returned, and the final answer. Read it. This is
    # how you debug an agent.
    for message in result["messages"]:
        kind = message.__class__.__name__

        if kind == "AIMessage" and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"  [model decided] call {call['name']}({call['args']})")
        elif kind == "ToolMessage":
            preview = message.content.replace("\n", " ")[:120]
            print(f"  [tool returned] {preview}...")
        elif kind == "AIMessage" and message.text.strip():
            print(f"\nANSWER: {message.text.strip()}")


if __name__ == "__main__":
    # 1. needs the tool
    ask("I have 70 percent attendance. Can I write the exam?")

    # 2. needs the tool, and the wording does not match the handbook wording
    ask("How late can I keep a library book before I start paying?")

    # 3. the handbook says nothing about this - watch the agent admit it
    ask("What is the Wi-Fi password in the hostel?")

    # 4. no tool needed at all - watch the agent skip the search
    ask("Hello, who are you?")

    print(
        "\n" + "-" * 70 + "\n"
        "EXERCISE: comment out the line 'Always call search_handbook before...'\n"
        "in SYSTEM_PROMPT and re-run. Note which questions the agent now answers\n"
        "from memory instead of from the handbook. That is prompt engineering\n"
        "doing real work."
    )

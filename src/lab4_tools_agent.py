"""
LAB 4 - Giving an agent several tools, and letting it choose.

You will build four tools of three different kinds:
    calculate        - pure Python, deterministic  (fixes the arithmetic gap)
    current_datetime - reads the machine clock     (fixes the "today" gap)
    search_wikipedia - calls an external library   (fixes the knowledge gap)
    search_web       - calls a live search engine  (fixes the freshness gap)

Then you will watch the model pick the right one, and sometimes chain two.

Run:  python lab4_tools_agent.py
"""

import ast
import datetime as dt
import operator

from langchain.agents import create_agent
from langchain_core.tools import tool

from config import banner, get_llm


# ---------------------------------------------------------------------------
# TOOL 1 - a calculator that is actually safe
# ---------------------------------------------------------------------------
# Do NOT use eval(). eval() will happily run os.system("rm -rf /") if the model
# is tricked into passing it. We parse the expression and allow only arithmetic.

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / ** % are allowed")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact result. Use this for
    EVERY calculation, however easy it looks. Accepts + - * / ** % and brackets,
    for example "84937 * 2946" or "(1200 - 450) / 3"."""
    try:
        return str(_eval_node(ast.parse(expression, mode="eval").body))
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# TOOL 2 - the machine clock
# ---------------------------------------------------------------------------
@tool
def current_datetime() -> str:
    """Return today's date and the current time. Use this whenever the question
    involves today, now, this week, or how many days until something."""
    return dt.datetime.now().strftime("%A, %d %B %Y, %H:%M")


# ---------------------------------------------------------------------------
# TOOL 3 - Wikipedia
# ---------------------------------------------------------------------------
@tool
def search_wikipedia(topic: str) -> str:
    """Look up a factual topic on Wikipedia and return a short summary. Good for
    people, places, organisations, scientific terms and historical events. Pass a
    single topic name, not a full question."""
    import wikipedia

    try:
        return wikipedia.summary(topic, sentences=4, auto_suggest=True)
    except wikipedia.DisambiguationError as exc:
        return f"'{topic}' is ambiguous. Did you mean: {', '.join(exc.options[:5])}?"
    except wikipedia.PageError:
        return f"No Wikipedia page found for '{topic}'."
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia lookup failed: {exc}"


# ---------------------------------------------------------------------------
# TOOL 4 - live web search
# ---------------------------------------------------------------------------
@tool
def search_web(query: str) -> str:
    """Search the live web and return the top results with their URLs. Use this
    for recent news, prices, and anything that changes from day to day."""
    from ddgs import DDGS

    try:
        with DDGS() as engine:
            results = list(engine.text(query, max_results=4))
    except Exception as exc:  # noqa: BLE001
        return f"Web search failed: {exc}"

    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
        for r in results
    )


# ---------------------------------------------------------------------------
# Build the agent
# ---------------------------------------------------------------------------
TOOLS = [calculate, current_datetime, search_wikipedia, search_web]

agent = create_agent(
    model=get_llm(),
    tools=TOOLS,
    system_prompt=(
        "You are a careful research assistant for engineering students.\n"
        "- Never do arithmetic in your head. Always use the calculate tool.\n"
        "- Never guess today's date. Always use current_datetime.\n"
        "- Prefer search_wikipedia for settled facts and search_web for recent news.\n"
        "- You may call several tools before answering.\n"
        "- State which tool gave you each fact."
    ),
)


def ask(question: str) -> None:
    banner(f"Q: {question}")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        if message.__class__.__name__ == "AIMessage" and getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"  -> calling {call['name']}({call['args']})")
        elif message.__class__.__name__ == "ToolMessage":
            print(f"  <- {message.content.replace(chr(10), ' ')[:110]}")
    print(f"\nANSWER: {result['messages'][-1].text.strip()}")


if __name__ == "__main__":
    print("Tools registered:")
    for t in TOOLS:
        print(f"  {t.name:<18} {t.description.splitlines()[0][:70]}")

    ask("What is 84937 multiplied by 2946?")
    ask("Who invented the transistor, and in what year?")
    ask("How many days are there between today and 1 January 2027?")
    ask("What is the latest version of Python, and what is its version number times 100?")

    print(
        "\n" + "-" * 70 + "\n"
        "OBSERVE the last question. The agent had to search the web, read the\n"
        "answer, and then feed that answer into a second tool. Nobody wrote that\n"
        "sequence. The loop discovered it. That is what makes it an agent.\n\n"
        "EXERCISE: delete the calculate tool from TOOLS and re-run question 1.\n"
        "Compare the answer with 84937 * 2946 = 250,224,402."
    )

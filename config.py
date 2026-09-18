"""
config.py  --  shared setup for every lab in Module V.

Every lab file starts with `from config import get_llm, get_embeddings`.
Keeping this in one place means that if a model name changes, you edit ONE file
instead of eight.
"""

import os
import sys

from dotenv import load_dotenv

# Reads the .env file sitting next to this script and puts the values into
# os.environ. This is how your API key reaches the library without ever being
# typed into your source code.
load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")


def _require_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "paste_your_key_here":
        sys.exit(
            "\nGOOGLE_API_KEY is not set.\n"
            "Fix: copy .env.example to .env and paste your key from\n"
            "     https://aistudio.google.com/apikey\n"
        )
    return key


def get_llm(temperature: float = 0.0):
    """Return the chat model used by all labs.

    temperature=0.0 means 'be as predictable as possible'. Use it while you are
    learning, so that re-running a lab gives you nearly the same answer twice.
    """
    _require_key()
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
        max_retries=2,  # ride out the occasional free-tier hiccup
    )


def get_embeddings():
    """Return the embedding model used by the retrieval labs (Lab 2 and Lab 3)."""
    _require_key()
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def banner(title: str) -> None:
    """Small helper so lab output is readable in a terminal."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

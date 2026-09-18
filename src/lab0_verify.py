"""
LAB 0 - Environment check.

Run this BEFORE every other lab. It answers one question: "is my machine ready?"

    python lab0_verify.py
    python lab0_verify.py --list-models    # show models your key can use
"""

import sys

from config import CHAT_MODEL, EMBED_MODEL, banner


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    print(f"[{'PASS' if ok else 'FAIL'}] Python {major}.{minor} (need 3.10 or newer)")
    return ok


def check_imports() -> bool:
    required = [
        ("langchain", "langchain"),
        ("langchain_core", "langchain-core"),
        ("langgraph", "langgraph"),
        ("langchain_google_genai", "langchain-google-genai"),
        ("dotenv", "python-dotenv"),
    ]
    all_ok = True
    for module_name, pip_name in required:
        try:
            __import__(module_name)
            print(f"[PASS] import {module_name}")
        except ImportError:
            print(f"[FAIL] import {module_name}  ->  pip install {pip_name}")
            all_ok = False
    return all_ok


def check_key() -> bool:
    import os

    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key or key == "paste_your_key_here":
        print("[FAIL] GOOGLE_API_KEY not found. Create a .env file (see .env.example).")
        return False
    # Never print a key. Show only enough to confirm the right one was loaded.
    print(f"[PASS] GOOGLE_API_KEY loaded (ends with ...{key[-4:]})")
    return True


def check_llm_call() -> bool:
    from config import get_llm

    try:
        reply = get_llm().invoke("Reply with exactly one word: ready")
        print(f"[PASS] {CHAT_MODEL} answered: {reply.text.strip()!r}")
        return True
    except Exception as exc:  # noqa: BLE001 - we want the raw message for students
        print(f"[FAIL] chat model call failed: {type(exc).__name__}: {exc}")
        return False


def check_embedding_call() -> bool:
    from config import get_embeddings

    try:
        vector = get_embeddings().embed_query("hello")
        print(f"[PASS] {EMBED_MODEL} returned a vector of {len(vector)} numbers")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] embedding call failed: {type(exc).__name__}: {exc}")
        return False


def list_models() -> None:
    """Print every model your API key can reach, so the lab survives renames."""
    import os

    from google import genai

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    banner("Models available to your key")
    for model in client.models.list():
        actions = ", ".join(model.supported_actions or [])
        print(f"{model.name:<55} {actions}")


def main() -> None:
    if "--list-models" in sys.argv:
        list_models()
        return

    banner("LAB 0 - Environment check")
    results = [
        check_python(),
        check_imports(),
        check_key(),
    ]
    if all(results):
        results.append(check_llm_call())
        results.append(check_embedding_call())

    banner("RESULT: " + ("ALL CHECKS PASSED - you may start Lab 1"
                         if all(results) else
                         "SOME CHECKS FAILED - fix the FAIL lines above first"))


if __name__ == "__main__":
    main()

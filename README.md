# Building AI Agents — Hands-on Labs (Module V)

A complete, step-by-step laboratory module for building production-style AI agents with **LangChain 1.x**, **LangGraph**, and the free **Google Gemini** API.

Designed for B.E. / B.Tech engineering students.  
**Prerequisite:** basic Python only. No prior AI or machine learning required.

---

## What you will build

| Lab | Title | Core concept |
|-----|-------|--------------|
| 0 | Environment setup & verification | Clean install, API key, model check |
| 1 | The model alone is not an agent | Memory, tools & private data gaps |
| 2 | Knowledge retrieval pipeline (RAG) | Load → Split → Embed → Store → Retrieve |
| 3 | Retrieval as a tool | First real agent that decides when to search |
| 4 | Multiple tools | Calculator, clock, Wikipedia, web search + chaining |
| 5 | Memory | Short-term (checkpointer) + long-term (tools) + summarisation |
| 6 | Workflow agent (LangGraph) | State graph, conditional routing, bounded revision loop |
| 7 | Multi-agent collaboration | Supervisor pattern (researcher → analyst → writer) |

**Duration:** ~9 hours (3 sessions of 3 hours)  
**Cost:** Zero — free Gemini API key, no credit card  
**Hardware:** Any laptop with 4 GB RAM and internet (no GPU)

---

## Learning outcomes

After completing this module you will be able to:

- Explain the difference between a language model, a retrieval pipeline and an agent
- Build a RAG pipeline over private documents and measure retrieval quality
- Write custom tools, expose them to a model, and diagnose tool selection
- Add short-term and long-term memory with proper thread isolation
- Design multi-step workflows as state graphs with conditional branching
- Decompose tasks across cooperating agents under a supervisor

---

## Project structure

```
agentic-labs/
├── .env                      # Your API key (never commit this)
├── .env.example
├── requirements.txt          # Pinned, verified package versions
├── config.py                 # Shared LLM + embeddings setup
├── lab0_verify.py
├── lab1_first_llm.py
├── lab2_rag_pipeline.py
├── lab3_retrieval_agent.py
├── lab4_tools_agent.py
├── lab5_memory_agent.py
├── lab6_workflow_graph.py
├── lab7_multi_agent.py
├── data/
│   └── college_handbook.md   # Fictional handbook for RAG labs
└── README.md
```

---

## Quick start

### 1. Clone & enter the project

```bash
git clone https://github.com/Rajendran2201/building-ai-agents-labs.git
cd agentic-labs
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Gemini API key

1. Open [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with any Google account
3. Click **Create API key**
4. Copy the key

### 5. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` and paste your key:

```
GOOGLE_API_KEY=AIza...your_real_key...
CHAT_MODEL=gemini-2.0-flash          # or whatever free model is available
EMBED_MODEL=gemini-embedding-001
```

> **Note:** Model names change. If you get a 404, run  
> `python lab0_verify.py --list-models` and update `CHAT_MODEL`.

### 6. Verify the setup

```bash
python lab0_verify.py
```

You must see:

```
RESULT: ALL CHECKS PASSED - you may start Lab 1
```

---

## Running the labs

Run them **in order**. Later labs depend on earlier ones.

```bash
python lab0_verify.py
python lab1_first_llm.py
python lab2_rag_pipeline.py      # creates handbook_index.json
python lab3_retrieval_agent.py
python lab4_tools_agent.py
python lab5_memory_agent.py
python lab6_workflow_graph.py
python lab7_multi_agent.py
```

### Free-tier tips

- Gemini free tier has tight rate limits (requests per minute **and** per day).
- If you see `429 RESOURCE_EXHAUSTED`, wait 40–60 seconds or switch to another free model.
- Labs 6 and 7 make many sequential calls — run them when you have remaining daily quota.

---

## Important design rules taught in this module

- **Never put API keys in source code** — always use `.env`
- **Never use `eval()` on model-supplied input** — parse safely
- **Always give loops an escape hatch** (`MAX_REVISIONS`, `MAX_STEPS`)
- **Thread isolation is a security control** — different users must have different `thread_id`s
- **Write tool docstrings for the model**, not for humans
- **Normalise free-text model output** before using it for control flow

---

## Tech stack

| Component | Version / details |
|-----------|-------------------|
| Language | Python 3.10+ |
| Framework | LangChain 1.4.0, LangGraph 1.2.11 |
| Model | Google Gemini (free tier) |
| Embeddings | `gemini-embedding-001` |
| Vector store | `InMemoryVectorStore` (langchain-core) |
| Tools | Wikipedia, DuckDuckGo Search, safe calculator |

---

## License

Educational use. The sample handbook is fictional and written specifically for these labs.

---

## Acknowledgements

Verified against LangChain 1.4.0 · LangGraph 1.2.11 · Python 3.11 — September 2026  
Laboratory Manual — Module V: Hands on Labs: Building AI Agents

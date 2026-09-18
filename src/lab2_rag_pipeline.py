"""
LAB 2 - Build a knowledge retrieval pipeline (RAG) from scratch.

This lab has NO agent in it. You are building the machinery that the agent in
Lab 3 will use as a tool. Do not skip it: almost every "my agent gives wrong
answers" bug is really a retrieval bug.

The five steps, in order:
    LOAD    read the raw document
    SPLIT   cut it into chunks small enough to embed
    EMBED   turn each chunk into a list of numbers (a vector)
    STORE   put the vectors in a searchable store
    RETRIEVE find the chunks closest to a question

Run:  python lab2_rag_pipeline.py
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import banner, get_embeddings, get_llm

DATA_FILE = Path(__file__).parent / "data" / "college_handbook.md"
INDEX_FILE = Path(__file__).parent / "handbook_index.json"


# ---------------------------------------------------------------------------
# STEP 1 - LOAD
# ---------------------------------------------------------------------------
banner("STEP 1 - Load the document")

raw_text = DATA_FILE.read_text(encoding="utf-8")
document = Document(page_content=raw_text, metadata={"source": DATA_FILE.name})

print(f"Loaded {DATA_FILE.name}: {len(raw_text)} characters")
print(f"A Document has two parts: .page_content and .metadata")
print(f"metadata = {document.metadata}")


# ---------------------------------------------------------------------------
# STEP 2 - SPLIT
# ---------------------------------------------------------------------------
banner("STEP 2 - Split into chunks")

# chunk_size    : roughly how many characters per chunk
# chunk_overlap : characters repeated from the previous chunk, so that a sentence
#                 cut in half still appears whole in one of the two chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n## ", "\n\n", "\n", ". ", " "],  # try to break at headings first
)
chunks = splitter.split_documents([document])

print(f"Produced {len(chunks)} chunks")
print("\n--- chunk 0 ---")
print(chunks[0].page_content[:300])
print("\n--- chunk 1 ---")
print(chunks[1].page_content[:300])


# ---------------------------------------------------------------------------
# STEP 3 - EMBED
# ---------------------------------------------------------------------------
banner("STEP 3 - Turn text into vectors")

embeddings = get_embeddings()

sample_vector = embeddings.embed_query("How many books can I borrow?")
print(f"A question becomes a vector of {len(sample_vector)} numbers.")
print(f"First five numbers: {[round(x, 4) for x in sample_vector[:5]]}")
print(
    "\nWhy this matters: two texts with similar MEANING produce vectors that\n"
    "point in a similar direction, even when they share no common words."
)


# ---------------------------------------------------------------------------
# STEP 4 - STORE
# ---------------------------------------------------------------------------
banner("STEP 4 - Store the vectors")

# InMemoryVectorStore ships inside langchain-core, so there is nothing extra to
# install. It embeds every chunk and keeps the vectors in RAM.
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)
print(f"Indexed {len(chunks)} chunks.")

# Save to disk so Lab 3 does not have to pay for embeddings all over again.
vector_store.dump(str(INDEX_FILE))
print(f"Saved the index to {INDEX_FILE.name}")


# ---------------------------------------------------------------------------
# STEP 5 - RETRIEVE
# ---------------------------------------------------------------------------
banner("STEP 5 - Search")

question = "What happens if my attendance is 70 percent?"
hits = vector_store.similarity_search_with_score(question, k=3)

print(f"Question: {question}")
print("score = cosine similarity, so HIGHER is more relevant (max 1.0)\n")
for rank, (doc, score) in enumerate(hits, start=1):
    preview = doc.page_content.replace("\n", " ")[:160]
    print(f"[{rank}] score={score:.4f}  {preview}...")

print(
    "\nNotice: the question never uses the word 'condonation', but the chunk that\n"
    "explains condonation comes back first. That is semantic search working."
)


# ---------------------------------------------------------------------------
# STEP 6 - RETRIEVE, THEN ANSWER (this is what "RAG" means)
# ---------------------------------------------------------------------------
banner("STEP 6 - Feed the retrieved chunks to the model")

context = "\n\n---\n\n".join(doc.page_content for doc, _ in hits)

prompt = f"""Answer the student's question using ONLY the handbook extract below.
If the extract does not contain the answer, reply exactly: "Not covered in the handbook."

HANDBOOK EXTRACT:
{context}

QUESTION: {question}

ANSWER:"""

print(get_llm().invoke(prompt).text)

print(
    "\n" + "-" * 70 + "\n"
    "That is Retrieval Augmented Generation: retrieve first, then generate.\n"
    "It is still NOT an agent, because YOU decided to search. The model was\n"
    "never given a choice. In Lab 3 we hand that decision to the model."
)

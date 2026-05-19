# Task 4 — Retrieval-Based AI Assistant (RAG): v2 Implementation

This folder houses the **v2 RAG (Retrieval-Augmented Generation)** assistant, adding overlap chunking, a persistent SQLite-backed ChromaDB vector database, strict grounded completions via the Groq API (using the Llama3-8B model), rich auditing, and a multi-query terminal-based CLI.

---

## 1. Directory Structure

```text
task4/
└── v2/
    ├── docs/                                  # Copied from v1
    │   ├── maintenance_guide.txt
    │   ├── equipment_manual.txt
    │   ├── troubleshooting_faq.txt
    │   ├── safety_procedures.txt
    │   └── preventive_maintenance.txt
    ├── src/
    │   ├── chunker.py                         # Overlap chunking logic (300/50 split)
    │   ├── embedder.py                        # local sentence-transformers loader
    │   ├── vector_store.py                    # ChromaDB Persistent Client integration
    │   ├── retriever.py                       # query embeds + ChromaDB retriever
    │   ├── generator.py                       # Grounding prompt + Groq completions calls
    │   └── logger.py                          # Terminal printing & audit trail logging
    ├── outputs/
    │   ├── chroma_db/                         # Persistent SQLite vector index files
    │   └── retrieved_chunks.log               # Append-only search logs
    ├── main.py                                # Modular master orchestrator & interactive CLI
    ├── requirements.txt                       # Standalone requirements
    └── README.md                              # This document
```

---

## 2. Technical Additions and Architectural Advantages

### Step 1 — Overlap Chunking (`src/chunker.py`)
* **Methodology:** Rather than splitting files into rigid disjointed segments, v2 slides a window of `chunk_size = 300` words across the text using an `overlap = 50` words. Each chunk tracks rich coordinates (`start_word`, `end_word`, `word_count`).
* **Why it is Required:** Rigid partitions slice sentences in half, causing semantic disconnection. By overlapping chunks, sentences lying on the boundary are fully preserved in both adjacent chunks, guaranteeing context integrity.
* **Example:**
  > "To initiate LOTO, open the main cabinet and [boundary] turn off the 480V circuit breaker."
  Without overlap, Chunk 1 contains only the cabinet step, while Chunk 2 contains the breaker step. An engineer searching for "how to initiate LOTO" would fail to retrieve the full procedural context. Overlap duplicates the boundary zone, keeping the instruction whole in at least one chunk.

### Step 2 — ChromaDB Persistent Vector Store (`src/vector_store.py`)
* **Methodology:** We replaced the in-memory FAISS library with a serverless, persistent SQLite-backed ChromaDB store (`chromadb.PersistentClient`) pointed at `outputs/chroma_db/`. On the first run, text chunks are embedded and added to the `maintenance_kb` collection. On subsequent runs, it checks for existing data, skips the compute-heavy re-embedding phase entirely, and reads the indexed database directly from disk.
* **Why ChromaDB is Superior to FAISS:**
  1. **Persistence & Lifecycle:** FAISS is a low-level mathematical similarity index that stores vectors in memory. Developing a persistence layer requires manually writing index serialization and maintaining secondary lookups. ChromaDB automatically handles both vector indexes and rich metadata mapping in a unified SQLite database.
  2. **Metadata Filtering:** ChromaDB natively supports document filtering during the vector query stage (e.g. `where={"doc_name": "safety_procedures.txt"}`), while FAISS requires complex post-retrieval scripting.

### Step 3 — Grounded Generation via Groq API (`src/generator.py`)
* **Methodology:** Generates answers via Groq's low-latency LPU completions engine using `llama3-8b-8192`.
* **Zero Hallucination Guardrails:** The model is constrained via a strict grounding prompt with `temperature = 0.0` ensuring deterministic responses. If the retrieved context chunks cannot answer the query, it returns exactly: *"I don't have enough information in my knowledge base to answer this question."*
* **Why Groq:** Groq is powered by custom LPU hardware, generating answers at 500+ tokens per second. It offers a free-tier API key and supports Llama-3-8B which is outstanding at factual Q&A.

### Step 4 — Chunk Source Logging (`src/logger.py`)
Appends rich metadata parameters (query, exact timestamps, retrieved document titles, similarity scores, word indexes, previews, and final answers) to `outputs/retrieved_chunks.log` and outputs them to the terminal.
* **Why it matters in Industrial Systems:** Industrial operations carry extreme safety risk. If a technician executes a false instruction, it can lead to physical injury or capital destruction. Source logging provides an unalterable audit trail proving which corporate manuals and passages were consulted to generate an answer.

### Step 5 — Multi-Query CLI Loop (`main.py`)
Orchestrates an interactive, user-friendly terminal session where engineers can iteratively type questions. On startup, the script automatically tests four baseline validation scenarios:
1. *"What are the warning signs of bearing failure?"* (Should match FAQ)
2. *"What PPE is required during equipment maintenance?"* (Should match Safety)
3. *"How do I perform lockout tagout procedure?"* (Should match Safety)
4. *"What is the meaning of life?"* (Should trigger the strict grounding fallback)

---

## 3. Running and Verifying the RAG Assistant

To run the pipeline, activate your virtual environment, export your Groq API key, and run `main.py`:

```bash
# 1. Navigate to task4/v2/
cd task4/v2

# 2. Activate virtual environment
source ../../venv/bin/activate

# 3. Export your Groq API Key
export GROQ_API_KEY="your-groq-api-key"

# 4. Run the pipeline
python main.py
```

### Automatic Verification Points:
* **Fresh Build:** On the first execution, watch the logs print: `[ChromaDB] Collection 'maintenance_kb' not found. Creating a new one...` followed by chunk database insertions.
* **Disk Reloading:** Run `python main.py` a second time, and verify that the logs instantly print: `[ChromaDB] Collection 'maintenance_kb' successfully loaded from disk.` and skip re-embedding.
* **Grounding Fallback:** Observe that query 4 ("What is the meaning of life?") successfully triggers the grounding fallback.

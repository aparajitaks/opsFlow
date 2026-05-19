# Task 4 — Retrieval-Based AI Assistant (RAG): v1 Implementation

This folder contains the **v1 RAG (Retrieval-Augmented Generation)** assistant designed to answer industrial mechanical and electrical maintenance queries using a locally indexed technical knowledge base and the Anthropic Claude LLM.

---

## 1. Directory Structure

```text
task4/
└── v1/
    ├── docs/                                  # Knowledge Base Documents
    │   ├── maintenance_guide.txt              # General schedules & lubrication
    │   ├── equipment_manual.txt               # Motor specifications & limits
    │   ├── troubleshooting_faq.txt            # Failure symptoms & error codes
    │   ├── safety_procedures.txt              # LOTO & PPE protocols
    │   └── preventive_maintenance.txt         # Sensors & predictive analytics
    ├── outputs/
    │   └── retrieved_chunks.log               # Append-only search logs
    ├── rag_pipeline.py                        # Single-file RAG pipeline orchestrator
    ├── requirements.txt                       # Task specific dependencies
    └── README.md                              # This documentation guide
```

---

## 2. RAG Pipeline Core Workflow

The system implements Retrieval-Augmented Generation in a purely custom workflow without high-level wrappers like LangChain or LlamaIndex:

1. **Knowledge Base Ingestion:** Loads five plain-text industrial manuals (containing realistic motor limits, LOTO sequences, sensor velocity thresholds, and schedules).
2. **Fixed-Size Chunking (Step 2):** Groups document text into chunks of exactly 300 words with no overlap. Metadata (`doc_name`, `chunk_index`, `word_count`) is attached to each chunk.
3. **Local Embedding Generation (Step 3):** Computes high-density 384-dimensional vectors for every chunk using the local `sentence-transformers/all-MiniLM-L6-v2` transformer.
4. **FAISS Vector Storage (Step 4):** Standardizes the embedding vectors (L2 Normalization) and stores them in a `faiss.IndexFlatIP` index to resolve query similarities via cosine similarity.
5. **Dense Contextual Retrieval (Step 5):** Embeds user questions, retrieves the top $k=3$ most similar knowledge chunks, and passes them along with the query to the Anthropic Claude (`claude-3-5-sonnet-20241022` or latest `claude-sonnet-4-20250514`) via the official `anthropic` Python SDK.
6. **Detailed Audit Logging (Step 6):** Appends query text, timestamps, matching chunks, and similarity scores into `outputs/retrieved_chunks.log` after every execution.

---

## 3. Running the RAG Assistant Locally

Before executing the assistant, ensure your virtual environment is active and that your Anthropic API Key is set in your environment:

```bash
# 1. Navigate to the task4 v1 folder
cd task4/v1

# 2. Activate the virtual environment
source ../../venv/bin/activate

# 3. Export your Anthropic API Key
export ANTHROPIC_API_KEY="your-api-key-here"

# 4. Run the RAG pipeline end-to-end
python rag_pipeline.py
```

Three default query scenarios will execute to demonstrate full retrieval capabilities and log outputs directly to the terminal and `outputs/retrieved_chunks.log`.

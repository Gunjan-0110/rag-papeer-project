# 📄 Papeer — Research Paper Assistant

<p align="center">
  <b>A conversational AI research assistant for reading, exploring, and verifying research papers.</b>
</p>

<p align="center">
  Built with RAG, LangGraph, Gemini, local embeddings, ChromaDB, Tavily, and ArXiv.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangGraph-Workflow-1C3C3C" alt="LangGraph">
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Gemini-LLM-4285F4?logo=google" alt="Gemini">
</p>

---

## 🧠 What is Papeer?

**Papeer** is a research-paper assistant designed for students, researchers, and developers who work with technical papers and documents.

Instead of treating every question as a simple LLM prompt, Papeer uses a **LangGraph-based workflow with intelligent routing** to determine whether a query should use document retrieval, external search, claim verification, or direct generation.

Depending on the query, Papeer can:

- 📚 Search uploaded research papers
- 🌐 Search the web for external information
- 🔬 Search ArXiv for research evidence
- ✅ Verify research claims
- 💬 Answer general questions directly
- 🗂️ Maintain independent research sessions
- ⚡ Stream responses through the UI

The project focuses on reproducing the **research-assistant experience and functionality** of a modern RAG application while replacing expensive infrastructure with **free or locally running alternatives wherever practical**.

---

# ✨ Features

| Feature | Description |
|---|---|
| 📄 **Paper Q&A** | Ask questions about uploaded research papers and documents |
| 🔎 **Semantic Retrieval** | Retrieve relevant document chunks using local embeddings |
| 🌐 **Web Search** | Search the web when external or recent information is required |
| ✅ **Claim Verification** | Verify claims using web and ArXiv evidence |
| 🧠 **Intelligent Routing** | LangGraph routes queries to the appropriate workflow |
| 💬 **Multi-Session Chat** | Maintain separate research conversations |
| 🗂️ **Session Isolation** | Each session maintains its own document retrieval space |
| 📚 **Multiple Sources** | PDF, TXT, Markdown, Web URLs, and ArXiv |
| ⚡ **Streaming Responses** | Responses are progressively displayed in the UI |
| 📝 **Auto Session Naming** | Sessions receive automatically generated titles |
| 🔍 **Graph State Inspector** | Inspect the state flowing through the LangGraph workflow |
| 🔒 **`/btw` Side Channel** | Ask temporary questions without adding them to the main chat |
| 📊 **RAG Evaluation** | Evaluate generated responses using DeepEval |

---

# 🏗️ Architecture

```text
                         ┌─────────────────┐
                         │    User Query   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ LangGraph Router│
                         └────────┬────────┘
                                  │
          ┌───────────────┬───────┼────────┬──────────────┐
          │               │       │        │              │
          ▼               ▼       ▼        ▼              ▼
    Direct Answer     Retrieval   Web    Claim Verify    /btw
                          │        │         │
                          ▼        ▼         ▼
                       ChromaDB  Tavily   Tavily
                          │                  +
                          │                ArXiv
                          │                  │
                          └────────┬─────────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │  Gemini LLM  │
                            └──────┬───────┘
                                   │
                                   ▼
                            ┌──────────────┐
                            │ Streamlit UI │
                            └──────────────┘
```

---

# 🔄 How Papeer Works

1. **User asks a question** — For example: *What is the main contribution of this paper?*

2. **LangGraph routes the query** — The query is routed to document retrieval, direct generation, or claim verification.

3. **Relevant sources are used** — Papeer can use uploaded documents, ChromaDB, web search, ArXiv, or the LLM directly.

4. **Gemini generates the response** — Retrieved context and relevant information are used to generate the final answer.

---

# 📚 Supported Sources

Papeer supports multiple research sources:

- **Local:** PDF, TXT, Markdown
- **Web:** Web page URLs
- **ArXiv:** Paper IDs and research queries

---

# 🔎 Retrieval Pipeline

For document-based questions, Papeer follows a standard RAG pipeline:

```text
Document
    ↓
Load
    ↓
Split into Chunks
    ↓
Generate Embeddings
    ↓
Store in ChromaDB
    ↓
User Query
    ↓
Similarity Search
    ↓
Relevant Chunks
    ↓
Gemini
    ↓
Answer
```

### Current chunking configuration

```text
Chunk Size    : 1000 characters
Chunk Overlap : 200 characters
Retrieval     : Top 4 chunks
```

The overlap helps preserve context when information spans across chunk boundaries.

---

# 🤗 Local Embeddings

Papeer uses `sentence-transformers/all-MiniLM-L6-v2` through HuggingFace embeddings.

Embeddings are generated locally and stored in ChromaDB, avoiding the need for a separate paid embedding API.
---

# 🗄️ Session-Isolated Vector Store

Each research session gets its own ChromaDB storage.

Conceptually:

```text
chroma_db/
│
├── session_1/
├── session_2/
└── session_3/
```

This prevents documents belonging to one research session from unintentionally becoming retrieval context for another.

For example:

```text
Session A
├── Transformer Paper
└── BERT Paper

Session B
├── CNN Paper
└── Vision Transformer Paper
```

A query in Session A searches the documents associated with Session A.

---

# 💬 Multi-Session Research

Papeer supports independent research sessions, each maintaining its own conversation context, loaded documents, and vector-store space.

This prevents research contexts from different topics from being mixed.

---

# 🌐 Web Search

When uploaded papers are not sufficient, Papeer can use **Tavily** to retrieve external information.

Example:

```text
What are the latest developments in this research area?
```

---

# ✅ Claim Verification

Papeer also supports research-oriented claim verification.

Example:

```text
Verify this claim against recent research:
"Transformers always outperform RNNs."
```

The verification workflow uses:

```text
                Claim
                  │
          ┌───────┴────────┐
          ▼                ▼
     Tavily Search     ArXiv Search
          │                │
          └───────┬────────┘
                  ▼
               Evidence
                  │
                  ▼
                Gemini
                  │
                  ▼
          Verification Result
```

This allows claims to be checked against both **web information** and **research literature**.

---

# 🔒 `/btw` — Side Channel

Papeer includes a lightweight side-channel command for temporary questions.

Example:

```text
/btw What is the capital of France?
```

These questions can be answered without becoming part of the main research conversation.

This keeps the primary research history focused on the actual paper or topic being studied.

---

# ⚡ Streaming Responses

Papeer uses LangGraph's streaming execution together with Streamlit's streaming interface.

Instead of waiting for the entire workflow to finish before displaying the response, the application progressively renders the generated output.

```text
User Query
    ↓
LangGraph
    ↓
Routing / Retrieval / Search
    ↓
Generation
    ↓
Streaming Output
    ↓
Streamlit
```

---


---

# 📝 Automatic Session Naming

When a new research session receives its first question, Papeer automatically generates a concise session title, making multiple research sessions easier to identify.

---

# 🔍 LangGraph State Inspector

Papeer exposes a **Graph State Inspector** for understanding what happens inside the RAG workflow.

This is particularly useful for development and debugging.

The inspector can expose information such as:

```text
Query
Session ID
Route
Documents
Generation
```

This makes the application more transparent than a black-box chatbot.

---

# 🧪 Evaluation

Papeer includes a DeepEval-based evaluation workflow.

The current implementation evaluates three important RAG properties:

| Metric | Purpose |
|---|---|
| **Faithfulness** | Checks whether the generated answer is supported by the retrieved context |
| **Answer Relevancy** | Measures whether the response addresses the question |
| **Contextual Relevancy** | Measures whether the retrieved context is relevant to the question |

### Evaluation threshold

```text
0.7
```

The evaluation pipeline provides a foundation for measuring RAG quality rather than relying entirely on manual inspection.

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python |
| **Frontend** | Streamlit |
| **LLM** | Google Gemini |
| **Workflow** | LangGraph |
| **RAG Framework** | LangChain |
| **Embeddings** | HuggingFace / Sentence Transformers |
| **Embedding Model** | `all-MiniLM-L6-v2` |
| **Vector Database** | ChromaDB |
| **Web Search** | Tavily |
| **Research Search** | ArXiv |
| **Evaluation** | DeepEval |
| **Environment** | python-dotenv |

---

# 💰 Free / Local-First Approach

Papeer is designed as a **free/local-first implementation**.

- **Local:** Sentence Transformers, ChromaDB, document processing
- **API Services:** Gemini and Tavily

This approach minimizes dependence on paid infrastructure while keeping the application practical to run.

---

# 📁 Project Structure

```text
rag-papeer-project/
│
├── app.py                  # Streamlit application
├── evaluate.py             # DeepEval evaluation
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── README.md
│
├── backend/
│   ├── models.py           # LLM and graph state definitions
│   ├── paper_loader.py     # Document loading and chunking
│   ├── vector_store.py     # Embeddings + ChromaDB
│   ├── rag_graph.py        # LangGraph RAG workflow
│   └── btw_handler.py      # Side-channel handling
│
└── documents/              # Document storage
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Gunjan-0110/rag-papeer-project.git
cd rag-papeer-project
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# ▶️ Run Papeer

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

---

# 🚀 Quick Start

1. **Create a session** — Start a new research conversation.
2. **Load research material** — Upload a PDF, TXT, Markdown file, Web URL, or ArXiv paper.
3. **Ask questions** — Query your research material using natural language.
4. **Search externally** — Use web search when additional information is needed.
5. **Verify claims** — Use the claim-verification workflow for research evidence.
   
---

# 🧭 Current Implementation vs Future Improvements

The current project focuses on reproducing the **core research-assistant workflow** first.

### ✅ Currently Implemented

- Paper/document Q&A
- Semantic retrieval
- Local embeddings
- ChromaDB
- LangGraph routing
- Direct answers
- Web search
- Claim verification
- ArXiv search
- Multiple document sources
- Multi-session research
- Session-specific vector storage
- Automatic session naming
- Streaming responses
- `/btw` side channel
- Graph State Inspector
- DeepEval evaluation

### 🔮 Future Improvements

- Query rewriting
- Retrieval relevancy grading
- Retrieval retry loops
- Re-ranking
- Hybrid retrieval
- Better source/citation attribution
- Paper comparison
- Literature-review generation
- Related-paper discovery
- Persistent database-backed conversation history
- More extensive automated evaluation

Keeping these in the roadmap is intentional: they are **future extensions**, not claims about the current implementation.

---

<p align="center">
  <b>📄 Papeer — Read. Retrieve. Verify. Research.</b>
</p>
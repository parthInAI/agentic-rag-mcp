<div align="center">

# 🧠 Agentic RAG — MCP-Powered System

**A fully local, agentic Retrieval-Augmented Generation pipeline**  
Built with FastAPI · ChromaDB · Ollama · Streamlit · Sentence Transformers

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-MCP%20Server-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge&logo=llama&logoColor=white)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-F5A623?style=for-the-badge&logo=huggingface&logoColor=white)](https://sbert.net)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

This project implements a fully **local** agentic RAG (Retrieval-Augmented Generation) system using the **Model Context Protocol (MCP)** pattern to modularly connect an LLM to external tools — a vector database, a web search fallback, and persistent conversation memory.

The agent:
- **Retrieves** relevant context from your documents via semantic search (ChromaDB)
- **Falls back** to DuckDuckGo web search when local docs are insufficient
- **Remembers** conversation history across sessions via a JSON memory store
- **Generates** grounded answers using a local LLM served by Ollama

Everything runs **100% locally** — no OpenAI, no cloud APIs, no data leaving your machine.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Streamlit Frontend  (app.py)                 │
│       Upload docs · Chat · View memory · Reset store      │
└───────────────────────┬──────────────────────────────────┘
                        │ HTTP REST
┌───────────────────────▼──────────────────────────────────┐
│           FastAPI MCP Server  (main.py)                   │
│   /ask · /ingest/file · /ingest/text · /search · /memory  │
└──────┬────────────────┬───────────────────┬──────────────┘
       │                │                   │
┌──────▼──────┐  ┌──────▼──────┐   ┌───────▼──────┐
│  rag_agent  │  │  ChromaDB   │   │ Agent Memory  │
│ (orchestrat)│  │ (vector DB) │   │ (JSON store)  │
└──────┬──────┘  └─────────────┘   └──────────────┘
       │
┌──────▼──────────────────┐   ┌──────────────────────┐
│   Ollama  (local LLM)   │   │  DuckDuckGo Fallback  │
│  mistral · llama3 · …   │   │   web_search_tool.py  │
└─────────────────────────┘   └──────────────────────┘
```

---

## 🗂️ Project Structure

```
agentic-rag-mcp/
├── main.py                  # FastAPI MCP server (REST API)
├── app.py                   # Streamlit frontend UI
├── cli.py                   # CLI interface (no server needed)
├── rag_agent.py             # Core agent: tool routing, prompt, LLM call
├── ingest_docs.py           # Bulk document ingestion CLI
├── mcp_config.yaml          # Central configuration file
├── requirements.txt
├── tools/
│   ├── chromadb_tool.py     # ChromaDB vector search tool
│   └── web_search_tool.py   # DuckDuckGo web search fallback
├── memory/
│   └── agent_memory.py      # JSON-backed persistent conversation memory
├── vector_store/            # ChromaDB persisted data (auto-created)
└── data/
    └── sample_docs/         # Drop your PDFs / TXTs here
```

---

## 🛠️ Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **LLM** | [Ollama](https://ollama.com) | Run `mistral`, `llama3`, `phi3` locally |
| **MCP Server** | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn | REST API exposing agent tools |
| **Vector Store** | [ChromaDB](https://trychroma.com) | Persistent local vector database |
| **Embeddings** | [Sentence Transformers](https://sbert.net) (`all-MiniLM-L6-v2`) | Document & query embeddings |
| **RAG Pipeline** | Custom orchestration + [LangChain](https://langchain.com) | Retrieval, prompt building, tool routing |
| **Document Parsing** | [PyPDF](https://pypdf.readthedocs.io) | PDF text extraction & chunking |
| **Frontend** | [Streamlit](https://streamlit.io) | Chat UI with file upload & memory viewer |
| **Web Search** | DuckDuckGo Instant Answer API | No-key fallback search tool |
| **Memory** | JSON file store | Persistent multi-conversation history |
| **Runtime** | Python 3.10+ | Core language |

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/parthInAI/agentic-rag-mcp.git
cd agentic-rag-mcp

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2. Start Ollama & pull a model

```bash
ollama serve
ollama pull mistral      # or: ollama pull llama3 / phi3
```

### 3. Ingest documents

```bash
# Seeds a sample doc automatically if data/sample_docs/ is empty
python ingest_docs.py

# Ingest your own files
python ingest_docs.py path/to/doc.pdf path/to/notes.txt

# Reset the vector store then ingest fresh
python ingest_docs.py --reset
```

### 4a. Run with Streamlit UI

```bash
# Terminal 1 — API server
python main.py

# Terminal 2 — Frontend
streamlit run app.py
# Open http://localhost:8501
```

### 4b. Or use the CLI (no server required)

```bash
python cli.py
```

---

## 🔧 Configuration

All settings live in `mcp_config.yaml`:

```yaml
llm:
  model: mistral          # swap to llama3, phi3, gemma, etc.
  temperature: 0.7

vector_store:
  n_results: 3            # chunks retrieved per query
  embedding_model: all-MiniLM-L6-v2

memory:
  max_history: 20         # conversation turns to keep

tools:
  - name: web_search_fallback
    enabled: true         # set false to disable web search
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status + doc count |
| `POST` | `/ask` | Main RAG query (returns answer + sources) |
| `POST` | `/ingest/text` | Ingest raw text chunks |
| `POST` | `/ingest/file` | Upload PDF or TXT file |
| `GET` | `/search?q=…` | Direct vector search (no LLM) |
| `GET` | `/memory` | View conversation history |
| `POST` | `/memory/new` | Start a new conversation |
| `DELETE` | `/memory/clear` | Clear current conversation |
| `DELETE` | `/store/reset` | Wipe the entire vector store |

Interactive docs: **http://localhost:5000/docs**

---

## ✨ Features

- ✅ **Fully local** — no cloud APIs, no data leaves your machine
- ✅ **Agentic tool routing** — automatically picks vector search vs web fallback
- ✅ **Persistent vector store** — ChromaDB survives restarts
- ✅ **Persistent memory** — multi-conversation JSON store with sliding window
- ✅ **PDF + TXT ingestion** — chunking with overlap via CLI or drag-and-drop UI
- ✅ **Streamlit frontend** — chat, upload, memory viewer, store reset
- ✅ **CLI mode** — no server needed for quick local queries
- ✅ **Swap models easily** — change one line in `mcp_config.yaml`

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

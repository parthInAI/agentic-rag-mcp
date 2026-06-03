# Agentic RAG System with Model Context Protocol

A fully local retrieval-augmented generation pipeline where an AI agent autonomously decides when to retrieve from a vector store, fall back to live web search, or respond from context. Built with a FastAPI MCP server, Streamlit UI, and persistent JSON memory across sessions. No cloud APIs required.

---

## What this does

Most RAG systems retrieve on every query. This one does not. The agent reasons first, then decides:

- **Retrieve** from ChromaDB if the answer is likely in the document store
- **Search the web** if the question is time-sensitive or outside the stored knowledge
- **Respond directly** if context is sufficient

This produces faster, cheaper, and more accurate responses than naive retrieval on every turn.

---

## Architecture

```
User Query
    |
    v
MCP Client (Streamlit UI)
    |
    v
FastAPI MCP Server  <--->  Tool Registry (retrieve / web_search / respond)
    |
    v
LangChain Agent
    |         |
    v         v
ChromaDB   Ollama LLM (local)
(vector    (no cloud API)
 store)
    |
    v
Persistent JSON Memory (cross-session)
```

---

## Tech stack

| Component | Technology |
|---|---|
| Agent orchestration | LangChain |
| MCP server | FastAPI |
| Vector store | ChromaDB |
| Local LLM | Ollama |
| UI | Streamlit |
| Memory | Persistent JSON |
| Embeddings | Sentence Transformers |

---

## Getting started

```bash
# Clone the repo
git clone https://github.com/parthInAI/agentic-rag-mcp
cd agentic-rag-mcp

# Install dependencies
pip install -r requirements.txt

# Pull a local model via Ollama
ollama pull llama3

# Start the MCP server
uvicorn mcp_server:app --reload

# Launch the UI
streamlit run app.py
```

---

## Key design decisions

**Why MCP?** Model Context Protocol standardises how AI agents call tools. Using it here means the agent-tool interface is swappable — any MCP-compatible client can connect to this server without code changes.

**Why local-only?** No API keys, no usage costs, no data leaving your machine. The full pipeline runs on a laptop with 16GB RAM.

**Why persistent memory?** Most demos reset on every session. This one stores conversation history to disk so the agent remembers context across separate runs — closer to how a real production system would behave.

---

## Skills demonstrated

- Production RAG pipeline architecture
- MCP server and client implementation
- Agentic tool-use and autonomous decision-making
- Local LLM deployment with Ollama
- FastAPI REST API design
- Vector database management (ChromaDB)
- Persistent state management

---

## Related projects

- [Agent2Agent Multi-Agent System](https://github.com/parthInAI/agent2agent) — multi-agent orchestration using Google A2A protocol
- [Portfolio](https://parthinai.github.io/) — full project overview

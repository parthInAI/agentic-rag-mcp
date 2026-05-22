"""
main.py
FastAPI-based MCP server exposing the agentic RAG pipeline.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, yaml, os, tempfile

from tools.chromadb_tool import (
    search_documents,
    ingest_texts,
    collection_count,
    reset_collection,
)
from rag_agent import get_rag_response, new_conversation, get_memory_instance

# ── Config ────────────────────────────────────────────────────────────────────
with open("mcp_config.yaml") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(
    title="Agentic RAG MCP Server",
    description="MCP-powered RAG agent with ChromaDB, Ollama, and agent memory.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    conversation_id: str | None = None


class IngestTextRequest(BaseModel):
    texts: list[str]
    source: str = "manual"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "docs_indexed": collection_count(), "model": cfg["llm"]["model"]}


@app.post("/ask")
def ask(req: AskRequest):
    """Main RAG query endpoint."""
    result = get_rag_response(req.query, conversation_id=req.conversation_id)
    return result


@app.post("/ingest/text")
def ingest_text(req: IngestTextRequest):
    """Ingest plain text chunks directly."""
    metas = [{"source": req.source}] * len(req.texts)
    n = ingest_texts(req.texts, metadatas=metas)
    return {"ingested": n, "total": collection_count()}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF or .txt file; it is chunked and ingested into ChromaDB.
    """
    suffix = os.path.splitext(file.filename)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunks = _extract_chunks(tmp_path, suffix, file.filename)
    finally:
        os.unlink(tmp_path)

    if not chunks:
        raise HTTPException(400, "No text could be extracted from the file.")

    metas = [{"source": file.filename}] * len(chunks)
    n = ingest_texts(chunks, metadatas=metas)
    return {"file": file.filename, "chunks_ingested": n, "total": collection_count()}


def _extract_chunks(path: str, suffix: str, filename: str, chunk_size: int = 500) -> list[str]:
    """Extract text from PDF or TXT and split into ~chunk_size char chunks."""
    text = ""
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(400, f"PDF extraction failed: {e}")
    elif suffix in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    # Simple fixed-size chunking with overlap
    words = text.split()
    chunks, buf, buf_len = [], [], 0
    for word in words:
        buf.append(word)
        buf_len += len(word) + 1
        if buf_len >= chunk_size:
            chunks.append(" ".join(buf))
            # 20% overlap
            overlap = max(1, len(buf) // 5)
            buf = buf[-overlap:]
            buf_len = sum(len(w) + 1 for w in buf)
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(c.strip()) > 30]


@app.get("/search")
def search(q: str, n: int = 3):
    """Direct vector search (no LLM)."""
    return {"results": search_documents(q, n_results=n)}


@app.get("/memory")
def memory_history(n: int = 10):
    """Return recent conversation turns."""
    mem = get_memory_instance()
    return {
        "conversation_id": mem.current_id(),
        "turns": mem.get_history(n),
        "all_conversations": [c["id"] for c in mem.all_conversations()],
    }


@app.post("/memory/new")
def start_new_conversation():
    cid = new_conversation()
    return {"conversation_id": cid}


@app.delete("/memory/clear")
def clear_memory():
    get_memory_instance().clear_current()
    return {"status": "cleared"}


@app.delete("/store/reset")
def reset_store():
    reset_collection()
    return {"status": "reset", "total": collection_count()}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    print(f"\n🚀  MCP RAG Server running at http://{host}:{port}")
    print(f"📚  Docs: http://{host}:{port}/docs\n")
    uvicorn.run("main:app", host=host, port=port, reload=True)

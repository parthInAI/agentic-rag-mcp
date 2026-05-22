"""
tools/chromadb_tool.py
ChromaDB vector search tool for the MCP RAG agent.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import yaml, os

# Load config
_cfg_path = os.path.join(os.path.dirname(__file__), "..", "mcp_config.yaml")
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

_vs_cfg = _cfg["vector_store"]

# ── Shared client / collection (module-level singleton) ──────────────────────
_client: chromadb.ClientAPI | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    persist_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", _vs_cfg["persist_directory"])
    )
    os.makedirs(persist_dir, exist_ok=True)

    _client = chromadb.PersistentClient(path=persist_dir)
    emb_fn = SentenceTransformerEmbeddingFunction(
        model_name=_vs_cfg["embedding_model"]
    )
    _collection = _client.get_or_create_collection(
        name=_vs_cfg["collection_name"],
        embedding_function=emb_fn,
    )
    return _collection


# ── Public helpers ────────────────────────────────────────────────────────────

def search_documents(query: str, n_results: int | None = None) -> str:
    """
    Semantic search over the ChromaDB collection.
    Returns a formatted string of the top matching chunks.
    """
    col = _get_collection()
    k = n_results or _vs_cfg.get("n_results", 3)

    if col.count() == 0:
        return "No documents have been ingested yet. Please upload documents first."

    results = col.query(query_texts=[query], n_results=min(k, col.count()))
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "No relevant documents found."

    chunks = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        source = meta.get("source", "unknown") if meta else "unknown"
        score = round(1 - dist, 3)          # cosine similarity proxy
        chunks.append(f"[{i}] (source: {source}, relevance: {score})\n{doc}")

    return "\n\n---\n\n".join(chunks)


def ingest_texts(texts: list[str], metadatas: list[dict] | None = None, ids: list[str] | None = None):
    """Add raw text chunks to the collection."""
    col = _get_collection()
    count = col.count()
    _ids = ids or [f"doc_{count + i}" for i in range(len(texts))]
    _metas = metadatas or [{}] * len(texts)
    col.add(documents=texts, metadatas=_metas, ids=_ids)
    return len(texts)


def collection_count() -> int:
    return _get_collection().count()


def reset_collection():
    col = _get_collection()
    col.delete(ids=[item for item in col.get()["ids"]])

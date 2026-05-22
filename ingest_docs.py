"""
ingest_docs.py
CLI helper to bulk-ingest documents from data/sample_docs/ into ChromaDB.

Usage:
    python ingest_docs.py                        # ingest everything in data/sample_docs/
    python ingest_docs.py path/to/file.pdf       # ingest a single file
    python ingest_docs.py --reset                # wipe the store first, then ingest
"""

import sys, os, argparse
from tools.chromadb_tool import ingest_texts, collection_count, reset_collection

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "sample_docs")


def _chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    chunks, buf, buf_len = [], [], 0
    for word in words:
        buf.append(word)
        buf_len += len(word) + 1
        if buf_len >= chunk_size:
            chunks.append(" ".join(buf))
            overlap = max(1, len(buf) // 5)
            buf = buf[-overlap:]
            buf_len = sum(len(w) + 1 for w in buf)
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(c.strip()) > 30]


def ingest_file(path: str) -> int:
    suffix = os.path.splitext(path)[-1].lower()
    name = os.path.basename(path)

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif suffix in (".txt", ".md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        print(f"  ⚠️  Skipping unsupported file: {name}")
        return 0

    chunks = _chunk_text(text)
    if not chunks:
        print(f"  ⚠️  No text extracted from {name}")
        return 0

    metas = [{"source": name}] * len(chunks)
    n = ingest_texts(chunks, metadatas=metas)
    return n


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("paths", nargs="*", help="Files or directories to ingest")
    parser.add_argument("--reset", action="store_true", help="Reset the store before ingesting")
    args = parser.parse_args()

    if args.reset:
        reset_collection()
        print("🗑️  Vector store reset.\n")

    targets = args.paths or [SAMPLE_DIR]
    total = 0

    for target in targets:
        if os.path.isfile(target):
            print(f"📄 Ingesting: {target}")
            n = ingest_file(target)
            print(f"   ✅ {n} chunks added")
            total += n
        elif os.path.isdir(target):
            for fname in sorted(os.listdir(target)):
                fpath = os.path.join(target, fname)
                if os.path.isfile(fpath):
                    print(f"📄 Ingesting: {fname}")
                    n = ingest_file(fpath)
                    print(f"   ✅ {n} chunks added")
                    total += n
        else:
            print(f"⚠️  Not found: {target}")

    print(f"\n✨ Done. {total} chunks added. Total in store: {collection_count()}")


if __name__ == "__main__":
    # Also seed some built-in sample docs if the sample_docs folder is empty
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    if not os.listdir(SAMPLE_DIR):
        sample_path = os.path.join(SAMPLE_DIR, "intro.txt")
        with open(sample_path, "w") as f:
            f.write("""Model Context Protocol (MCP)
MCP is an open protocol developed by Anthropic that enables AI assistants to integrate modularly with external tools such as databases, APIs, file systems, and search engines. It standardises the way agents communicate with tools, making it easy to swap components without rewriting the core agent logic.

Retrieval-Augmented Generation (RAG)
RAG combines a language model with a retrieval mechanism. Instead of relying purely on parametric knowledge (weights), the model first retrieves relevant passages from an external knowledge base and then conditions its generation on those passages. This dramatically reduces hallucination and allows the system to reason over private or up-to-date documents.

ChromaDB
ChromaDB is an open-source, embedding-first vector database designed for AI applications. It supports local and persistent storage, multiple embedding functions, and metadata filtering. It is commonly used for semantic search and as the retrieval backend in RAG systems.

Ollama
Ollama lets you run large language models locally on your own hardware. It supports a growing list of open-source models including Mistral, LLaMA 3, Phi-3, Gemma, and more. Models are served via a REST API compatible with the OpenAI format.

Agentic RAG
Agentic RAG extends basic RAG by giving the model the ability to decide which tools to call, when to retrieve, and how to combine multiple sources of information. The agent can loop, reflect on intermediate results, and call different tools (vector search, web search, calculators) depending on what the query requires.
""")
        print(f"📝 Created sample document: {sample_path}")
    main()

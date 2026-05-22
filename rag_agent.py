"""
rag_agent.py
Core agentic RAG logic: tool routing, prompt construction, LLM call.
"""

import yaml, os, json, re, requests
from tools.chromadb_tool import search_documents, collection_count
from tools.web_search_tool import web_search
from memory.agent_memory import AgentMemory

# ── Config ────────────────────────────────────────────────────────────────────
_cfg_path = os.path.join(os.path.dirname(__file__), "mcp_config.yaml")
with open(_cfg_path) as f:
    _cfg = yaml.safe_load(f)

_llm_cfg = _cfg["llm"]
_mem_cfg = _cfg["memory"]

# ── Shared memory instance ────────────────────────────────────────────────────
_memory = AgentMemory(
    store_path=os.path.join(os.path.dirname(__file__), _mem_cfg["store_path"]),
    max_history=_mem_cfg["max_history"],
)


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str) -> str:
    endpoint = _llm_cfg["endpoint"]
    payload = {
        "model": _llm_cfg["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": _llm_cfg["max_tokens"],
            "temperature": _llm_cfg["temperature"],
        },
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return (
            "⚠️  Cannot reach Ollama. Make sure it is running: `ollama serve` "
            f"and that model '{_llm_cfg['model']}' is pulled: "
            f"`ollama pull {_llm_cfg['model']}`"
        )
    except Exception as e:
        return f"⚠️  LLM error: {e}"


# ── Tool routing ──────────────────────────────────────────────────────────────

def _decide_tools(query: str) -> dict[str, bool]:
    """Simple heuristic: always try vector search first; use web as fallback."""
    return {
        "document_search": True,
        "web_search": _cfg["tools"][1]["enabled"],  # per config
    }


def _build_prompt(query: str, doc_context: str, web_context: str, history: str) -> str:
    parts = ["You are a helpful AI assistant with access to a knowledge base."]

    if history:
        parts.append(f"\n## Conversation History\n{history}")

    if doc_context and "No documents" not in doc_context and "No relevant" not in doc_context:
        parts.append(f"\n## Retrieved Documents\n{doc_context}")

    if web_context and "unavailable" not in web_context and "No instant" not in web_context:
        parts.append(f"\n## Web Search Results\n{web_context}")

    parts.append(
        "\nUsing the information above (if relevant), answer the following question. "
        "If the documents don't contain enough information, say so and rely on your "
        "general knowledge.\n"
        f"\n## Question\n{query}\n\n## Answer"
    )
    return "\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def get_rag_response(query: str, conversation_id: str | None = None) -> dict:
    """
    Main entry point.  Returns a dict with:
        response, sources, tool_used, conversation_id
    """
    # Switch conversation if requested
    if conversation_id and conversation_id != _memory.current_id():
        _memory._data["current_conversation_id"] = conversation_id

    _memory.add_turn("user", query)
    history = _memory.format_for_prompt(n=5)

    tools = _decide_tools(query)

    # 1. Vector search
    doc_context = ""
    tool_used = []
    if tools["document_search"] and collection_count() > 0:
        doc_context = search_documents(query)
        tool_used.append("document_search")

    # 2. Web fallback – kick in if docs are sparse
    web_context = ""
    if tools["web_search"]:
        needs_web = (
            not doc_context
            or "No documents" in doc_context
            or "No relevant" in doc_context
        )
        if needs_web:
            web_context = web_search(query)
            tool_used.append("web_search")

    # 3. Build prompt & call LLM
    prompt = _build_prompt(query, doc_context, web_context, history)
    answer = _call_ollama(prompt)

    # 4. Persist assistant turn
    _memory.add_turn("assistant", answer)

    return {
        "response": answer,
        "doc_context": doc_context,
        "web_context": web_context,
        "tools_used": tool_used,
        "conversation_id": _memory.current_id(),
        "model": _llm_cfg["model"],
    }


def new_conversation() -> str:
    return _memory.new_conversation()


def get_memory_instance() -> AgentMemory:
    return _memory

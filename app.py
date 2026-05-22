"""
app.py
Streamlit frontend for the Agentic RAG MCP system.
"""

import streamlit as st
import requests, json, os, time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("RAG_API_BASE", "http://localhost:5000")

st.set_page_config(
    page_title="Agentic RAG · MCP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #0d0f14; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330;
}

.chat-bubble-user {
    background: #1a2035;
    border-left: 3px solid #4f9cf9;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.95rem;
}
.chat-bubble-assistant {
    background: #12181f;
    border-left: 3px solid #22c55e;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.95rem;
}
.tag {
    display: inline-block;
    background: #1e2a3a;
    color: #4f9cf9;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 7px;
    border-radius: 4px;
    margin-right: 4px;
    border: 1px solid #2a3a52;
}
.tag-green { color: #22c55e; border-color: #1a3a2a; background: #0f2a1a; }
.tag-yellow { color: #eab308; border-color: #3a340a; background: #2a240a; }
.metric-box {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 600; color: #4f9cf9; font-family: 'IBM Plex Mono', monospace; }
.metric-lbl { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
hr { border-color: #1e2330 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "server_ok" not in st.session_state:
    st.session_state.server_ok = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_server():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def ask(query: str):
    payload = {"query": query, "conversation_id": st.session_state.conversation_id}
    r = requests.post(f"{API_BASE}/ask", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def ingest_text_api(texts: list[str], source: str = "manual"):
    r = requests.post(f"{API_BASE}/ingest/text", json={"texts": texts, "source": source}, timeout=30)
    r.raise_for_status()
    return r.json()


def ingest_file_api(file_bytes: bytes, filename: str):
    r = requests.post(
        f"{API_BASE}/ingest/file",
        files={"file": (filename, file_bytes)},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def get_memory():
    r = requests.get(f"{API_BASE}/memory", timeout=10)
    r.raise_for_status()
    return r.json()


def new_conv():
    r = requests.post(f"{API_BASE}/memory/new", timeout=10)
    r.raise_for_status()
    return r.json()["conversation_id"]


def clear_mem():
    r = requests.delete(f"{API_BASE}/memory/clear", timeout=10)
    r.raise_for_status()


def reset_store():
    r = requests.delete(f"{API_BASE}/store/reset", timeout=10)
    r.raise_for_status()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Agentic RAG")
    st.markdown('<hr>', unsafe_allow_html=True)

    # Server status
    health = check_server()
    if health:
        st.session_state.server_ok = True
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-val">{health["docs_indexed"]}</div><div class="metric-lbl">Docs indexed</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-val" style="font-size:1rem">{health["model"]}</div><div class="metric-lbl">Model</div></div>', unsafe_allow_html=True)
        st.success("Server online", icon="✅")
    else:
        st.error("Server offline. Run: `python main.py`", icon="🔴")

    st.markdown('<hr>', unsafe_allow_html=True)

    # Document ingestion
    st.markdown("### 📥 Ingest Documents")

    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt", "md"], key="uploader")
    if uploaded and st.button("Ingest File", use_container_width=True):
        with st.spinner("Ingesting…"):
            try:
                result = ingest_file_api(uploaded.read(), uploaded.name)
                st.success(f"✅ {result['chunks_ingested']} chunks from *{uploaded.name}*")
            except Exception as e:
                st.error(f"Error: {e}")

    with st.expander("Paste raw text"):
        raw = st.text_area("Text to ingest", height=120, key="raw_text")
        src = st.text_input("Source label", value="manual", key="raw_src")
        if st.button("Ingest Text", use_container_width=True):
            if raw.strip():
                try:
                    result = ingest_text_api([raw.strip()], source=src)
                    st.success(f"✅ Ingested. Total: {result['total']}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("🗑️ Reset Vector Store", use_container_width=True):
        try:
            reset_store()
            st.warning("Vector store cleared.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown('<hr>', unsafe_allow_html=True)

    # Memory controls
    st.markdown("### 💾 Memory")
    if st.button("➕ New Conversation", use_container_width=True):
        if st.session_state.server_ok:
            cid = new_conv()
            st.session_state.conversation_id = cid
            st.session_state.messages = []
            st.success(f"Started: `{cid[:20]}…`")

    if st.button("🧹 Clear Current Memory", use_container_width=True):
        if st.session_state.server_ok:
            clear_mem()
            st.session_state.messages = []
            st.info("Memory cleared.")

    with st.expander("View raw memory"):
        if st.button("Fetch", key="fetch_mem"):
            try:
                mem = get_memory()
                st.code(json.dumps(mem, indent=2), language="json")
            except Exception as e:
                st.error(str(e))


# ── Main chat area ────────────────────────────────────────────────────────────

st.markdown("# Agentic RAG — MCP Interface")
st.markdown(
    "Chat with your documents. The agent will search your knowledge base, "
    "fall back to web search if needed, and remember the conversation."
)
st.markdown("---")

# Render message history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        # Tools used tags
        tools_html = ""
        for t in msg.get("tools_used", []):
            cls = "tag-green" if t == "document_search" else "tag-yellow"
            label = "📚 doc_search" if t == "document_search" else "🌐 web_search"
            tools_html += f'<span class="tag {cls}">{label}</span>'

        st.markdown(
            f'<div class="chat-bubble-assistant">'
            f'<div style="margin-bottom:6px">{tools_html}</div>'
            f'🤖 {msg["content"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Expandable context
        if msg.get("doc_context") and "No documents" not in msg["doc_context"]:
            with st.expander("📄 Retrieved document context"):
                st.text(msg["doc_context"])
        if msg.get("web_context") and "unavailable" not in msg["web_context"]:
            with st.expander("🌐 Web search context"):
                st.text(msg["web_context"])

# Input
st.markdown("---")
query = st.chat_input("Ask a question…", disabled=not st.session_state.server_ok)

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking…"):
        try:
            result = ask(query)
            st.session_state.conversation_id = result.get("conversation_id")
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "tools_used": result.get("tools_used", []),
                "doc_context": result.get("doc_context", ""),
                "web_context": result.get("web_context", ""),
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error calling the RAG server: {e}",
                "tools_used": [],
            })

    st.rerun()

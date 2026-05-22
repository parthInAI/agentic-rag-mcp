"""
cli.py
Simple CLI to query the RAG agent directly (no server needed).
"""

import sys, os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from rag_agent import get_rag_response, new_conversation

BANNER = """
╔══════════════════════════════════════════════╗
║   🧠  Agentic RAG — MCP CLI Interface        ║
║   Type your question, or:                    ║
║     /new   – start a new conversation        ║
║     /quit  – exit                            ║
╚══════════════════════════════════════════════╝
"""

def main():
    print(BANNER)
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("/quit", "/exit", "q"):
            print("Bye!")
            break
        if query.lower() == "/new":
            cid = new_conversation()
            print(f"  ✅ New conversation: {cid}")
            continue

        print("\n⏳ Thinking…")
        result = get_rag_response(query)
        print(f"\n🤖 [{', '.join(result['tools_used']) or 'no tools'}]\n")
        print(result["response"])
        print(f"\n  conversation: {result['conversation_id'][:30]}…")


if __name__ == "__main__":
    main()

"""
tools/web_search_tool.py
DuckDuckGo-based web search fallback tool (no API key needed).
"""

import urllib.request, urllib.parse, json, re


def web_search(query: str, max_results: int = 3) -> str:
    """
    Lightweight DuckDuckGo Instant Answer API search.
    Returns a summary string; falls back to a direct snippet scrape.
    """
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agentic-rag/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())

        results = []

        # Abstract (best single answer)
        if data.get("AbstractText"):
            src = data.get("AbstractURL", "DuckDuckGo")
            results.append(f"Summary: {data['AbstractText']}\nSource: {src}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                href = topic.get("FirstURL", "")
                results.append(f"- {topic['Text']}\n  {href}")

        if results:
            return "\n\n".join(results)
        return f"No instant answer found for: {query}"

    except Exception as e:
        return f"Web search unavailable ({e}). Please check your internet connection."

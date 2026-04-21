import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime


def get_datetime() -> str:
    """Return current date/time as a readable string. Never raises."""
    try:
        return f"Current date and time: {datetime.now().strftime('%A, %d %B %Y, %H:%M:%S')}"
    except Exception as exc:
        return f"Datetime error: {exc}"


def arxiv_search(topic: str) -> str:
    """Search ArXiv by title for up to 3 results. Never raises."""
    try:
        url = f"https://export.arxiv.org/api/query?search_query=ti:{topic}&max_results=3"
        resp = requests.get(url, timeout=10)
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        if not entries:
            return "No ArXiv results found."

        results = []
        for entry in entries:
            title = entry.find("atom:title", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.strip()
            results.append(f"Title: {title}\nSummary: {summary[:300]}")
        return "\n\n".join(results)

    except Exception as exc:
        return f"ArXiv search error: {exc}"


def web_research_search(query: str) -> str:
    """Search web research context via DuckDuckGo instant answers. Never raises."""
    try:
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1,
        }
        resp = requests.get("https://api.duckduckgo.com/", params=params, timeout=10)
        data = resp.json()

        snippets: list[str] = []
        abstract = str(data.get("AbstractText", "")).strip()
        if abstract:
            source = str(data.get("AbstractSource", "Web")).strip()
            heading = str(data.get("Heading", "")).strip()
            title = heading or query
            snippets.append(f"Title: {title}\nSource: {source}\nSummary: {abstract[:500]}")

        related = data.get("RelatedTopics", []) or []
        count = 0
        for item in related:
            if count >= 4:
                break
            # RelatedTopics can include grouped entries under "Topics".
            candidates = item.get("Topics", []) if isinstance(item, dict) and "Topics" in item else [item]
            for entry in candidates:
                if count >= 4:
                    break
                if not isinstance(entry, dict):
                    continue
                text = str(entry.get("Text", "")).strip()
                url = str(entry.get("FirstURL", "")).strip()
                if text:
                    snippets.append(f"Result: {text[:420]}\nLink: {url}" if url else f"Result: {text[:420]}")
                    count += 1

        if not snippets:
            return "No web research snippets found for this query."

        return "\n\n".join(snippets)
    except Exception as exc:
        return f"Web search error: {exc}"


def route_tool(question: str, allow_web_search: bool = True) -> str:
    """Dispatch to the right tool based on question keywords."""
    q = question.lower()
    if any(word in q for word in ["date", "time", "today"]):
        return get_datetime()

    web_intent = any(
        phrase in q
        for phrase in ["search web", "web search", "research web", "internet", "online", "latest"]
    )
    if web_intent:
        if not allow_web_search:
            return (
                "Web context is disabled for this session. "
                "Enable 'Web context' in the sidebar or ask a PDF/ArXiv question."
            )
        match = re.search(r"(?:search web for|web search for|about|on)\s+(.+)", q)
        topic = match.group(1).strip() if match else question
        return web_research_search(topic)

    match = re.search(r"(?:search for|papers on|find papers on|about)\s+(.+)", q)
    topic = match.group(1).strip() if match else question
    return arxiv_search(topic)

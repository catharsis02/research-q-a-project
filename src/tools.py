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


def route_tool(question: str) -> str:
    """Dispatch to the right tool based on question keywords."""
    q = question.lower()
    if any(word in q for word in ["date", "time", "today"]):
        return get_datetime()
    match = re.search(r"(?:search for|papers on|find papers on|about)\s+(.+)", q)
    topic = match.group(1) if match else question
    return arxiv_search(topic)

from __future__ import annotations

from pathlib import Path
import hashlib

from src.extractor import extract_all
from src.knowledge_base import build_kb
from src.graph import build_graph, ask, build_filter_map


def _discover_papers(papers_dir: Path) -> dict[str, str]:
    """Return {label: path} from all PDFs in the papers directory."""
    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {papers_dir}")

    discovered: dict[str, str] = {}
    for pdf_path in pdf_paths:
        label = pdf_path.stem
        discovered[label] = str(pdf_path)
    return discovered


def _collection_name_from_papers(papers: dict[str, str]) -> str:
    signature = "|".join(f"{label}:{path}" for label, path in sorted(papers.items()))
    digest = hashlib.md5(signature.encode()).hexdigest()[:12]
    return f"phd_papers_{digest}"


if __name__ == "__main__":
    papers_dir = Path("papers")
    papers = _discover_papers(papers_dir)

    # Use labels as paper IDs so filters and sources are always generic.
    paper_meta = {label: label for label in papers}

    raw_texts = extract_all(papers)
    collection_name = _collection_name_from_papers(papers)
    collection = build_kb(raw_texts, paper_meta, collection_name=collection_name)
    filter_map = build_filter_map(paper_meta)
    app = build_graph(collection, filter_map)

    print(f"Loaded {len(papers)} paper(s) from {papers_dir}")
    print("Ask a question (or 'exit' to quit):")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        result = ask(question, app, thread_id="main_cli")
        print(f"\n{result['answer']}\n")

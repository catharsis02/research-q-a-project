import contextlib
import chromadb
import re
from src.config import embedder

CHUNK_LABELS = [
    "Abstract-Introduction",
    "Methodology-Architecture",
    "Results-Conclusion",
]

MIN_CHUNK_WORDS = 150
MAX_CHUNK_WORDS = 400
SHORT_CHUNK_EXTENSION = 10  # percent

def chunk_paper(label: str, text: str, paper_id: str) -> list[dict]:
    """Split text into 3 chunks by character range."""
    total_chars = len(text)
    cuts = [0, total_chars // 3, 2 * total_chars // 3, total_chars]
    chunks = []

    for i, chunk_label in enumerate(CHUNK_LABELS):
        segment = text[cuts[i]:cuts[i + 1]].strip()

        words = segment.split()
        if len(words) < MIN_CHUNK_WORDS:
            extra_end = min(cuts[i + 1] + total_chars // SHORT_CHUNK_EXTENSION, total_chars)
            segment = text[cuts[i]:extra_end].strip()

        segment = " ".join(segment.split()[:MAX_CHUNK_WORDS])

        chunks.append({
            "id": f"doc_{label}_{i + 1:02d}",
            "topic": f"{label} — {chunk_label}",
            "paper": paper_id,
            "text": segment,
        })

    return chunks

def build_kb(
    raw_texts: dict,
    paper_meta: dict,
    collection_name: str = "phd_papers",
) -> chromadb.Collection:
    """Build a ChromaDB collection from extracted paper texts.

    raw_texts:  {label: full_text}
    paper_meta: {label: arxiv_base_id_or_label}
    """
    client = chromadb.Client()
    with contextlib.suppress(Exception):
        client.delete_collection(name=collection_name)
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    all_docs = []
    for label, text in raw_texts.items():
        paper_id = paper_meta.get(label, label)
        chunks = chunk_paper(label, text, paper_id)
        all_docs.extend(chunks)

    expected_chunks = len(raw_texts) * 3
    assert len(all_docs) == expected_chunks, (
        f"Expected {expected_chunks} chunks (3 per paper), "
        f"got {len(all_docs)}. Check that chunk_paper "
        f"returns exactly 3 chunks per paper."
    )

    embeddings = embedder.encode([d["text"] for d in all_docs]).tolist()

    collection.add(
        documents=[d["text"] for d in all_docs],
        embeddings=embeddings,
        ids=[d["id"] for d in all_docs],
        metadatas=[{"topic": d["topic"], "paper": d["paper"]} for d in all_docs],
    )

    print(f"KB built: {collection.count()} chunks from {len(raw_texts)} paper(s)")
    return collection

def _query_from_source_text(source_text: str, max_words: int = 18) -> str:
    """Build a fallback gate query from informative paper text."""
    compact = re.sub(r"\s+", " ", source_text or "").strip()
    if not compact:
        return ""

    lower = compact.lower()
    start = lower.find("abstract")
    if start != -1:
        compact = compact[start + len("abstract"):].strip(" :.-")
    else:
        compact = compact[len(compact) // 6:]

    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", compact)
    if not words:
        return ""

    stop = {
        "copyright", "license", "permission", "provided", "google", "grants",
        "rights", "reserved", "arxiv", "http", "https", "using", "paper",
    }
    filtered = [w for w in words if w.lower() not in stop]
    tokens = filtered if filtered else words
    return " ".join(tokens[:max_words])

def _fallback_queries(gate_query: str, expected_paper_id: str, source_text: str = "") -> list[str]:
    """Generate retry queries, ordered from strict to broader signals."""
    candidates = [gate_query.strip()]

    id_as_terms = re.sub(r"[^\w]+", " ", expected_paper_id).strip()
    if id_as_terms:
        candidates.append(id_as_terms)

    text_query = _query_from_source_text(source_text)
    if text_query:
        candidates.append(text_query)

    seen = set()
    ordered = []
    for query in candidates:
        q = query.strip()
        if q and q not in seen:
            seen.add(q)
            ordered.append(q)
    return ordered


def retrieval_gate(
    collection: chromadb.Collection,
    gate_query: str,
    expected_paper_id: str,
    source_text: str = "",
) -> bool:
    """Sanity-check that expected paper appears in top-3 results."""
    attempts = _fallback_queries(gate_query, expected_paper_id, source_text=source_text)
    diagnostics = []

    for idx, query in enumerate(attempts, start=1):
        emb = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=emb, n_results=3)
        papers = [m["paper"] for m in results["metadatas"][0]]
        topics = [m["topic"] for m in results["metadatas"][0]]
        diagnostics.append((idx, query, papers))

        if expected_paper_id in papers:
            print(f"RETRIEVAL GATE PASSED — '{expected_paper_id}' found in top-3")
            print(f"   Query  : {query}")
            print(f"   Topics : {topics}")
            if idx > 1:
                print(f"   Note   : Passed on fallback attempt #{idx}")
            return True

    details = "\n".join(
        f"  Attempt {idx}: query='{query}' -> {papers}"
        for idx, query, papers in diagnostics
    )
    raise AssertionError(
        "RETRIEVAL GATE FAILED\n"
        f"  Expected: {expected_paper_id}\n"
        f"{details}\n"
        f"  Fix     : Check chunk quality for {expected_paper_id} — "
        "the text may be too short, too noisy, or poorly extracted for embedding."
    )

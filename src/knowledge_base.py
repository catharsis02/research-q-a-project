import contextlib
import chromadb
from src.config import embedder

CHUNK_LABELS = [
    "Abstract-Introduction",
    "Methodology-Architecture",
    "Results-Conclusion",
]

# Chunks shorter than this are padded — ensures enough content for retrieval
MIN_CHUNK_WORDS = 150
# Hard cap so individual chunks don't blow up LLM context windows
MAX_CHUNK_WORDS = 400
# How much extra text to grab when a chunk is too short (fraction of doc)
SHORT_CHUNK_EXTENSION = 10  # percent


def chunk_paper(label: str, text: str, paper_id: str) -> list[dict]:
    """Split text into 3 equal thirds by character count.

    Each chunk gets padded to MIN_CHUNK_WORDS or trimmed to MAX_CHUNK_WORDS
    so retrieval quality stays consistent regardless of paper length.
    """
    total_chars = len(text)
    cuts = [0, total_chars // 3, 2 * total_chars // 3, total_chars]
    chunks = []

    for i, chunk_label in enumerate(CHUNK_LABELS):
        segment = text[cuts[i]:cuts[i + 1]].strip()

        words = segment.split()
        if len(words) < MIN_CHUNK_WORDS:
            # Extend forward by 10% of the doc to capture enough content
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

    print(f"✅ KB built: {collection.count()} chunks from {len(raw_texts)} paper(s)")
    return collection


def retrieval_gate(collection: chromadb.Collection, gate_query: str, expected_paper_id: str) -> bool:
    """Sanity-check that the expected paper surfaces in top-3 for gate_query.

    Catches embedding or chunking bugs early — if the most obvious query
    for a paper doesn't retrieve it, something is wrong with the KB.
    """
    emb = embedder.encode([gate_query]).tolist()
    results = collection.query(query_embeddings=emb, n_results=3)
    papers = [m["paper"] for m in results["metadatas"][0]]
    topics = [m["topic"] for m in results["metadatas"][0]]

    if expected_paper_id not in papers:
        raise AssertionError(
            f"RETRIEVAL GATE FAILED\n"
            f"  Query   : {gate_query}\n"
            f"  Expected: {expected_paper_id}\n"
            f"  Got     : {papers}\n"
            f"  Fix     : Check chunk quality for {expected_paper_id} — "
            f"the text may be too short or garbled for embedding."
        )

    print(f"✅ RETRIEVAL GATE PASSED — '{expected_paper_id}' found in top-3")
    print(f"   Query  : {gate_query}")
    print(f"   Topics : {topics}")
    return True

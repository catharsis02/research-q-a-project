import streamlit as st
import uuid
import hashlib
from pathlib import Path
from src.extractor import extract_all
from src.knowledge_base import build_kb, retrieval_gate
from src.graph import build_graph, ask, build_filter_map
from src.state import make_initial_state

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Upload sidebar ───────────────────────────────────────
with st.sidebar:
    st.title("📄 Research Paper Q&A")
    st.markdown("Upload **1 to 5 research PDFs** to begin.")

    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 1 to 5 PDF files to build your KB.",
    )

    # Label inputs — one per uploaded file
    paper_labels = {}
    if uploaded_files:
        st.markdown("**Label each paper** (optional):")
        for f in uploaded_files:
            label = st.text_input(
                f"Label for {f.name}",
                value=f.name.replace(".pdf", ""),
                key=f"label_{f.name}",
            )
            paper_labels[f.name] = label

    st.divider()
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    if "thread_id" in st.session_state:
        st.caption(f"Session: {st.session_state.thread_id[:8]}")


# ── KB build (cached per unique file set) ────────────────
def file_hash(files) -> str:
    """Stable hash of uploaded file names + sizes for cache keying."""
    sig = "_".join(
        f"{f.name}:{f.size}" for f in sorted(files, key=lambda x: x.name)
    )
    return hashlib.md5(sig.encode()).hexdigest()


def versioned_filename(original_name: str, file_bytes: bytes) -> str:
    """Return a deterministic filename using content hash to avoid collisions."""
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".pdf"
    digest = hashlib.md5(file_bytes).hexdigest()[:10]
    return f"{stem}__{digest}{suffix}"


@st.cache_resource
def build_session_resources(_files, labels: dict, _file_hash: str):
    """Extract text, build KB, run retrieval gate, compile graph.

    _files and _file_hash are underscore-prefixed so Streamlit
    skips hashing them (it uses _file_hash as the cache key instead).
    """
    papers_dir = Path("papers")
    papers_dir.mkdir(parents=True, exist_ok=True)

    sources = {}
    paper_meta = {}

    for f in _files:
        label = labels.get(f.name, f.name)
        f.seek(0)
        file_bytes = f.read()
        safe_name = Path(versioned_filename(f.name, file_bytes)).name
        output_path = papers_dir / safe_name
        output_path.write_bytes(file_bytes)
        sources[label] = str(output_path)
        paper_meta[label] = label

    raw_texts = extract_all(sources)

    collection_name = f"phd_papers_{_file_hash[:12]}"
    collection = build_kb(raw_texts, paper_meta, collection_name=collection_name)

    # Gate on first paper — if its own words don't retrieve it, KB is broken
    first_label = list(raw_texts.keys())[0]
    first_words = raw_texts[first_label].split()[:8]
    gate_query = " ".join(first_words)
    retrieval_gate(collection, gate_query=gate_query, expected_paper_id=first_label)

    filter_map = build_filter_map(paper_meta)
    app = build_graph(collection, filter_map)
    topics = [d["topic"] for d in collection.get()["metadatas"]]
    return app, paper_meta, topics


# ── Main UI ──────────────────────────────────────────────
if not uploaded_files or len(uploaded_files) == 0:
    st.info("👈 Upload up to 5 research PDFs in the sidebar to begin.")
    st.stop()

MAX_PAPERS = 5
if len(uploaded_files) > MAX_PAPERS:
    st.warning(f"Please upload at most {MAX_PAPERS} PDFs. You uploaded {len(uploaded_files)}.")
    st.stop()

with st.spinner("Extracting text, building KB, compiling graph..."):
    fhash = file_hash(uploaded_files)
    app, paper_meta, topics = build_session_resources(
        _files=uploaded_files,
        labels=paper_labels,
        _file_hash=fhash,
    )

st.success(
    f"✅ {len(uploaded_files)} paper(s) loaded. "
    f"{len(uploaded_files) * 3} chunks indexed. "
    f"Saved to papers/. Start asking!"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Papers loaded summary
with st.sidebar:
    st.markdown(f"📚 **{len(uploaded_files)} paper(s) loaded**")
    for label in paper_meta:
        st.markdown(f"- {label}")

# Chat history display
st.header("Ask questions about your papers")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("faithfulness") is not None:
            st.caption(
                f"Faithfulness: {msg['faithfulness']:.2f} | "
                f"Route: {msg.get('route', '')}"
            )
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Chat input
if prompt := st.chat_input("Ask anything about your uploaded papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(prompt, app, thread_id=st.session_state.thread_id)
        st.write(result["answer"])
        st.caption(
            f"Faithfulness: {result['faithfulness']:.2f} | "
            f"Route: {result['route']}"
        )
        if result.get("sources"):
            with st.expander("📚 Sources"):
                for s in result["sources"]:
                    st.markdown(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "faithfulness": result["faithfulness"],
        "route": result["route"],
        "sources": result.get("sources", []),
    })

st.divider()
st.caption("Answers are grounded in your uploaded PDFs only. Always verify against the original documents.")

import streamlit as st
import uuid
import hashlib
import re
from pathlib import Path
from src.extractor import extract_all
from src.knowledge_base import build_kb, retrieval_gate
from src.graph import build_graph, ask, build_filter_map

st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon=None,
    layout="wide",
)

with st.sidebar:
    st.title("Research Paper Q&A")
    st.markdown("Upload **1 to 5 research PDFs** to begin.")

    uploaded_files = st.file_uploader(
        "Upload PDF papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload 1 to 5 PDF files to build your KB.",
    )

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
    enable_web_context = st.toggle(
        "Enable web context",
        value=False,
        help="When on, the assistant may use web research snippets for broader context.",
    )
    st.caption(
        "Off = strict local PDFs/ArXiv/date tools only. "
        "On = allows web search when explicitly requested."
    )
    st.divider()
    if st.button("New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    if "thread_id" in st.session_state:
        st.caption(f"Session: {st.session_state.thread_id[:8]}")


def file_hash(files) -> str:
    """Stable hash of uploaded file names and sizes."""
    sig = "_".join(
        f"{f.name}:{f.size}" for f in sorted(files, key=lambda x: x.name)
    )
    return hashlib.md5(sig.encode()).hexdigest()


def versioned_filename(original_name: str, file_bytes: bytes) -> str:
    """Build deterministic filename from content hash."""
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix or ".pdf"
    digest = hashlib.md5(file_bytes).hexdigest()[:10]
    return f"{stem}__{digest}{suffix}"


def build_gate_query_from_text(text: str, max_words: int = 18) -> str:
    """Build retrieval gate query from informative paper text."""
    compact = re.sub(r"\s+", " ", text).strip()
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


@st.cache_resource
def build_session_resources(
    _files,
    labels: dict,
    _file_hash: str,
    allow_web_context: bool,
):
    """Extract text, build KB, run retrieval gate, and compile graph."""
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

    first_label = list(raw_texts.keys())[0]
    gate_query = build_gate_query_from_text(raw_texts[first_label])
    if not gate_query:
        gate_query = first_label.replace("_", " ")
    retrieval_gate(
        collection,
        gate_query=gate_query,
        expected_paper_id=first_label,
        source_text=raw_texts[first_label],
    )

    filter_map = build_filter_map(paper_meta)
    app = build_graph(collection, filter_map, allow_web_search=allow_web_context)
    topics = [d["topic"] for d in collection.get()["metadatas"]]
    return app, paper_meta, topics


if not uploaded_files or len(uploaded_files) == 0:
    st.info("Upload up to 5 research PDFs in the sidebar to begin.")
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
        allow_web_context=enable_web_context,
    )

st.success(
    f"{len(uploaded_files)} paper(s) loaded. "
    f"{len(uploaded_files) * 3} chunks indexed. "
    f"Saved to papers/. Start asking!"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.markdown(f"**{len(uploaded_files)} paper(s) loaded**")
    for label in paper_meta:
        st.markdown(f"- {label}")

st.header("Ask questions about your papers")
st.caption(f"Web context: {'ON' if enable_web_context else 'OFF'}")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("faithfulness") is not None:
            st.caption(
                f"Faithfulness: {msg['faithfulness']:.2f} | "
                f"Route: {msg.get('route', '')}"
            )
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if prompt := st.chat_input("Ask anything about your uploaded papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(prompt, app, thread_id=st.session_state.thread_id)
        st.markdown(result["answer"])
        st.caption(
            f"Faithfulness: {result['faithfulness']:.2f} | "
            f"Route: {result['route']}"
        )
        if result.get("sources"):
            with st.expander("Sources"):
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

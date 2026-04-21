import fitz
import re


# Fewer than this many chars usually means a scanned/image-only PDF
# that PyMuPDF can't extract text from
MIN_EXTRACTABLE_CHARS = 500


def extract_text(source) -> str:
    """Extract full text from a PDF file path or in-memory file object."""
    if isinstance(source, str):
        doc = fitz.open(source)
    else:
        # Streamlit UploadedFile or BytesIO — read bytes if needed
        data = source.read() if hasattr(source, "read") else source
        doc = fitz.open(stream=data, filetype="pdf")

    pages = [page.get_text() for page in doc]
    text = "\n".join(pages).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) < MIN_EXTRACTABLE_CHARS:
        raise ValueError(
            f"Extracted only {len(text)} chars from PDF. "
            f"Minimum is {MIN_EXTRACTABLE_CHARS}. "
            f"The file may be scanned/image-based — "
            f"try a text-layer PDF instead."
        )
    return text


def extract_all(sources: dict) -> dict:
    """Batch-extract text from multiple PDFs. Prints a preview per paper."""
    results = {}
    for label, src in sources.items():
        text = extract_text(src)
        results[label] = text
        print(f"✅ {label}: {len(text)} chars extracted")
        print(f"   Preview: {text[:500]}\n")
    return results

import fitz
import re

MIN_EXTRACTABLE_CHARS = 500

def extract_text(source) -> str:
    """Extract full text from a PDF source."""
    if isinstance(source, str):
        doc = fitz.open(source)
    else:
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
    """Batch extract text from multiple PDFs."""
    results = {}
    for label, src in sources.items():
        text = extract_text(src)
        results[label] = text
        print(f"{label}: {len(text)} chars extracted")
        print(f"   Preview: {text[:500]}\n")
    return results

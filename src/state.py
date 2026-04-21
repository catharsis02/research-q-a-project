from typing import TypedDict


class ResearchState(TypedDict):
    question: str
    messages: list[dict]
    route: str
    retrieved: str
    sources: list[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    paper_filter: str
    user_name: str


# Maps lowercase substrings found in questions to arxiv base IDs.
# Extended at runtime when user uploads PDFs — see graph.py build_filter_map()
DEFAULT_FILTER_MAP: dict[str, str] = {
    "transformer": "1706.03762",
    "attention is all": "1706.03762",
    "bert": "1810.04805",
    "rag": "2005.11401",
    "retrieval augmented": "2005.11401",
    "react": "2210.03629",
    "ragas": "2309.15217",
}


def make_initial_state(question: str) -> ResearchState:
    """Create a blank state seeded with a question."""
    return ResearchState(
        question=question,
        messages=[],
        route="",
        retrieved="",
        sources=[],
        tool_result="",
        answer="",
        faithfulness=1.0,
        eval_retries=0,
        paper_filter="",
        user_name="",
    )

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import ResearchState, make_initial_state
from src.nodes import make_nodes
import chromadb


def route_decision(state: ResearchState) -> str:
    r = state.get("route", "")
    if r == "retrieve":
        return "retrieve"
    if r == "tool":
        return "tool"
    return "skip"


def eval_decision(state: ResearchState) -> str:
    """Retry answer generation if faithfulness is low, up to 2 attempts."""
    if state.get("faithfulness", 1.0) < 0.7 and state.get("eval_retries", 0) < 2:
        return "answer"
    return "save"


def build_graph(
    collection: chromadb.Collection,
    filter_map: dict = None,
    allow_web_search: bool = True,
):
    """Build and compile the LangGraph StateGraph.

    Called once per Streamlit session after KB is built.
    Returns a compiled app ready for invoke().
    """
    nodes = make_nodes(
        collection,
        filter_map,
        allow_web_search=allow_web_search,
    )

    g = StateGraph(ResearchState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_edge("save", END)

    g.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    g.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"},
    )

    app = g.compile(checkpointer=MemorySaver())
    print("✅ Graph compiled successfully")
    return app


def ask(question: str, app, thread_id: str = "default") -> ResearchState:
    """Run a question through the compiled graph."""
    config = {"configurable": {"thread_id": thread_id}}
    initial = make_initial_state(question)
    return app.invoke(initial, config=config)


def build_filter_map(paper_meta: dict[str, str]) -> dict[str, str]:
    """Build a filter map from uploaded paper labels only."""
    fm: dict[str, str] = {}
    for label, paper_id in paper_meta.items():
        fm[label.lower()] = paper_id
    return fm

from src.graph import ask

def run_ragas(app, qa_pairs: list[dict], thread_prefix: str = "ragas") -> dict:
    """Run QA pairs and score with RAGAS or fallback scoring."""
    records = []
    for i, pair in enumerate(qa_pairs):
        result = ask(pair["question"], app, thread_id=f"{thread_prefix}_{i}")
        records.append({
            "question": pair["question"],
            "answer": result["answer"],
            "contexts": [result["retrieved"]],
            "ground_truth": pair["ground_truth"],
        })

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset

        dataset = Dataset.from_list(records)
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        print(scores.to_pandas().to_string())
        print(f"\nMean Faithfulness : {scores['faithfulness']:.3f}")
        print(f"Mean Relevancy    : {scores['answer_relevancy']:.3f}")
        print(f"Mean Precision    : {scores['context_precision']:.3f}")
        return dict(scores)

    except ImportError:
        print("RAGAS not installed — falling back to manual LLM scoring")
        return _manual_score(records)

def _manual_score(records: list[dict]) -> dict:
    """LLM-based faithfulness scoring when RAGAS isn't available."""
    from src.config import llm
    from langchain_core.messages import HumanMessage

    SCORE_CONTEXT_LIMIT = 1000

    scores = []
    for r in records:
        prompt = (
            "Rate faithfulness 0.0–1.0. Decimal only.\n"
            f"Context: {r['contexts'][0][:SCORE_CONTEXT_LIMIT]}\n"
            f"Answer : {r['answer']}"
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            score = float(resp.content.strip())
        except Exception:
            score = 0.5
        scores.append(score)
        print(f"  Q: {r['question'][:50]} → {score:.2f}")

    mean = sum(scores) / len(scores)
    print(f"\nManual mean faithfulness: {mean:.3f}")
    return {"faithfulness": mean}

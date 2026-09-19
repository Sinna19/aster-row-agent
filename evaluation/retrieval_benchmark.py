"""Reproducible document-retrieval benchmark for lexical, semantic, and hybrid search.

Run from the repository root:
    python -m evaluation.retrieval_benchmark
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from app.retriever import Retriever

ROOT = os.path.join(os.path.dirname(__file__), "..")
KB_DIR = os.path.join(ROOT, "knowledge-base")
CASES_PATH = os.path.join(os.path.dirname(__file__), "retrieval-benchmark-cases.json")
MODES = ("lexical", "semantic", "hybrid")
K_VALUES = (1, 3)


def load_cases(path: str = CASES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ranked_doc_ids(retriever: Retriever, query: str, mode: str, top_k: int) -> list[str]:
    """Deduplicate heading-level chunks so metrics evaluate document retrieval."""
    seen = set()
    docs = []
    for result in retriever.search(query, top_k=len(retriever.chunks), ranking=mode):
        if result.chunk.doc_id not in seen:
            seen.add(result.chunk.doc_id)
            docs.append(result.chunk.doc_id)
        if len(docs) == top_k:
            break
    return docs


def evaluate(retriever: Retriever, cases: list[dict], top_k: int = 3) -> dict[str, dict[str, float]]:
    """Return macro Recall@K, Hit@K, and MRR over a fixed labelled query set."""
    metrics = defaultdict(lambda: defaultdict(float))
    for mode in MODES:
        for case in cases:
            relevant = set(case["relevant_doc_ids"])
            ranked = ranked_doc_ids(retriever, case["query"], mode, top_k)
            for k in K_VALUES:
                retrieved = set(ranked[:k])
                metrics[mode][f"recall@{k}"] += len(relevant & retrieved) / len(relevant)
                metrics[mode][f"hit@{k}"] += float(bool(relevant & retrieved))
            first_rank = next((i for i, doc_id in enumerate(ranked, 1) if doc_id in relevant), None)
            metrics[mode]["mrr"] += 0.0 if first_rank is None else 1.0 / first_rank

        for metric in metrics[mode]:
            metrics[mode][metric] /= len(cases)
    return {mode: dict(values) for mode, values in metrics.items()}


def format_report(metrics: dict[str, dict[str, float]], case_count: int) -> str:
    headers = ["Retriever", "Recall@1", "Recall@3", "Hit@1", "Hit@3", "MRR"]
    rows = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for mode in MODES:
        values = metrics[mode]
        rows.append(" | ".join([
            mode.title(),
            f"{values['recall@1']:.1%}", f"{values['recall@3']:.1%}",
            f"{values['hit@1']:.1%}", f"{values['hit@3']:.1%}", f"{values['mrr']:.3f}",
        ]))
    return f"Retrieval benchmark ({case_count} labelled queries)\n\n" + "\n".join(rows)


def main() -> None:
    cases = load_cases()
    metrics = evaluate(Retriever(KB_DIR), cases)
    print(format_report(metrics, len(cases)))


if __name__ == "__main__":
    main()

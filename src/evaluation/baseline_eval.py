"""
Small baseline evaluator for the memory agent project.

What this script checks:
1. Intent router: does a user message go to the expected route?
2. Retriever: do top-k chunks contain an expected topic/category/question type?

This is intentionally lightweight. It does not call Gemini or any paid API, so it
can run while the API key problem is still unresolved.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.qa_retriever import QaRetriever, normalize_for_matching
from src.routing.routing import classify_intent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Test cases
# =============================================================================

@dataclass(frozen=True)
class IntentCase:
    """One expected router result."""

    name: str
    message: str
    expected_intent: str


@dataclass(frozen=True)
class RetrievalCase:
    """One retrieval expectation."""

    name: str
    query: str
    expected_any_terms: tuple[str, ...]
    expected_category: str | None = None
    expected_question_type: str | None = None


INTENT_CASES = [
    IntentCase(
        name="preference_update",
        message="Tôi thích lễ hội và ẩm thực",
        expected_intent="preference_update",
    ),
    IntentCase(
        name="memory_query",
        message="Tôi thích gì?",
        expected_intent="memory_query",
    ),
    IntentCase(
        name="recommendation_request",
        message="Gợi ý cho tôi vài chủ đề hay",
        expected_intent="recommendation_request",
    ),
    IntentCase(
        name="rag_question",
        message="Ý nghĩa văn hóa của bánh tét là gì?",
        expected_intent="rag_question",
    ),
    IntentCase(
        name="chitchat",
        message="Chào bạn",
        expected_intent="chitchat",
    ),
]


RETRIEVAL_CASES = [
    RetrievalCase(
        name="banh_tet_cultural",
        query="Ý nghĩa văn hóa của bánh tét là gì?",
        expected_any_terms=("banh tet", "bánh tét"),
        expected_category="am_thuc",
        expected_question_type="cultural",
    ),
    RetrievalCase(
        name="banh_chung_banh_tet_comparison",
        query="So sánh bánh chưng và bánh tét",
        expected_any_terms=("banh chung", "bánh chưng", "banh tet", "bánh tét"),
        expected_category="am_thuc",
        expected_question_type="comparison",
    ),
    RetrievalCase(
        name="xe_may",
        query="Xe máy có vai trò gì trong giao thông Việt Nam?",
        expected_any_terms=("xe may", "xe máy", "giao thong", "giao thông"),
        expected_category="giao_thong",
    ),
    RetrievalCase(
        name="tham_coi",
        query="Ý nghĩa văn hóa của thảm cói là gì?",
        expected_any_terms=("tham coi", "thảm cói", "thu cong", "thủ công"),
        expected_category="thu_cong_my_nghe",
    ),
]


# =============================================================================
# Evaluation helpers
# =============================================================================

def normalize_metadata_value(value: Any) -> str:
    """Normalize metadata before comparing expected terms."""

    return normalize_for_matching(str(value or ""))


def metadata_value_matches(actual_value: Any, expected_value: str) -> bool:
    """Compare metadata values after normalizing underscores and accents."""

    return normalize_metadata_value(actual_value) == normalize_for_matching(expected_value)


def join_retrieved_text(retrieved_chunks: list[Any]) -> str:
    """Build one searchable string from retrieved metadata and content."""

    text_parts: list[str] = []
    for chunk in retrieved_chunks:
        document = getattr(chunk, "document", chunk)
        metadata = getattr(document, "metadata", {}) or {}
        content = getattr(document, "page_content", "")
        text_parts.extend(str(value) for value in metadata.values())
        text_parts.append(content)

    return normalize_for_matching(" ".join(text_parts))


def evaluate_intents() -> tuple[int, int]:
    """Run all router cases and print pass/fail lines."""

    passed_count = 0
    print("\n=== INTENT ROUTER ===")

    for case in INTENT_CASES:
        decision = classify_intent(case.message)
        passed = decision.intent == case.expected_intent
        passed_count += int(passed)
        status = "PASS" if passed else "FAIL"

        print(
            f"{status} | {case.name} | expected={case.expected_intent} "
            f"| actual={decision.intent} | message={case.message}"
        )

    return passed_count, len(INTENT_CASES)


def evaluate_retrieval(retriever: QaRetriever, top_k: int, fetch_k: int) -> tuple[int, int]:
    """Run all retrieval cases and print pass/fail lines."""

    passed_count = 0
    print("\n=== RETRIEVAL ===")

    for case in RETRIEVAL_CASES:
        retrieved_chunks = retriever.retrieve(
            query=case.query,
            top_k=top_k,
            fetch_k=fetch_k,
        )
        searchable_text = join_retrieved_text(retrieved_chunks)
        top_metadata = retrieved_chunks[0].metadata if retrieved_chunks else {}

        term_ok = any(
            normalize_for_matching(term) in searchable_text
            for term in case.expected_any_terms
        )
        category_ok = True
        if case.expected_category:
            category_ok = any(
                metadata_value_matches(chunk.metadata.get("category"), case.expected_category)
                for chunk in retrieved_chunks
            )
        question_type_ok = True
        if case.expected_question_type:
            question_type_ok = any(
                metadata_value_matches(
                    chunk.metadata.get("question_type"),
                    case.expected_question_type,
                )
                for chunk in retrieved_chunks
            )

        passed = term_ok and category_ok and question_type_ok
        passed_count += int(passed)
        status = "PASS" if passed else "FAIL"

        print(
            f"{status} | {case.name} | query={case.query}\n"
            f"     top_topic={top_metadata.get('canonical_topic') or top_metadata.get('topic') or top_metadata.get('keyword')}\n"
            f"     term_ok={term_ok} category_ok={category_ok} question_type_ok={question_type_ok}"
        )

    return passed_count, len(RETRIEVAL_CASES)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the current RAG baseline.")
    parser.add_argument("--persist-dir", default=str(PROJECT_ROOT / "chroma_db"))
    parser.add_argument("--collection", default="langchain")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    """Run the baseline checks and print a compact summary."""

    args = parse_args()

    intent_passed, intent_total = evaluate_intents()

    retriever = QaRetriever(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        device=args.device,
    )
    retrieval_passed, retrieval_total = evaluate_retrieval(
        retriever=retriever,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
    )

    total_passed = intent_passed + retrieval_passed
    total_cases = intent_total + retrieval_total
    print("\n=== SUMMARY ===")
    print(f"Intent:    {intent_passed}/{intent_total}")
    print(f"Retrieval: {retrieval_passed}/{retrieval_total}")
    print(f"Total:     {total_passed}/{total_cases}")

    if total_passed != total_cases:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

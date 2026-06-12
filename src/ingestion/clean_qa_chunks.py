"""
Create clean QA chunks from the Vietnamese VQA dataset.

The dataset has image fields, but this module is NLP-first:
- A QA pair is the main retrieval unit.
- The canonical topic comes from QA/object content first.
- `keyword` is kept as an alias because it can be noisy or mismatched.
- `image_id` and `image_path` are kept only for source tracing.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:
    Document = None


# =============================================================================
# Configuration
# =============================================================================

GENERIC_IDENTIFICATION_PATTERNS = (
    "day la gi",
    "day la mon gi",
    "day la loai gi",
    "hinh anh nay la gi",
    "vat the nay la gi",
    "mon an nay la gi",
    "trang phuc nay la gi",
    "nhac cu nay la gi",
)

TOPIC_PREFIXES_TO_REMOVE = (
    "day la",
    "do la",
    "co ve la",
    "hinh anh mo ta",
    "hinh anh the hien",
    "hinh anh nay mo ta",
    "hinh anh nay the hien",
)

MAX_TOPIC_WORDS = 12


# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class ResolvedTopic:
    """Canonical topic and the source field used to resolve it."""

    text: str
    source: str


@dataclass(frozen=True)
class QaChunk:
    """Clean chunk format used before optional LangChain conversion."""

    page_content: str
    metadata: dict[str, Any]


# =============================================================================
# Generic text helpers
# =============================================================================

def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """Load the raw JSON dataset."""

    with Path(dataset_path).open("r", encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def safe_text(value: Any) -> str:
    """Convert strings, lists, and empty values into a clean display string."""

    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if item)

    return str(value).strip()


def normalize_text(value: Any) -> str:
    """
    Normalize text for matching and deduplication.

    Keep this output for internal keys only. Do not show normalized text to users
    because Vietnamese accents are intentionally removed.
    """

    text = safe_text(value).lower()
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# Topic resolution
# =============================================================================

def remove_topic_wrapper(topic_candidate: Any) -> str:
    """
    Remove common VQA wrappers while preserving Vietnamese display text.

    Example:
    "Có vẻ là món nhộng ong." -> "món nhộng ong"
    """

    display_topic = safe_text(topic_candidate).strip(" .,:;!?")
    normalized_topic = normalize_text(display_topic)

    for normalized_prefix in TOPIC_PREFIXES_TO_REMOVE:
        if not normalized_topic.startswith(normalized_prefix + " "):
            continue

        prefix_word_count = len(normalized_prefix.split())
        display_words = display_topic.split()

        return " ".join(display_words[prefix_word_count:]).strip(" .,:;!?")

    return display_topic


def is_identification_question(question_record: dict[str, Any]) -> bool:
    """
    Detect QA records whose answer probably names the item/topic.

    These answers are more trustworthy than `keyword` for this dataset.
    """

    question_type = normalize_text(question_record.get("question_type"))
    question_text = normalize_text(question_record.get("question"))

    if "identification" in question_type:
        return True

    return any(pattern in question_text for pattern in GENERIC_IDENTIFICATION_PATTERNS)


def is_usable_topic(topic_candidate: Any) -> bool:
    """Reject empty or overly long topic candidates."""

    normalized_topic = normalize_text(topic_candidate)
    if not normalized_topic:
        return False

    return len(normalized_topic.split()) <= MAX_TOPIC_WORDS


def resolve_topic_from_identification_answer(sample: dict[str, Any]) -> ResolvedTopic | None:
    """Use the answer to a generic identification question as the first choice."""

    for question_record in sample.get("questions", []) or []:
        if not is_identification_question(question_record):
            continue

        topic_text = remove_topic_wrapper(question_record.get("answer"))
        if is_usable_topic(topic_text):
            return ResolvedTopic(text=topic_text, source="identification_answer")

    return None


def resolve_topic_from_primary_object(sample: dict[str, Any]) -> ResolvedTopic | None:
    """Use `primary_cultural_objects[0]` when QA answers do not give a topic."""

    cultural_context = sample.get("cultural_context", {}) or {}
    primary_objects = cultural_context.get("primary_cultural_objects", []) or []

    for object_name in primary_objects:
        topic_text = remove_topic_wrapper(object_name)
        if is_usable_topic(topic_text):
            return ResolvedTopic(text=topic_text, source="primary_cultural_object")

    return None


def resolve_canonical_topic(sample: dict[str, Any]) -> ResolvedTopic:
    """
    Resolve the true NLP topic for a sample.

    Fallback order:
    1. Answer of an identification question.
    2. First primary cultural object.
    3. Dataset keyword.
    """

    topic_from_answer = resolve_topic_from_identification_answer(sample)
    if topic_from_answer:
        return topic_from_answer

    topic_from_object = resolve_topic_from_primary_object(sample)
    if topic_from_object:
        return topic_from_object

    keyword_topic = remove_topic_wrapper(sample.get("keyword"))
    return ResolvedTopic(text=keyword_topic, source="keyword_fallback")


# =============================================================================
# Chunk construction
# =============================================================================

def build_topic_aliases(sample: dict[str, Any], topic: str) -> list[str]:
    """
    Build alternate names for retrieval recall.

    The aliases can include noisy keywords, but aliases do not decide the
    canonical topic.
    """

    cultural_context = sample.get("cultural_context", {}) or {}
    primary_objects = cultural_context.get("primary_cultural_objects", []) or []

    raw_aliases = [
        topic,
        sample.get("keyword"),
        sample.get("category"),
        cultural_context.get("cultural_category"),
        *primary_objects,
    ]

    aliases: list[str] = []
    seen_normalized_aliases: set[str] = set()

    for raw_alias in raw_aliases:
        alias = safe_text(raw_alias)
        normalized_alias = normalize_text(alias)

        if not alias or not normalized_alias:
            continue

        if normalized_alias in seen_normalized_aliases:
            continue

        aliases.append(alias)
        seen_normalized_aliases.add(normalized_alias)

    return aliases


def build_page_content(
    sample: dict[str, Any],
    question_record: dict[str, Any],
    resolved_topic: ResolvedTopic,
    topic_aliases: list[str],
) -> str:
    """Create the text that will be embedded into Chroma later."""

    cultural_context = sample.get("cultural_context", {}) or {}
    additional_context = question_record.get("additional_context", {}) or {}

    sections = [
        "Chunk Type: qa_chunk",
        f"Canonical Topic: {resolved_topic.text}",
        f"Topic Source: {resolved_topic.source}",
        f"Aliases: {', '.join(topic_aliases)}",
        f"Category: {safe_text(sample.get('category'))}",
        f"Question Type: {safe_text(question_record.get('question_type'))}",
        f"Difficulty: {safe_text(question_record.get('difficulty'))}",
        f"Cognitive Level: {safe_text(question_record.get('cognitive_level'))}",
        "",
        "Question:",
        safe_text(question_record.get("question")),
        "",
        "Answer:",
        safe_text(question_record.get("answer")),
        "",
        "Detailed Explanation:",
        safe_text(question_record.get("detailed_explanation")),
        "",
        "Cultural Significance:",
        safe_text(question_record.get("cultural_significance")),
        "",
        "Historical Context:",
        safe_text(cultural_context.get("historical_context")),
        "",
        "Modern Relevance:",
        safe_text(cultural_context.get("modern_relevance")),
        "",
        "Origin:",
        safe_text(additional_context.get("origin")),
        "",
        "Usage:",
        safe_text(additional_context.get("usage")),
        "",
        "Symbolism:",
        safe_text(additional_context.get("symbolism")),
        "",
        "Regional Variations:",
        safe_text(additional_context.get("regional_variations")),
    ]

    return "\n".join(sections).strip()


def build_metadata(
    sample: dict[str, Any],
    question_record: dict[str, Any],
    resolved_topic: ResolvedTopic,
    topic_aliases: list[str],
) -> dict[str, Any]:
    """Create metadata used for tracing, filtering, and debugging."""

    question_text = safe_text(question_record.get("question"))
    category = safe_text(sample.get("category"))
    image_id = safe_text(sample.get("image_id"))
    question_id = safe_text(question_record.get("question_id"))

    return {
        "doc_id": f"{category}:{normalize_text(resolved_topic.text)}:{image_id}:{question_id}",
        "chunk_type": "qa_chunk",
        "category": category,
        "topic": resolved_topic.text,
        "normalized_topic": normalize_text(resolved_topic.text),
        "topic_source": resolved_topic.source,
        "aliases": " | ".join(topic_aliases),
        "keyword": safe_text(sample.get("keyword")),
        "normalized_keyword": normalize_text(sample.get("keyword")),
        "question": question_text,
        "normalized_question": normalize_text(question_text),
        "question_type": safe_text(question_record.get("question_type")),
        "difficulty": safe_text(question_record.get("difficulty")),
        "cognitive_level": safe_text(question_record.get("cognitive_level")),
        "image_id": image_id,
        "image_path": safe_text(sample.get("image_path")),
    }


def build_qa_chunk(sample: dict[str, Any], question_record: dict[str, Any]) -> QaChunk:
    """Build one clean QA chunk from one sample and one question."""

    resolved_topic = resolve_canonical_topic(sample)
    topic_aliases = build_topic_aliases(sample, resolved_topic.text)

    return QaChunk(
        page_content=build_page_content(
            sample=sample,
            question_record=question_record,
            resolved_topic=resolved_topic,
            topic_aliases=topic_aliases,
        ),
        metadata=build_metadata(
            sample=sample,
            question_record=question_record,
            resolved_topic=resolved_topic,
            topic_aliases=topic_aliases,
        ),
    )


def build_dedup_key(
    sample: dict[str, Any],
    question_record: dict[str, Any],
    resolved_topic: ResolvedTopic,
) -> tuple[str, str, str, str]:
    """
    Build a dedup key that ignores `image_id`.

    The same QA idea can appear under many images; for NLP retrieval, we keep
    the topic/question identity and only use image fields for tracing.
    """

    return (
        safe_text(sample.get("category")),
        normalize_text(resolved_topic.text),
        safe_text(question_record.get("question_type")),
        normalize_text(question_record.get("question")),
    )


def build_qa_chunks(dataset_records: list[dict[str, Any]]) -> list[QaChunk]:
    """Build deduplicated QA chunks from raw dataset records."""

    qa_chunks: list[QaChunk] = []
    seen_dedup_keys: set[tuple[str, str, str, str]] = set()

    for sample in dataset_records:
        resolved_topic = resolve_canonical_topic(sample)

        for question_record in sample.get("questions", []) or []:
            if not normalize_text(question_record.get("question")):
                continue

            dedup_key = build_dedup_key(sample, question_record, resolved_topic)
            if dedup_key in seen_dedup_keys:
                continue

            qa_chunks.append(build_qa_chunk(sample, question_record))
            seen_dedup_keys.add(dedup_key)

    return qa_chunks


# =============================================================================
# LangChain conversion
# =============================================================================

def convert_to_langchain_documents(qa_chunks: list[QaChunk]) -> list[Any]:
    """Convert clean chunks to LangChain Document objects."""

    if Document is None:
        raise ImportError(
            "langchain_core is not installed. Install LangChain before converting "
            "QA chunks to Document objects."
        )

    return [
        Document(page_content=chunk.page_content, metadata=chunk.metadata)
        for chunk in qa_chunks
    ]


def build_clean_qa_documents(dataset_path: str | Path) -> list[Any]:
    """Load dataset, build QA chunks, and convert them to LangChain Documents."""

    dataset_records = load_dataset(dataset_path)
    qa_chunks = build_qa_chunks(dataset_records)

    return convert_to_langchain_documents(qa_chunks)


# =============================================================================
# Preview and debugging helpers
# =============================================================================

def summarize_chunks(qa_chunks: list[QaChunk]) -> dict[str, Any]:
    """Return lightweight stats for notebook or terminal debugging."""

    return {
        "total_chunks": len(qa_chunks),
        "chunk_types": Counter(chunk.metadata["chunk_type"] for chunk in qa_chunks),
        "categories": Counter(chunk.metadata["category"] for chunk in qa_chunks),
        "topic_sources": Counter(chunk.metadata["topic_source"] for chunk in qa_chunks),
    }


def print_summary(qa_chunks: list[QaChunk]) -> None:
    """Print chunk counts in a compact, human-readable format."""

    summary = summarize_chunks(qa_chunks)

    print("Total chunks:", summary["total_chunks"])
    print("Chunk types:", dict(summary["chunk_types"]))
    print("Topic sources:", dict(summary["topic_sources"]))
    print("Categories:", summary["categories"].most_common())


def print_chunk_preview(qa_chunks: list[QaChunk], limit: int = 5) -> None:
    """Print sample chunks so humans can inspect topic and metadata quality."""

    for chunk_index, chunk in enumerate(qa_chunks[:limit], start=1):
        metadata = chunk.metadata

        print("=" * 80)
        print(f"CHUNK {chunk_index}")
        print(f"Topic: {metadata['topic']}")
        print(f"Topic Source: {metadata['topic_source']}")
        print(f"Keyword: {metadata['keyword']}")
        print(f"Aliases: {metadata['aliases']}")
        print(f"Question Type: {metadata['question_type']}")
        print(f"Question: {metadata['question']}")
        print()
        print(chunk.page_content[:1200])
        print()


def preview_dataset_chunks(
    dataset_path: str | Path,
    sample_size: int | None = 100,
    preview_limit: int = 5,
) -> list[QaChunk]:
    """Load a dataset slice, build chunks, print summary, and print examples."""

    dataset_records = load_dataset(dataset_path)

    if sample_size is not None:
        dataset_records = dataset_records[:sample_size]

    qa_chunks = build_qa_chunks(dataset_records)

    print_summary(qa_chunks)
    print_chunk_preview(qa_chunks, limit=preview_limit)

    return qa_chunks


# =============================================================================
# Command line preview
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for local preview runs."""

    parser = argparse.ArgumentParser(
        description="Preview clean QA chunks before building a vector index."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to vietnamese_vqa_dataset.json.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of raw dataset records to inspect. Use 0 for full dataset.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of chunks to print.",
    )

    return parser.parse_args()


def main() -> None:
    """Run a terminal preview of the chunking result."""

    args = parse_args()
    sample_size = None if args.sample_size == 0 else args.sample_size

    preview_dataset_chunks(
        dataset_path=args.dataset,
        sample_size=sample_size,
        preview_limit=args.preview,
    )


if __name__ == "__main__":
    main()

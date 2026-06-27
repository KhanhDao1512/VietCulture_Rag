"""
Create a balanced RAG benchmark candidate set from the VQA test split.

This script does not paraphrase questions yet. It prepares the clean candidate
file used for the next evaluation step:

test_data.json
-> flatten image-level records into QA-level records
-> remove overly direct VQA questions
-> sample a balanced set by category and question_type
-> print statistics and examples
-> save data/benchmark_candidates.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_DATA = PROJECT_ROOT / "data" / "test_data.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmark_candidates.json"

TARGET_PER_CATEGORY = 20
RANDOM_SEED = 42

# Per-category target. Total = 20 if all types are available.
QUESTION_TYPE_QUOTA = {
    "cultural": 5,
    "description": 4,
    "analysis": 4,
    "comparison": 4,
}

QUESTION_TYPE_PRIORITY = [
    "cultural",
    "description",
    "analysis",
    "comparison",
]

DIFFICULTY_PRIORITY = {
    "medium": 0,
    "hard": 1,
    "easy": 2,
}

DIRECT_VQA_PATTERNS = (
    r"^day la gi$",
    r"^day la .* gi$",
    r".* dang lam gi.*",
    r".* dang mac.*",
    r".* mac trang phuc gi.*",
    r"^hinh anh nay .* gi$",
    r"^anh nay .* gi$",
    r"^trong anh co gi$",
    r"^mo ta hinh anh nay$",
    r"^mo ta buc anh nay$",
    r"^hinh anh nay the hien dieu gi$",
    r"^vat the nay .* gi$",
    r"^mon .* nay .* gi$",
    r".* mon an gi.*",
    r"^trang phuc nay .* gi$",
    r".* trang phuc gi.*",
    r"^nhac cu nay .* gi$",
    r".* nhac cu gi.*",
    r".* tro choi gi.*",
    r".* nhung gi.*",
)

IMAGE_REFERENCE_TERMS = (
    "hinh anh",
    "anh nay",
    "buc anh",
    "trong anh",
    "trong hinh",
    "hinh nay",
)

DEICTIC_REFERENCE_TOKENS = {
    "nay",
    "day",
}


def safe_text(value: Any) -> str:
    """Convert a value into a stripped string."""

    if value is None:
        return ""
    return str(value).strip()


def normalize_text(value: Any) -> str:
    """Normalize Vietnamese text for filtering and deduplication."""

    text = safe_text(value).lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        character
        for character in text
        if unicodedata.category(character) != "Mn"
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load the test split JSON."""

    with path.open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")

    return data


def flatten_test_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert image-level VQA records into one row per QA pair."""

    flattened: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(records):
        cultural_context = sample.get("cultural_context", {}) or {}
        primary_objects = cultural_context.get("primary_cultural_objects", []) or []

        for question_record in sample.get("questions", []) or []:
            question = safe_text(question_record.get("question"))
            answer = safe_text(question_record.get("answer"))
            explanation = safe_text(question_record.get("detailed_explanation"))

            if not question or not answer:
                continue

            question_type = safe_text(question_record.get("question_type"))
            difficulty = safe_text(question_record.get("difficulty"))
            cognitive_level = safe_text(question_record.get("cognitive_level"))
            question_id = safe_text(question_record.get("question_id"))
            category = safe_text(sample.get("category"))
            keyword = safe_text(sample.get("keyword"))
            image_id = safe_text(sample.get("image_id"))

            chunk_id = build_ground_truth_chunk_id(
                category=category,
                keyword=keyword,
                image_id=image_id,
                question=question,
                question_id=question_id,
            )

            flattened.append(
                {
                    "source_sample_index": sample_index,
                    "image_id": image_id,
                    "image_path": safe_text(sample.get("image_path")),
                    "category": category,
                    "keyword": keyword,
                    "primary_cultural_objects": primary_objects,
                    "original_question": question,
                    "user_query": "",
                    "answer": answer,
                    "detailed_explanation": explanation,
                    "cultural_significance": safe_text(
                        question_record.get("cultural_significance")
                    ),
                    "question_type": question_type,
                    "difficulty": difficulty,
                    "cognitive_level": cognitive_level,
                    "question_id": question_id,
                    "ground_truth_chunk_id": chunk_id,
                }
            )

    return flattened


def build_ground_truth_chunk_id(
    category: str,
    keyword: str,
    image_id: str,
    question: str,
    question_id: str,
) -> str:
    """
    Build a stable chunk id compatible with the current Kaggle chunking logic.

    If you add `chunk_id` to Kaggle Chroma metadata later, use the same formula.
    """

    normalized_keyword = normalize_text(keyword)
    normalized_question = normalize_text(question)[:80]
    return (
        f"{category}:{normalized_keyword}:{image_id}:"
        f"{question_id}:{normalized_question}"
    )


def is_direct_vqa_question(question: str) -> bool:
    """Detect questions that mainly ask the model to identify the image."""

    normalized_question = normalize_text(question)
    return any(
        re.match(pattern, normalized_question)
        for pattern in DIRECT_VQA_PATTERNS
    )


def has_image_reference(question: str) -> bool:
    """Detect questions that still depend on explicit image wording."""

    normalized_question = normalize_text(question)
    return any(term in normalized_question for term in IMAGE_REFERENCE_TERMS)


def has_deictic_reference(question: str) -> bool:
    """Detect words like 'này'/'đây' that make the query image-dependent."""

    normalized_tokens = set(normalize_text(question).split())
    return bool(normalized_tokens & DEICTIC_REFERENCE_TOKENS)


def is_candidate_for_rag(row: dict[str, Any]) -> bool:
    """Return True when a QA pair is useful for a RAG-style benchmark."""

    question_type = safe_text(row.get("question_type"))
    if question_type not in QUESTION_TYPE_QUOTA:
        return False

    question = safe_text(row.get("original_question"))
    if is_direct_vqa_question(question):
        return False

    if has_image_reference(question):
        return False

    if has_deictic_reference(question):
        return False

    if not safe_text(row.get("detailed_explanation")):
        return False

    return True


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by category, keyword, question type, and normalized question."""

    deduped_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for row in rows:
        key = (
            safe_text(row.get("category")),
            normalize_text(row.get("keyword")),
            safe_text(row.get("question_type")),
            normalize_text(row.get("original_question")),
        )
        if key in seen_keys:
            continue

        deduped_rows.append(row)
        seen_keys.add(key)

    return deduped_rows


def sort_candidates(rows: list[dict[str, Any]], random_generator: random.Random) -> list[dict[str, Any]]:
    """Shuffle lightly, then prefer medium/hard and richer cognitive levels."""

    shuffled_rows = rows[:]
    random_generator.shuffle(shuffled_rows)

    return sorted(
        shuffled_rows,
        key=lambda row: (
            DIFFICULTY_PRIORITY.get(safe_text(row.get("difficulty")), 99),
            QUESTION_TYPE_PRIORITY.index(safe_text(row.get("question_type")))
            if safe_text(row.get("question_type")) in QUESTION_TYPE_PRIORITY
            else 99,
            normalize_text(row.get("keyword")),
        ),
    )


def select_keyword_diverse_rows(
    rows: list[dict[str, Any]],
    quota: int,
    random_generator: random.Random,
) -> list[dict[str, Any]]:
    """Select rows with a round-robin pass over keywords to reduce repetition."""

    if quota <= 0:
        return []

    by_keyword: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sort_candidates(rows, random_generator=random_generator):
        keyword_key = normalize_text(row.get("keyword")) or normalize_text(
            row.get("original_question")
        )
        by_keyword[keyword_key].append(row)

    selected_rows: list[dict[str, Any]] = []
    keyword_keys = sorted(by_keyword)

    while len(selected_rows) < quota and keyword_keys:
        made_progress = False

        for keyword_key in keyword_keys[:]:
            keyword_rows = by_keyword[keyword_key]
            if not keyword_rows:
                keyword_keys.remove(keyword_key)
                continue

            selected_rows.append(keyword_rows.pop(0))
            made_progress = True

            if len(selected_rows) >= quota:
                break

        if not made_progress:
            break

    return selected_rows


def sample_benchmark_candidates(
    rows: list[dict[str, Any]],
    target_per_category: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Sample balanced benchmark candidates by category and question_type."""

    rng = random.Random(random_seed)
    by_category_and_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (safe_text(row.get("category")), safe_text(row.get("question_type")))
        by_category_and_type[key].append(row)

    selected_rows: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    categories = sorted({safe_text(row.get("category")) for row in rows})

    for category in categories:
        category_selected: list[dict[str, Any]] = []

        for question_type, quota in QUESTION_TYPE_QUOTA.items():
            candidates = select_keyword_diverse_rows(
                by_category_and_type.get((category, question_type), []),
                quota=quota,
                random_generator=rng,
            )
            category_selected.extend(candidates)

        if len(category_selected) < target_per_category:
            category_pool = select_keyword_diverse_rows(
                [
                    row
                    for row in rows
                    if safe_text(row.get("category")) == category
                    and row["ground_truth_chunk_id"] not in selected_keys
                    and row not in category_selected
                ],
                quota=target_per_category - len(category_selected),
                random_generator=rng,
            )
            category_selected.extend(category_pool)

        for row in category_selected[:target_per_category]:
            if row["ground_truth_chunk_id"] in selected_keys:
                continue
            selected_rows.append(row)
            selected_keys.add(row["ground_truth_chunk_id"])

    for index, row in enumerate(selected_rows, start=1):
        row["benchmark_id"] = f"bench_{index:04d}"

    return selected_rows


def counter_for(rows: list[dict[str, Any]], field_name: str) -> Counter:
    """Count a field in selected rows."""

    return Counter(safe_text(row.get(field_name)) for row in rows)


def print_counter(title: str, counter: Counter) -> None:
    """Print a compact counter table."""

    print(f"\n=== {title} ===")
    for key, value in counter.most_common():
        print(f"{key or '<empty>':30s} {value}")


def print_cross_table(
    rows: list[dict[str, Any]],
    row_field: str,
    col_field: str,
) -> None:
    """Print a simple row x column table for quick inspection."""

    row_values = sorted({safe_text(row.get(row_field)) for row in rows})
    col_values = sorted({safe_text(row.get(col_field)) for row in rows})

    print(f"\n=== {row_field} x {col_field} ===")
    print(f"{row_field:24s}", end="")
    for col_value in col_values:
        print(f"{col_value[:12]:>12s}", end="")
    print()

    for row_value in row_values:
        print(f"{row_value[:24]:24s}", end="")
        for col_value in col_values:
            count = sum(
                1
                for row in rows
                if safe_text(row.get(row_field)) == row_value
                and safe_text(row.get(col_field)) == col_value
            )
            print(f"{count:12d}", end="")
        print()


def print_samples(rows: list[dict[str, Any]], limit: int) -> None:
    """Print a few selected examples."""

    print(f"\n=== SAMPLE CANDIDATES ({limit}) ===")
    for row in rows[:limit]:
        print("-" * 80)
        print("benchmark_id:", row["benchmark_id"])
        print("category:", row["category"])
        print("keyword:", row["keyword"])
        print("question_type:", row["question_type"])
        print("difficulty:", row["difficulty"])
        print("cognitive_level:", row["cognitive_level"])
        print("question:", row["original_question"])
        print("answer:", row["answer"])


def save_json(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Save benchmark candidates as UTF-8 JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Create balanced RAG benchmark candidates from test_data.json."
    )
    parser.add_argument("--input", default=str(DEFAULT_TEST_DATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-per-category", type=int, default=TARGET_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--sample-preview", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Run candidate creation and print summary statistics."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_json(input_path)
    flattened_rows = flatten_test_records(records)
    deduped_rows = deduplicate_rows(flattened_rows)
    filtered_rows = [row for row in deduped_rows if is_candidate_for_rag(row)]
    selected_rows = sample_benchmark_candidates(
        rows=filtered_rows,
        target_per_category=args.target_per_category,
        random_seed=args.seed,
    )

    print("Input:", input_path)
    print("Output:", output_path)
    print("Raw image-level samples:", len(records))
    print("Flattened QA pairs:", len(flattened_rows))
    print("Deduplicated QA pairs:", len(deduped_rows))
    print("RAG-suitable candidates:", len(filtered_rows))
    print("Selected benchmark candidates:", len(selected_rows))

    print_counter("Selected by category", counter_for(selected_rows, "category"))
    print_counter("Selected by question_type", counter_for(selected_rows, "question_type"))
    print_counter("Selected by difficulty", counter_for(selected_rows, "difficulty"))
    print_counter("Selected by cognitive_level", counter_for(selected_rows, "cognitive_level"))
    print_cross_table(selected_rows, "category", "question_type")
    print_cross_table(selected_rows, "question_type", "difficulty")
    print_samples(selected_rows, limit=args.sample_preview)

    save_json(selected_rows, output_path)
    print(f"\nSaved {len(selected_rows)} candidates to {output_path}")


if __name__ == "__main__":
    main()

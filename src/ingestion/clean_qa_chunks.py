"""
Tạo QA chunks sạch từ Vietnamese VQA dataset.

Mục đích file:
Chuyển raw VQA records thành text chunks hữu ích cho embedding và RAG. File này
chưa build vector; nó chỉ chuẩn bị documents sạch.

Flow chunking:
load_dataset()
-> build_qa_chunks()
-> resolve_canonical_topic()
-> build_page_content()
-> build_metadata()
-> convert_to_langchain_documents()

Quy ước dữ liệu:
- Một QA pair là đơn vị retrieval chính.
- Canonical topic ưu tiên lấy từ QA/object content trước.
- `keyword` giữ lại như alias vì có thể nhiễu hoặc lệch topic thật.
- `image_id` và `image_path` chỉ dùng để trace nguồn.
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
    """
    Topic chuẩn của một sample và nguồn dùng để suy ra topic đó.

    Ví dụ output:
    ResolvedTopic(text="bánh tét", source="identification_answer")

    Cách tự viết lại:
    Tạo dataclass nhỏ gồm text topic và source để khi debug biết topic đến từ
    answer, primary object hay keyword fallback.
    """

    text: str
    source: str


@dataclass(frozen=True)
class QaChunk:
    """
    Format chunk sạch trước khi convert sang LangChain Document.

    Biến:
    - page_content: text chính sẽ được embed.
    - metadata: thông tin dùng để filter/debug/rerank.

    Ví dụ output:
    QaChunk(page_content="Question: ...", metadata={"category": "am_thuc"})

    Cách tự viết lại:
    Tách rõ nội dung embed và metadata trace nguồn. Sau đó có thể convert thành
    Document(page_content=..., metadata=...).
    """

    page_content: str
    metadata: dict[str, Any]


# =============================================================================
# Generic text helpers
# =============================================================================

def load_dataset(dataset_path: str | Path) -> list[dict[str, Any]]:
    """
    Đọc raw JSON dataset.

    Ví dụ output:
    [{"image_id": "...", "category": "am_thuc", "questions": [...]}]

    Cách tự viết lại:
    Mở file bằng UTF-8 và json.load() thành list dict.
    """

    with Path(dataset_path).open("r", encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def safe_text(value: Any) -> str:
    """
    Chuyển giá trị bất kỳ thành string sạch để đưa vào chunk.

    Ví dụ output:
    safe_text(["a", "b"]) -> "a, b"
    safe_text(None) -> ""

    Cách tự viết lại:
    Nếu None thì trả "", nếu list thì join phần tử không rỗng, còn lại cast str
    và strip.
    """

    if value is None:
        return ""

    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if item)

    return str(value).strip()


def normalize_text(value: Any) -> str:
    """
    Normalize text để matching và dedup.

    Ví dụ output:
    normalize_text("Bánh Tét!") -> "banh tet"

    Cách tự viết lại:
    Chuyển lowercase, bỏ dấu tiếng Việt, bỏ ký tự đặc biệt, collapse khoảng
    trắng. Chỉ dùng output này cho key nội bộ, không hiển thị cho user.
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
    Bỏ các wrapper thường gặp quanh topic.

    Ví dụ output:
    "Có vẻ là món nhộng ong." -> "món nhộng ong"

    Cách tự viết lại:
    Normalize candidate để nhận diện prefix như "day la", nhưng khi cắt thì cắt
    trên display text gốc để giữ dấu tiếng Việt.
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
    Kiểm tra QA record có phải câu hỏi nhận diện object/topic không.

    Ví dụ output:
    question_type="identification" -> True

    Cách tự viết lại:
    Check question_type trước, sau đó fallback bằng pattern câu hỏi như "đây là
    gì", "món ăn này là gì". Answer của dạng này thường là topic đáng tin hơn keyword.
    """

    question_type = normalize_text(question_record.get("question_type"))
    question_text = normalize_text(question_record.get("question"))

    if "identification" in question_type:
        return True

    return any(pattern in question_text for pattern in GENERIC_IDENTIFICATION_PATTERNS)


def is_usable_topic(topic_candidate: Any) -> bool:
    """
    Loại topic rỗng hoặc quá dài.

    Ví dụ output:
    is_usable_topic("bánh tét") -> True

    Cách tự viết lại:
    Normalize topic, nếu rỗng thì False, nếu số từ vượt MAX_TOPIC_WORDS thì False.
    """

    normalized_topic = normalize_text(topic_candidate)
    if not normalized_topic:
        return False

    return len(normalized_topic.split()) <= MAX_TOPIC_WORDS


def resolve_topic_from_identification_answer(sample: dict[str, Any]) -> ResolvedTopic | None:
    """
    Lấy topic từ answer của câu hỏi identification nếu có.

    Ví dụ output:
    sample có Q "Đây là món gì?" A "Bánh tét"
    -> ResolvedTopic("Bánh tét", "identification_answer")

    Cách tự viết lại:
    Duyệt questions trong sample, tìm câu identification, làm sạch answer bằng
    remove_topic_wrapper(), rồi trả ResolvedTopic nếu topic usable.
    """

    for question_record in sample.get("questions", []) or []:
        if not is_identification_question(question_record):
            continue

        topic_text = remove_topic_wrapper(question_record.get("answer"))
        if is_usable_topic(topic_text):
            return ResolvedTopic(text=topic_text, source="identification_answer")

    return None


def resolve_topic_from_primary_object(sample: dict[str, Any]) -> ResolvedTopic | None:
    """
    Lấy topic từ `primary_cultural_objects` khi answer không cho topic rõ.

    Ví dụ output:
    primary_cultural_objects=["áo dài"] -> ResolvedTopic("áo dài", "primary_cultural_object")

    Cách tự viết lại:
    Lấy list object từ cultural_context, chọn object đầu tiên usable làm topic.
    """

    cultural_context = sample.get("cultural_context", {}) or {}
    primary_objects = cultural_context.get("primary_cultural_objects", []) or []

    for object_name in primary_objects:
        topic_text = remove_topic_wrapper(object_name)
        if is_usable_topic(topic_text):
            return ResolvedTopic(text=topic_text, source="primary_cultural_object")

    return None


def resolve_canonical_topic(sample: dict[str, Any]) -> ResolvedTopic:
    """
    Chọn canonical topic tốt nhất cho một sample.

    Thứ tự fallback:
    1. Answer of an identification question.
    2. First primary cultural object.
    3. Dataset keyword.

    Ví dụ output:
    ResolvedTopic(text="bánh chưng", source="identification_answer")

    Cách tự viết lại:
    Thử từng nguồn theo độ tin cậy giảm dần. Nguồn nào trả topic usable trước
    thì dùng, cuối cùng mới fallback sang keyword.
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
    Tạo alias topic để tăng recall retrieval.

    Ví dụ output:
    ["Bánh tét", "banh tet", "am_thuc"]

    Cách tự viết lại:
    Gom canonical topic, keyword, category, cultural_category, primary objects.
    Deduplicate bằng normalized form để tránh alias trùng.
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
    """
    Tạo text chính sẽ được embed vào Chroma.

    Biến đầu vào:
    - sample: raw record.
    - question_record: một QA trong sample.
    - resolved_topic/topic_aliases: topic chuẩn và alias.

    Ví dụ output rút gọn:
    Chunk Type: qa_chunk
    Canonical Topic: Bánh tét
    Question:
    Ý nghĩa văn hóa của bánh tét là gì?
    Answer:
    ...

    Cách tự viết lại:
    Ghép các section ổn định như Question, Answer, Cultural Significance. Section
    label rõ giúp LLM và hàm extract_content_section đọc dễ hơn.
    """

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
    """
    Tạo metadata cho chunk để trace/filter/rerank.

    Ví dụ output:
    {"category": "am_thuc", "topic": "Bánh tét", "question_type": "cultural"}

    Cách tự viết lại:
    Lưu các field ngắn, ổn định: doc_id, category, topic, normalized_topic,
    question_type, image_id. Không nhét text quá dài vào metadata.
    """

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
    """
    Build một QaChunk từ một sample và một question.

    Ví dụ output:
    QaChunk(page_content="Chunk Type: qa_chunk...", metadata={...})

    Cách tự viết lại:
    Resolve topic, build aliases, build page_content, build metadata, rồi trả
    QaChunk.
    """

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
    Tạo key dedup, cố ý bỏ qua `image_id`.

    Ví dụ output:
    ("am_thuc", "banh tet", "cultural", "y nghia van hoa cua banh tet la gi")

    Cách tự viết lại:
    Key nên dựa trên category + normalized topic + question_type + normalized
    question. Không dùng image_id vì cùng một ý QA có thể lặp ở nhiều ảnh.
    """

    return (
        safe_text(sample.get("category")),
        normalize_text(resolved_topic.text),
        safe_text(question_record.get("question_type")),
        normalize_text(question_record.get("question")),
    )


def build_qa_chunks(dataset_records: list[dict[str, Any]]) -> list[QaChunk]:
    """
    Build toàn bộ QA chunks đã dedup từ raw records.

    Ví dụ output:
    [QaChunk(...), QaChunk(...)]

    Cách tự viết lại:
    Duyệt từng sample, resolve topic một lần, duyệt questions, bỏ question rỗng,
    tạo dedup key, rồi append chunk nếu key chưa gặp.
    """

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
    """
    Convert QaChunk sang LangChain Document.

    Ví dụ output:
    Document(page_content=chunk.page_content, metadata=chunk.metadata)

    Cách tự viết lại:
    Với mỗi QaChunk, truyền page_content và metadata vào Document. Hàm này là
    cầu nối giữa format nội bộ và Chroma/LangChain.
    """

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
    """
    Hàm tiện lợi: raw dataset -> LangChain Documents.

    Ví dụ output:
    build_clean_qa_documents("vietnamese_vqa_dataset.json") -> [Document(...)]

    Cách tự viết lại:
    Gọi load_dataset(), build_qa_chunks(), rồi convert_to_langchain_documents().
    """

    dataset_records = load_dataset(dataset_path)
    qa_chunks = build_qa_chunks(dataset_records)

    return convert_to_langchain_documents(qa_chunks)


# =============================================================================
# Preview and debugging helpers
# =============================================================================

def summarize_chunks(qa_chunks: list[QaChunk]) -> dict[str, Any]:
    """
    Tạo thống kê nhanh để debug chất lượng chunk.

    Ví dụ output:
    {"total_chunks": 1200, "categories": Counter({"am_thuc": 100, ...})}

    Cách tự viết lại:
    Dùng Counter đếm chunk_type, category, topic_source để biết dataset sau
    chunking có bị lệch category hay nguồn topic không.
    """

    return {
        "total_chunks": len(qa_chunks),
        "chunk_types": Counter(chunk.metadata["chunk_type"] for chunk in qa_chunks),
        "categories": Counter(chunk.metadata["category"] for chunk in qa_chunks),
        "topic_sources": Counter(chunk.metadata["topic_source"] for chunk in qa_chunks),
    }


def print_summary(qa_chunks: list[QaChunk]) -> None:
    """
    In summary chunk ra terminal.

    Ví dụ output:
    Total chunks: 1200
    Chunk types: {"qa_chunk": 1200}

    Cách tự viết lại:
    Gọi summarize_chunks(), rồi print các trường quan trọng ở dạng dễ đọc.
    """

    summary = summarize_chunks(qa_chunks)

    print("Total chunks:", summary["total_chunks"])
    print("Chunk types:", dict(summary["chunk_types"]))
    print("Topic sources:", dict(summary["topic_sources"]))
    print("Categories:", summary["categories"].most_common())


def print_chunk_preview(qa_chunks: list[QaChunk], limit: int = 5) -> None:
    """
    In một vài chunk mẫu để con người kiểm tra.

    Ví dụ output:
    CHUNK 1
    Topic: Bánh tét
    Question: Ý nghĩa văn hóa...

    Cách tự viết lại:
    Duyệt vài chunk đầu, in topic/source/keyword/question và một đoạn page_content.
    """

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
    """
    Preview chunking trên một phần dataset.

    Ví dụ output:
    In summary + 5 chunk mẫu, đồng thời trả list QaChunk.

    Cách tự viết lại:
    Load dataset, lấy sample_size records nếu cần, build chunks, print summary
    và preview để kiểm tra trước khi build Chroma thật.
    """

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
    """
    Parse CLI args cho việc preview chunking.

    Ví dụ command:
    python src/ingestion/clean_qa_chunks.py --dataset vietnamese_vqa_dataset.json

    Cách tự viết lại:
    Dùng argparse, expose dataset path, sample size và số chunk preview.
    """

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
    """
    Entry point CLI để preview kết quả chunking.

    Ví dụ output:
    Total chunks, category counts, và vài chunk mẫu.

    Cách tự viết lại:
    Parse args, đổi sample_size=0 thành None, rồi gọi preview_dataset_chunks().
    """

    args = parse_args()
    sample_size = None if args.sample_size == 0 else args.sample_size

    preview_dataset_chunks(
        dataset_path=args.dataset,
        sample_size=sample_size,
        preview_limit=args.preview,
    )


if __name__ == "__main__":
    main()

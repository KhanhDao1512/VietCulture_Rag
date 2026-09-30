"""Build QA chunks theo category và topic, loại câu hỏi phụ thuộc ảnh.

Mục đích file:
Tạo một thử nghiệm ingestion độc lập để so sánh với index Chroma hiện tại.

Luồng xử lý:
load JSON VQA
-> chuẩn hóa category/topic từ image_path và alias
-> lọc QA không tự nêu chủ đề hoặc còn tham chiếu ảnh
-> chọn tối đa N QA đa dạng cho mỗi category/topic
-> ghi JSONL, thống kê, mẫu preview và Chroma index riêng

Ghi chú quan trọng:
- Một QA đủ điều kiện tạo thành một chunk; tối đa 5 QA cho một cặp category/topic.
- Answer và detailed explanation là nội dung evidence chính.
- Bộ lọc là heuristic; cần duyệt preview trước khi dùng làm sản phẩm cuối.
- Không ghi đè index runtime hiện có.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.chroma_store import DEFAULT_EMBED_MODEL, E5Embeddings


CATEGORY_LABELS = {
    "am_thuc": "Ẩm thực",
    "doi_song_hang_ngay": "Đời sống hằng ngày",
    "giao_thong": "Giao thông",
    "kien_truc": "Kiến trúc",
    "le_hoi": "Lễ hội",
    "nhac_cu": "Nhạc cụ",
    "phong_canh": "Phong cảnh",
    "the_thao_truyen_thong": "Thể thao truyền thống",
    "thu_cong_my_nghe": "Thủ công mỹ nghệ",
    "trang_phuc": "Trang phục",
    "tro_choi_dan_gian": "Trò chơi dân gian",
    "van_hoa_dan_gian": "Văn hóa dân gian",
}

QUESTION_TYPE_PRIORITY = {
    "cultural": 0,
    "analysis": 1,
    "historical": 2,
    "comparison": 3,
    "description": 4,
    "identification": 5,
}

# Chỉ giữ dạng câu có khả năng tạo kiến thức văn hóa độc lập với ảnh.
INDEPENDENT_QUESTION_TYPES = {"cultural", "analysis", "historical", "comparison"}

IMAGE_REFERENCES = (
    "hinh anh",
    "hinh anh cua",
    "hinh anh ve",
    "hinh anh mo ta",
    "hinh anh nay",
    "hinh anh trong",
    "trong hinh anh",
    "trong hinh",
    "trong buc hinh",
    "trong buc anh",
    "buc anh nay",
    "hinh nay",
    "hinh la gi",
    "hinh the hien",
    "hinh cho thay",
    "anh nay",
    "vat the nay",
    "mon an nay",
    "trang phuc nay",
    "nhac cu nay",
    "nhung nguoi trong anh",
    "nguoi trong hinh",
    "cac doi tuong nay",
    "nhung vat dung nay",
    "mon nay",
    "cong trinh nay",
    "su vat nay",
    "doi tuong nay",
    "cac vat nay",
    "nhung vat nay",
    "vat pham nay",
    "cac vat pham nay",
    "nhung vat pham nay",
    "dieu nay",
    "nhung chiec",
    "vat dung nay",
    "cac vat the",
    "qua hinh anh",
    "trong khung hinh",
    "duoc the hien",
    "trong buc tranh",
    "buc tranh nay",
    "hinh anh nay la gi",
)


@dataclass(frozen=True)
class CandidateQa:
    """QA độc lập đã gắn category/topic và provenance nguồn."""

    category_id: str
    category_label: str
    topic_id: str
    topic: str
    topic_aliases: list[str]
    primary_objects: list[str]
    question_type: str
    question: str
    answer: str
    detailed_explanation: str
    cultural_significance: str
    historical_context: str
    origin: str
    usage: str
    regional_variations: str
    keyword: str
    image_id: str
    image_path: str


def normalize_text(value: Any) -> str:
    """Chuẩn hóa dấu và khoảng trắng để matching/dedup."""

    text = unicodedata.normalize("NFD", str(value or "").casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def get_topic_slug(record: dict[str, Any]) -> str:
    """Lấy topic group từ folder cuối của image_path."""

    image_path = str(record.get("image_path") or "").replace("\\", "/")
    parts = PurePosixPath(image_path).parts
    if len(parts) >= 2 and parts[-2]:
        return parts[-2].replace("_", " ").strip()
    return str(record.get("keyword") or "chua ro topic").replace("_", " ").strip()


def collect_topic_aliases(record: dict[str, Any], topic_slug: str) -> list[str]:
    """Tập hợp tên folder, keyword và cultural objects làm tên gọi đồng nghĩa."""

    context = record.get("cultural_context") or {}
    aliases: list[str] = [topic_slug, str(record.get("keyword") or "")]
    # Chỉ dùng đối tượng văn hóa chính làm alias; visual main_objects thường
    # chứa nguyên liệu/đạo cụ không đại diện cho topic của cả folder.
    aliases.extend(str(value) for value in context.get("primary_cultural_objects", []) or [])

    clean_aliases: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        alias = alias.replace("_", " ").strip()
        normalized = normalize_text(alias)
        if len(normalized) < 3 or normalized in seen:
            continue
        seen.add(normalized)
        clean_aliases.append(alias)
    return clean_aliases


def choose_display_topic(topic_slug: str, aliases: list[str]) -> str:
    """Chuẩn hóa tên folder làm topic, không để alias nhiễu ghi đè."""

    normalized_slug = unicodedata.normalize("NFC", topic_slug.replace("_", " ").strip())
    alias_keys = {normalize_text(alias) for alias in aliases}
    if normalize_text(normalized_slug) in alias_keys:
        return normalized_slug
    return normalized_slug


def has_image_reference(question: str) -> bool:
    """Bỏ câu hỏi dùng chỉ thị phụ thuộc vào ảnh hoặc vật thể không định danh."""

    normalized = normalize_text(question)
    if any(reference in normalized for reference in IMAGE_REFERENCES):
        return True
    # Chặn các chỉ thị quy chiếu bằng đại từ, kể cả khi không nhắc trực tiếp ảnh.
    return any(
        phrase in normalized
        for phrase in (
            "trong buoi bieu dien nay",
            "trong buoi trinh dien nay",
            "trong su kien nay",
            "trong buc tranh nay",
            "nhu the nay",
        )
    )


def has_weak_answer(question: dict[str, Any]) -> bool:
    """Loại đáp án từ chối, quá ngắn hoặc không có phần giải thích đủ dùng."""

    answer = normalize_text(question.get("answer"))
    explanation = normalize_text(question.get("detailed_explanation"))
    significance = normalize_text(question.get("cultural_significance"))
    weak_prefixes = (
        "can them thong tin",
        "khong co du thong tin",
        "khong the khang dinh",
        "kho xac dinh chinh xac",
    )
    if not answer or len(answer) < 18 or answer.startswith(weak_prefixes):
        return True
    return max(len(explanation), len(significance)) < 100


def question_names_topic(question: str, aliases: list[str]) -> bool:
    """Chỉ giữ câu hỏi tự nêu topic hoặc một alias nhận diện được."""

    normalized_question = f" {normalize_text(question)} "
    for alias in aliases:
        normalized_alias = normalize_text(alias)
        if len(normalized_alias) < 3:
            continue
        if f" {normalized_alias} " in normalized_question:
            return True
    return False


def build_candidate(record: dict[str, Any], question: dict[str, Any]) -> CandidateQa:
    """Chuyển một QA raw thành candidate sạch, có metadata truy nguyên."""

    topic_slug = get_topic_slug(record)
    aliases = collect_topic_aliases(record, topic_slug)
    category_id = str(record.get("category") or "unknown")
    context = record.get("cultural_context") or {}
    additional = question.get("additional_context") or {}
    return CandidateQa(
        category_id=category_id,
        category_label=CATEGORY_LABELS.get(category_id, category_id.replace("_", " ").title()),
        topic_id=normalize_text(topic_slug).replace(" ", "_"),
        topic=choose_display_topic(topic_slug, aliases),
        topic_aliases=aliases,
        primary_objects=[
            unicodedata.normalize("NFC", str(value)).replace("_", " ").strip()
            for value in context.get("primary_cultural_objects", []) or []
            if str(value).strip()
        ],
        question_type=str(question.get("question_type") or "unknown").strip().lower(),
        question=str(question.get("question") or "").strip(),
        answer=str(question.get("answer") or "").strip(),
        detailed_explanation=str(question.get("detailed_explanation") or "").strip(),
        cultural_significance=str(question.get("cultural_significance") or "").strip(),
        historical_context=str(context.get("historical_context") or "").strip(),
        origin=str(additional.get("origin") or "").strip(),
        usage=str(additional.get("usage") or "").strip(),
        regional_variations=str(additional.get("regional_variations") or "").strip(),
        keyword=str(record.get("keyword") or "").strip(),
        image_id=str(record.get("image_id") or "").strip(),
        image_path=str(record.get("image_path") or "").strip(),
    )


def candidate_content(candidate: CandidateQa) -> str:
    """Tạo một QA chunk; answer/explanation làm evidence chính."""

    sections = [
        f"Category: {candidate.category_label}",
        f"Topic: {candidate.topic}",
        f"Aliases: {', '.join(candidate.topic_aliases)}",
        f"Question type: {candidate.question_type}",
        f"Question: {candidate.question}",
        f"Answer: {candidate.answer}",
    ]
    optional_sections = [
        ("Detailed explanation", candidate.detailed_explanation),
        ("Cultural significance", candidate.cultural_significance),
    ]
    sections.extend(f"{label}: {value}" for label, value in optional_sections if value)
    return "\n\n".join(sections)


def scan_dataset(dataset_path: Path) -> tuple[list[CandidateQa], Counter[str], Counter[str]]:
    """Đọc dataset, thống kê lý do loại và gom QA hợp lệ theo category/topic."""

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        records = json.load(dataset_file)

    candidates_by_topic: dict[tuple[str, str], list[CandidateQa]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    raw_question_types: Counter[str] = Counter()

    for record in records:
        topic_slug = get_topic_slug(record)
        aliases = collect_topic_aliases(record, topic_slug)
        category_id = str(record.get("category") or "unknown")
        topic_id = normalize_text(topic_slug).replace(" ", "_")
        for question in record.get("questions", []) or []:
            question_text = str(question.get("question") or "").strip()
            answer = str(question.get("answer") or "").strip()
            question_type = str(question.get("question_type") or "unknown").strip().lower()
            raw_question_types[question_type] += 1

            if not question_text or not answer:
                rejection_counts["missing_question_or_answer"] += 1
                continue
            if question_type not in INDEPENDENT_QUESTION_TYPES:
                rejection_counts["visual_or_low_priority_question_type"] += 1
                continue
            if has_image_reference(question_text):
                rejection_counts["explicit_image_reference"] += 1
                continue
            if has_weak_answer(question):
                rejection_counts["weak_answer_or_explanation"] += 1
                continue
            if not question_names_topic(question_text, aliases):
                rejection_counts["question_does_not_name_topic"] += 1
                continue
            if not (question.get("detailed_explanation") or question.get("cultural_significance")):
                rejection_counts["missing_supporting_explanation"] += 1
                continue

            candidate = build_candidate(record, question)
            candidates_by_topic[(category_id, topic_id)].append(candidate)

    flattened: list[CandidateQa] = []
    for group in candidates_by_topic.values():
        # Chọn tên hiển thị thống nhất cho topic qua nhiều ảnh.
        alias_counts: Counter[str] = Counter()
        alias_display: dict[str, str] = {}
        primary_counts: Counter[str] = Counter()
        primary_display: dict[str, str] = {}
        topic_slug = get_topic_slug(
            {"image_path": group[0].image_path, "keyword": group[0].keyword}
        )
        slug_terms = set(normalize_text(topic_slug).split())
        for candidate in group:
            for primary in candidate.primary_objects:
                primary = unicodedata.normalize("NFC", primary)
                primary_key = normalize_text(primary)
                if primary_key:
                    primary_counts[primary_key] += 1
                    primary_display.setdefault(primary_key, primary)
            for alias in candidate.topic_aliases:
                alias_key = normalize_text(alias)
                if alias_key:
                    alias_counts[alias_key] += 1
                    alias_display.setdefault(alias_key, alias)
        ranked_primary = sorted(
            primary_counts,
            key=lambda key: (
                len(slug_terms & set(key.split())),
                primary_counts[key],
                int(any(ord(char) > 127 for char in primary_display[key])),
                -len(key),
            ),
            reverse=True,
        )
        ranked_aliases = sorted(
            alias_counts,
            key=lambda alias_key: (
                len(slug_terms & set(alias_key.split()))
                + 3 * int(any(ord(char) > 127 for char in alias_display[alias_key])),
                alias_counts[alias_key],
                -len(alias_key),
            ),
            reverse=True,
        )
        # Folder xác định topic group; metadata không được phép đổi nhãn group.
        canonical_topic = unicodedata.normalize("NFC", topic_slug.replace("_", " ").strip())
        flattened.extend(replace(candidate, topic=canonical_topic) for candidate in group)

    return flattened, rejection_counts, raw_question_types


def deduplicate_and_select(
    candidates: list[CandidateQa], max_qa_per_topic: int
) -> tuple[list[CandidateQa], dict[str, int]]:
    """Bỏ QA trùng, chọn đa dạng question type, giới hạn theo topic."""

    grouped: dict[tuple[str, str], list[CandidateQa]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.category_id, candidate.topic_id)].append(candidate)

    selected: list[CandidateQa] = []
    stats = {"eligible_before_dedup": 0, "duplicates_removed": 0}
    for group in grouped.values():
        stats["eligible_before_dedup"] += len(group)
        unique_candidates: dict[tuple[str, str], CandidateQa] = {}
        for candidate in group:
            key = (candidate.question_type, normalize_text(candidate.question))
            unique_candidates.setdefault(key, candidate)
        stats["duplicates_removed"] += len(group) - len(unique_candidates)
        ranked = sorted(
            unique_candidates.values(),
            key=lambda candidate: (
                QUESTION_TYPE_PRIORITY.get(candidate.question_type, 99),
                -len(candidate.detailed_explanation),
                normalize_text(candidate.question),
            ),
        )

        # Chọn đa dạng type trước, sau đó bổ sung cùng type đến giới hạn.
        chosen: list[CandidateQa] = []
        chosen_ids: set[tuple[str, str]] = set()
        for candidate in ranked:
            key = (candidate.question_type, normalize_text(candidate.question))
            if candidate.question_type not in {item.question_type for item in chosen}:
                chosen.append(candidate)
                chosen_ids.add(key)
                if len(chosen) >= max_qa_per_topic:
                    break
        if len(chosen) < max_qa_per_topic:
            for candidate in ranked:
                key = (candidate.question_type, normalize_text(candidate.question))
                if key in chosen_ids:
                    continue
                chosen.append(candidate)
                chosen_ids.add(key)
                if len(chosen) >= max_qa_per_topic:
                    break

        selected.extend(chosen)
    return selected, stats


def build_document(candidate: CandidateQa) -> Any:
    """Chuyển candidate sang LangChain Document với metadata đơn giản."""

    from langchain_core.documents import Document

    doc_id = "|".join(
        [candidate.category_id, candidate.topic_id, candidate.image_id, candidate.question_type,
         normalize_text(candidate.question).replace(" ", "_")[:80]]
    )
    metadata = {
        "doc_id": doc_id,
        "category": candidate.category_id,
        "category_label": candidate.category_label,
        "topic_id": candidate.topic_id,
        "topic": candidate.topic,
        "question_type": candidate.question_type,
        "keyword": candidate.keyword,
        "image_id": candidate.image_id,
        "image_path": candidate.image_path,
    }
    return Document(page_content=candidate_content(candidate), metadata=metadata)


def write_outputs(
    output_dir: Path,
    selected: list[CandidateQa],
    rejection_counts: Counter[str],
    raw_question_types: Counter[str],
    selection_stats: dict[str, int],
    dataset_path: Path,
    max_qa_per_topic: int,
    sample_count: int,
) -> dict[str, Any]:
    """Ghi JSONL, JSON thống kê, Markdown report và preview mẫu."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "selected_qa_chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as output_file:
        for candidate in selected:
            row = asdict(candidate)
            row["page_content"] = candidate_content(candidate)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    category_counts = Counter(item.category_id for item in selected)
    topic_counts = Counter((item.category_id, item.topic_id) for item in selected)
    type_counts = Counter(item.question_type for item in selected)
    stats = {
        "dataset_path": str(dataset_path),
        "dataset_size_bytes": dataset_path.stat().st_size,
        "raw_records": None,
        "raw_questions": sum(raw_question_types.values()),
        "raw_question_types": dict(raw_question_types),
        "eligible_before_dedup": selection_stats["eligible_before_dedup"],
        "duplicates_removed": selection_stats["duplicates_removed"],
        "selected_chunks": len(selected),
        "unique_category_topic_pairs": len(topic_counts),
        "max_qa_per_topic": max_qa_per_topic,
        "selected_question_types": dict(type_counts),
        "rejections": dict(rejection_counts),
        "category_summary": {
            category: {
                "topic_count": len({topic for cat, topic in topic_counts if cat == category}),
                "chunk_count": count,
            }
            for category, count in sorted(category_counts.items())
        },
        "chunks_by_category_topic": {
            f"{category}/{topic}": count
            for (category, topic), count in sorted(topic_counts.items())
        },
    }
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        stats["raw_records"] = len(json.load(dataset_file))
    (output_dir / "statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Topic-based QA chunking preview",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Raw image records: **{stats['raw_records']:,}**",
        f"- Raw QA pairs: **{stats['raw_questions']:,}**",
        f"- Eligible before exact dedup: **{selection_stats['eligible_before_dedup']:,}**",
        f"- Exact duplicate QA removed: **{selection_stats['duplicates_removed']:,}**",
        f"- Selected QA chunks: **{len(selected):,}**",
        f"- Category/topic pairs represented: **{len(topic_counts):,}**",
        f"- Maximum QA per pair: **{max_qa_per_topic}**",
        "",
        "## Lý do loại QA",
        "",
    ]
    report_lines.extend(f"- `{reason}`: {count:,}" for reason, count in rejection_counts.most_common())
    report_lines.extend(["", "## Theo category", "", "| Category | Topics | Selected chunks |", "|---|---:|---:|"])
    for category, values in stats["category_summary"].items():
        report_lines.append(f"| {CATEGORY_LABELS.get(category, category)} (`{category}`) | {values['topic_count']} | {values['chunk_count']} |")
    report_lines.extend(["", "## Question type", ""])
    report_lines.extend(f"- `{kind}`: {count:,}" for kind, count in type_counts.most_common())
    (output_dir / "statistics.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    sample_lines = ["# QA chunk samples", ""]
    samples: list[CandidateQa] = []
    preferred_categories = ["am_thuc", "le_hoi", "thu_cong_my_nghe", "kien_truc", "nhac_cu"]
    for category_id in preferred_categories:
        candidate = next((item for item in selected if item.category_id == category_id), None)
        if candidate is not None and candidate not in samples and len(samples) < sample_count:
            samples.append(candidate)
    seen_categories = {candidate.category_id for candidate in samples}
    for candidate in selected:
        if candidate.category_id not in seen_categories and len(samples) < sample_count:
            samples.append(candidate)
            seen_categories.add(candidate.category_id)
    if len(samples) < sample_count:
        sampled_ids = {id(candidate) for candidate in samples}
        samples.extend(candidate for candidate in selected if id(candidate) not in sampled_ids and len(samples) < sample_count)
    for index, candidate in enumerate(samples, start=1):
        sample_lines.extend([
            f"## Sample {index}: {candidate.category_label} / {candidate.topic}",
            "",
            f"**Type:** {candidate.question_type}",
            f"**Question:** {candidate.question}",
            f"**Answer:** {candidate.answer}",
            "",
            "**Detailed explanation:**",
            candidate.detailed_explanation or "(empty)",
            "",
            f"**Cultural significance:** {candidate.cultural_significance or '(empty)' }",
            f"**Source:** `{candidate.image_path}` · image_id `{candidate.image_id}`",
            "",
        ])
    (output_dir / "samples.md").write_text("\n".join(sample_lines), encoding="utf-8")

    catalog_path = output_dir / "topic_catalog.csv"
    with catalog_path.open("w", encoding="utf-8-sig", newline="") as catalog_file:
        writer = csv.writer(catalog_file)
        writer.writerow(["category_id", "category", "topic_id", "topic", "selected_qa_count"])
        grouped_topics: dict[tuple[str, str], CandidateQa] = {}
        for candidate in selected:
            grouped_topics.setdefault((candidate.category_id, candidate.topic_id), candidate)
        for (category_id, topic_id), representative in sorted(grouped_topics.items()):
            writer.writerow([category_id, representative.category_label, topic_id, representative.topic, topic_counts[(category_id, topic_id)]])
    return stats


def build_chroma_index(
    output_dir: Path,
    selected: list[CandidateQa],
    model_name: str,
    device: str,
    batch_size: int,
) -> int:
    """Embed selected QA chunks in a brand-new Chroma directory."""

    from langchain_chroma import Chroma

    persist_dir = output_dir / "chroma_db"
    if persist_dir.exists() and any(persist_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing index: {persist_dir}")
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = E5Embeddings(model_name=model_name, device=device)
    vectorstore = Chroma(
        collection_name="vietculture_topic_qa_v1",
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    documents = [build_document(candidate) for candidate in selected]
    ids = [document.metadata["doc_id"] for document in documents]
    for start in range(0, len(documents), batch_size):
        vectorstore.add_documents(
            documents=documents[start : start + batch_size],
            ids=ids[start : start + batch_size],
        )
        print(f"Indexed {min(start + batch_size, len(documents))}/{len(documents)}")
    return vectorstore._collection.count()


def parse_args() -> argparse.Namespace:
    """Đọc tham số build, hỗ trợ bỏ qua embedding để chỉ xem chunk preview."""

    parser = argparse.ArgumentParser(description="Build a topic-grouped VQA QA index.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "vietnamese_vqa_dataset.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "rebuild_outputs" / "topic_qa_v1")
    parser.add_argument("--max-qa-per-topic", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-embedding", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Điều phối scan, lọc/chọn QA, báo cáo và embedding index riêng."""

    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if args.max_qa_per_topic < 1:
        raise ValueError("--max-qa-per-topic must be at least 1")

    print(f"Scanning {args.dataset} ...")
    candidates, rejection_counts, raw_question_types = scan_dataset(args.dataset)
    selected, selection_stats = deduplicate_and_select(candidates, args.max_qa_per_topic)
    stats = write_outputs(
        output_dir=args.output_dir,
        selected=selected,
        rejection_counts=rejection_counts,
        raw_question_types=raw_question_types,
        selection_stats=selection_stats,
        dataset_path=args.dataset,
        max_qa_per_topic=args.max_qa_per_topic,
        sample_count=args.sample_count,
    )
    print(json.dumps({key: stats[key] for key in [
        "raw_records", "raw_questions", "eligible_before_dedup", "duplicates_removed",
        "selected_chunks", "unique_category_topic_pairs", "rejections",
    ]}, ensure_ascii=False, indent=2))

    if args.skip_embedding:
        print("Embedding skipped by request.")
        return

    count = build_chroma_index(
        output_dir=args.output_dir,
        selected=selected,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    (args.output_dir / "index_summary.json").write_text(
        json.dumps({"collection": "vietculture_topic_qa_v1", "persist_directory": str(args.output_dir / "chroma_db"), "count": count, "model": args.model, "device": args.device}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Chroma index ready: {count} vectors at {args.output_dir / 'chroma_db'}")


if __name__ == "__main__":
    main()

"""
Các hàm gợi ý chủ đề có căn cứ từ dataset VietCulture.

Mục đích file:
Biến sở thích đã lưu trong memory thành query retrieval, rồi format các chunks
lấy từ Chroma thành danh sách gợi ý cá nhân hóa.

Luồng xử lý:
memory -> build_recommendation_query()
-> retriever.retrieve(query)
-> memory + retrieved_chunks -> build_grounded_recommendation_message()

Ý tưởng cá nhân hóa:
Memory quyết định "nên tìm gì", còn retrieved chunks quyết định "nói gì cho có
căn cứ". Memory không được dùng như bằng chứng tri thức.
"""

from __future__ import annotations

import re
from typing import Any

from src.memory.store import DATASET_CATEGORY_LABELS, load_memory_json
from src.routing.text_utils import normalize_text, unique_values


def build_recommendation_query(memory: dict[str, Any]) -> str:
    """
    Tạo query retrieval từ sở thích đã lưu.

    Biến đầu vào:
    - memory: dict chứa categories/topics/normalized_keywords của user.

    Ví dụ output:
    memory có topics=["lễ hội"], categories=["le_hoi"]
    -> "gợi ý chủ đề văn hóa Việt Nam lễ hội"

    Cách tự viết lại:
    Lấy topics + normalized keywords + category labels, deduplicate, rồi ghép
    thành một câu query ngắn để đưa vào retriever.
    """

    memory = load_memory_json(memory)
    topics = memory.get("topics", []) or memory.get("keywords", [])
    normalized_keywords = memory.get("normalized_keywords", [])
    category_labels = [
        DATASET_CATEGORY_LABELS.get(category, category)
        for category in memory.get("categories", [])
    ]

    focus_terms = unique_values(topics + normalized_keywords + category_labels)
    if not focus_terms:
        return ""

    return "gợi ý chủ đề văn hóa Việt Nam " + " ".join(focus_terms)


def build_recommendation_message(memory: dict[str, Any]) -> str:
    """
    Tạo fallback recommendation khi chưa retrieve documents.

    Ví dụ output khi có memory:
    "Dựa trên sở thích đã lưu, bạn có thể bắt đầu với: lễ hội..."

    Ví dụ output khi chưa có memory:
    "Mình chưa có đủ sở thích của bạn... Bạn quan tâm hơn đến nhóm nào: ..."

    Cách tự viết lại:
    Nếu build được recommendation query thì nhắc lại focus terms. Nếu chưa có
    sở thích, hỏi user chọn một nhóm category để cold-start.
    """

    recommendation_query = build_recommendation_query(memory)
    if recommendation_query:
        focus = recommendation_query.replace("gợi ý chủ đề văn hóa Việt Nam ", "")
        return (
            f"Dựa trên sở thích đã lưu, bạn có thể bắt đầu với: {focus}. "
            "Bạn hãy chọn một chủ đề cụ thể, mình sẽ dùng dataset để trả lời có căn cứ."
        )

    available_categories = ", ".join(DATASET_CATEGORY_LABELS.values())
    return (
        "Mình chưa có đủ sở thích của bạn để gợi ý chắc tay. "
        f"Bạn quan tâm hơn đến nhóm nào: {available_categories}?"
    )


def build_grounded_recommendation_message(
    memory: dict[str, Any],
    retrieved_chunks: list[Any],
) -> str:
    """
    Format retrieved chunks thành danh sách gợi ý cá nhân hóa.

    Biến đầu vào:
    - memory: sở thích user.
    - retrieved_chunks: kết quả từ QaRetriever, mỗi chunk có document + metadata.

    Ví dụ output:
    1. Bánh tét
       - Phân loại: nhóm ẩm thực; dạng câu hỏi cultural
       - Vì sao hợp: khớp với sở thích ẩm thực đã lưu trong memory.
       - Câu hỏi nên thử: Ý nghĩa văn hóa của bánh tét là gì?

    Cách tự viết lại:
    Duyệt top chunks, lấy topic/category/question từ metadata, bỏ trùng topic,
    tạo lý do vì sao hợp với memory, rồi giới hạn khoảng 3 gợi ý.
    """

    memory = load_memory_json(memory)
    recommendation_query = build_recommendation_query(memory)
    if not recommendation_query:
        return build_recommendation_message(memory)

    if not retrieved_chunks:
        return (
            "Mình có sở thích của bạn trong memory, nhưng chưa tìm thấy tài liệu "
            "phù hợp trong dataset để gợi ý có căn cứ."
        )

    user_interest_summary = build_user_interest_summary(memory)
    recommendations: list[str] = []
    seen_topics: set[str] = set()

    for chunk in retrieved_chunks:
        document = getattr(chunk, "document", chunk)
        metadata = getattr(document, "metadata", {}) or {}
        content = getattr(document, "page_content", "")

        topic = (
            metadata.get("canonical_topic")
            or metadata.get("topic")
            or metadata.get("retrieval_anchor")
            or metadata.get("keyword")
            or "chủ đề trong dataset"
        )
        topic = str(topic).strip()
        normalized_topic = normalize_text(topic)
        if normalized_topic in seen_topics:
            continue

        seen_topics.add(normalized_topic)
        display_topic = prettify_topic_for_display(topic)
        category = DATASET_CATEGORY_LABELS.get(
            str(metadata.get("category", "")),
            str(metadata.get("category", "")),
        )
        question_type = str(metadata.get("question_type", "")).strip()
        question = str(metadata.get("question", "")).strip()
        if not question:
            question = extract_content_section(content, ["Question:"])

        short_answer = extract_content_section(content, ["Answer:"])
        reason = extract_recommendation_reason(content)
        fit_reason = build_recommendation_fit_reason(
            memory=memory,
            category=category,
            topic=display_topic,
            content=content,
        )

        recommendation_lines = [f"{len(recommendations) + 1}. {display_topic}"]
        detail_parts = []
        if category:
            detail_parts.append(f"nhóm {category}")
        if question_type:
            detail_parts.append(f"dạng câu hỏi {question_type}")
        if detail_parts:
            recommendation_lines.append("   - Phân loại: " + "; ".join(detail_parts))
        if fit_reason:
            recommendation_lines.append("   - Vì sao hợp: " + fit_reason)
        if question:
            recommendation_lines.append("   - Câu hỏi nên thử: " + question)
        if short_answer:
            recommendation_lines.append("   - Trả lời ngắn: " + compact_text(short_answer, max_chars=180))
        elif reason:
            recommendation_lines.append("   - Gợi ý nội dung: " + compact_text(reason, max_chars=180))

        recommendations.append("\n".join(recommendation_lines))

        if len(recommendations) >= 3:
            break

    if not recommendations:
        return build_recommendation_message(memory)

    return (
        "Dựa trên sở thích đã lưu"
        + user_interest_summary
        + " và các tài liệu tìm được trong dataset, mình gợi ý bạn bắt đầu với:\n\n"
        + "\n\n".join(recommendations)
    )


def extract_recommendation_reason(page_content: str, max_chars: int = 180) -> str:
    """
    Lấy một đoạn ngắn trong chunk để làm lý do/gợi ý nội dung.

    Ví dụ output:
    "Bánh tét thường xuất hiện trong dịp Tết và mang ý nghĩa đoàn tụ..."

    Cách tự viết lại:
    Ưu tiên lấy các section giàu ý nghĩa như Cultural Significance, Detailed
    Explanation, Answer. Nếu không có marker thì lấy đoạn đầu page_content.
    """

    if not page_content:
        return ""

    preferred_markers = [
        "Cultural Knowledge:\n",
        "Cultural Significance:\n",
        "Detailed Explanation:\n",
        "Answer:\n",
    ]
    for marker in preferred_markers:
        if marker not in page_content:
            continue

        start = page_content.index(marker) + len(marker)
        end = page_content.find("\n\n", start)
        if end == -1:
            end = len(page_content)
        return page_content[start:end].strip()[:max_chars]

    return page_content.strip().replace("\n", " ")[:max_chars]


def build_user_interest_summary(memory: dict[str, Any]) -> str:
    """
    Tóm tắt sở thích user thành một cụm ngắn để đưa vào câu mở đầu.

    Ví dụ output:
    memory categories=["le_hoi"], topics=["Tết"]
    -> " về lễ hội, Tết"

    Cách tự viết lại:
    Chuyển category id sang nhãn tiếng Việt, cộng với topics, deduplicate, rồi
    ghép bằng dấu phẩy.
    """

    categories = [
        DATASET_CATEGORY_LABELS.get(category, category)
        for category in memory.get("categories", [])
    ]
    topics = memory.get("topics", []) or memory.get("keywords", [])
    focus_terms = unique_values(categories + topics)

    if not focus_terms:
        return ""

    return " về " + ", ".join(focus_terms)


def build_recommendation_fit_reason(
    memory: dict[str, Any],
    category: str,
    topic: str,
    content: str,
) -> str:
    """
    Giải thích vì sao một chunk phù hợp với sở thích user.

    Biến đầu vào:
    - memory: sở thích đã lưu.
    - category/topic/content: thông tin của chunk đang xét.

    Ví dụ output:
    "khớp với sở thích lễ hội đã lưu trong memory."

    Cách tự viết lại:
    Normalize topic/content/category, so khớp với topics/categories trong memory.
    Nếu không match rõ thì trả lý do fallback: được chọn từ retrieval gần nhất.
    """

    normalized_category = normalize_text(category)
    normalized_topic = normalize_text(topic)
    normalized_content = normalize_text(content)

    matched_terms: list[str] = []
    for term in memory.get("topics", []) + memory.get("keywords", []):
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        if normalized_term in normalized_topic or normalized_term in normalized_content:
            matched_terms.append(str(term))

    for category_key in memory.get("categories", []):
        category_label = DATASET_CATEGORY_LABELS.get(category_key, category_key)
        if normalize_text(category_label) == normalized_category:
            matched_terms.append(category_label)

    matched_terms = unique_values(matched_terms)
    if matched_terms:
        return "khớp với sở thích " + ", ".join(matched_terms) + " đã lưu trong memory."

    return "được chọn từ các tài liệu gần nhất với truy vấn recommendation."


def extract_content_section(page_content: str, section_markers: list[str]) -> str:
    """
    Lấy nội dung của section đầu tiên khớp marker.

    Ví dụ output:
    extract_content_section(chunk, ["Answer:"]) -> "Bánh tét là..."

    Cách tự viết lại:
    Tìm vị trí marker, lấy text từ sau marker đến dòng trống tiếp theo. Đây là
    parser đơn giản vì chunk hiện được format bằng các section text rõ ràng.
    """

    if not page_content:
        return ""

    for marker in section_markers:
        if marker not in page_content:
            continue

        start = page_content.index(marker) + len(marker)
        end = page_content.find("\n\n", start)
        if end == -1:
            end = len(page_content)
        return page_content[start:end].strip()

    return ""


def compact_text(text: str, max_chars: int = 180) -> str:
    """
    Rút gọn text dài để hiển thị trong recommendation.

    Ví dụ output:
    compact_text("A" * 300, 20) -> "AAAAAAAAAAAAAAAAA..."

    Cách tự viết lại:
    Chuẩn hóa whitespace, nếu text dài hơn max_chars thì cắt và thêm "...".
    """

    compacted_text = re.sub(r"\s+", " ", str(text)).strip()
    if len(compacted_text) <= max_chars:
        return compacted_text

    return compacted_text[: max_chars - 3].rstrip() + "..."


def prettify_topic_for_display(topic: str) -> str:
    """
    Làm đẹp topic legacy/không dấu trước khi hiển thị.

    Ví dụ output:
    prettify_topic_for_display("banh tet") -> "Bánh tét"

    Cách tự viết lại:
    Tạo map replacements cho các từ phổ biến trong dataset. Normalize topic,
    thay các cụm không dấu bằng tiếng Việt có dấu, rồi viết hoa chữ đầu.
    """

    display_topic = str(topic or "").strip()
    if not display_topic:
        return "Chủ đề trong dataset"

    replacements = {
        "le hoi": "lễ hội",
        "am thuc": "ẩm thực",
        "banh chung": "bánh chưng",
        "banh tet": "bánh tét",
        "xe may": "xe máy",
        "tham coi": "thảm cói",
        "ao dai": "áo dài",
        "non la": "nón lá",
        "phong canh": "phong cảnh",
        "thu cong": "thủ công",
        "my nghe": "mỹ nghệ",
        "giao thong": "giao thông",
        "kien truc": "kiến trúc",
    }

    normalized_display = normalize_text(display_topic)
    for plain_text, vietnamese_text in replacements.items():
        normalized_display = normalized_display.replace(plain_text, vietnamese_text)

    if normalized_display != normalize_text(display_topic):
        display_topic = normalized_display

    if display_topic.islower():
        return display_topic[:1].upper() + display_topic[1:]

    return display_topic

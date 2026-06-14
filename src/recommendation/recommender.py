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
    category_labels = [
        DATASET_CATEGORY_LABELS.get(category, category)
        for category in memory.get("categories", [])
    ]

    focus_terms = unique_values_by_normalized(topics + category_labels)
    if not focus_terms:
        return ""

    return "gợi ý chủ đề văn hóa Việt Nam " + " ".join(focus_terms)


def build_recommendation_queries(memory: dict[str, Any]) -> list[str]:
    """
    Tạo nhiều query nhỏ để recommendation đa dạng theo từng sở thích.

    Biến đầu vào:
    - memory: dict sở thích hiện tại của user.

    Ví dụ output:
    ["gợi ý chủ đề văn hóa Việt Nam thể thao",
     "gợi ý chủ đề văn hóa Việt Nam kiến trúc"]

    Cách tự viết lại:
    Thay vì nhồi mọi sở thích vào một query dài, tách từng topic/category thành
    query riêng. Retrieval sẽ lấy ứng viên đa dạng hơn trước khi formatter chọn.
    """

    memory = load_memory_json(memory)
    topics = memory.get("topics", []) or memory.get("keywords", [])
    category_labels = [
        DATASET_CATEGORY_LABELS.get(category, category)
        for category in memory.get("categories", [])
    ]
    focus_terms = unique_values_by_normalized(topics + category_labels)
    if not focus_terms:
        return []

    queries = [
        "gợi ý chủ đề văn hóa Việt Nam " + focus_term
        for focus_term in focus_terms
    ]
    combined_query = build_recommendation_query(memory)
    if combined_query:
        queries.append(combined_query)
    return unique_values_by_normalized(queries)


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
    Mình chọn 3 hướng khá hợp với bạn:

    1. Bánh tét
       Vì bạn đang quan tâm đến ẩm thực, chủ đề này giúp đi từ món ăn quen
       thuộc sang ý nghĩa đoàn tụ trong ngày Tết.
       Bạn có thể hỏi tiếp: Ý nghĩa văn hóa của bánh tét là gì?

    Cách tự viết lại:
    Duyệt top chunks, lấy topic/category/question từ metadata, bỏ trùng topic,
    tạo lý do tự nhiên, rồi giới hạn khoảng 3 gợi ý để người dùng dễ chọn.
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

    for chunk in rank_recommendation_candidates(memory, retrieved_chunks):
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
        question = build_suggested_question(
            topic=display_topic,
            raw_question=question,
            question_type=question_type,
        )

        short_answer = extract_content_section(content, ["Answer:"])
        reason = extract_recommendation_reason(content)
        fit_reason = build_recommendation_fit_reason(
            memory=memory,
            category=category,
            topic=display_topic,
            content=content,
        )

        category_phrase = build_category_phrase(category)
        recommendation_lines = [f"{len(recommendations) + 1}. {display_topic}"]
        if fit_reason:
            recommendation_lines.append(
                "   " + build_recommendation_intro_sentence(
                    topic=display_topic,
                    category_phrase=category_phrase,
                    fit_reason=fit_reason,
                )
            )
        if question:
            recommendation_lines.append("   Bạn có thể hỏi tiếp: " + question)
        if short_answer:
            recommendation_lines.append("   Điểm thú vị: " + compact_text(short_answer, max_chars=170))
        elif reason:
            recommendation_lines.append("   Điểm thú vị: " + compact_text(reason, max_chars=170))

        recommendations.append("\n".join(recommendation_lines))

        if len(recommendations) >= 3:
            break

    if not recommendations:
        return build_recommendation_message(memory)

    return (
        "Mình chọn 3 hướng khá hợp với sở thích đã lưu"
        + user_interest_summary
        + ". Bạn có thể bắt đầu từ một trong các chủ đề này:\n\n"
        + "\n\n".join(recommendations)
    )


def build_category_phrase(category: str) -> str:
    """
    Chuyển category thành cụm diễn đạt tự nhiên trong câu recommendation.

    Biến đầu vào:
    - category: nhãn category đã chuyển sang tiếng Việt, ví dụ "ẩm thực".

    Ví dụ output:
    build_category_phrase("ẩm thực") -> "mảng ẩm thực"

    Cách tự viết lại:
    Nếu có category thì thêm tiền tố "mảng"; nếu rỗng thì dùng cụm trung tính.
    """

    clean_category = str(category or "").strip()
    if not clean_category:
        return "một mảng văn hóa trong dataset"
    return "mảng " + clean_category


def build_recommendation_intro_sentence(
    topic: str,
    category_phrase: str,
    fit_reason: str,
) -> str:
    """
    Tạo một câu giải thích ngắn thay cho dòng metadata khô cứng.

    Biến đầu vào:
    - topic: tên chủ đề đang gợi ý.
    - category_phrase: cụm category tự nhiên, ví dụ "mảng kiến trúc".
    - fit_reason: lý do match với memory đã build ở bước trước.

    Ví dụ output:
    "Chủ đề này nằm trong mảng ẩm thực và khớp với sở thích Tết đã lưu..."

    Cách tự viết lại:
    Ghép category + fit_reason thành một câu hoàn chỉnh. Không nhắc tới score,
    vector, question_type hay metadata để người dùng không thấy cảm giác debug.
    """

    clean_topic = str(topic or "chủ đề này").strip()
    clean_fit_reason = str(fit_reason or "").strip()
    if clean_fit_reason:
        return (
            f"{clean_topic} nằm trong {category_phrase}; "
            + clean_fit_reason[:1].lower()
            + clean_fit_reason[1:]
        )
    return f"{clean_topic} nằm trong {category_phrase}, phù hợp để bạn mở rộng cuộc trò chuyện."


def unique_values_by_normalized(values: list[str]) -> list[str]:
    """
    Deduplicate theo normalize_text để tránh lặp có dấu/không dấu.

    Ví dụ output:
    ["thể thao", "the thao", "kiến trúc"] -> ["thể thao", "kiến trúc"]

    Cách tự viết lại:
    Dùng normalize_text(value) làm key, nhưng giữ lại display text gốc đầu tiên.
    """

    seen_normalized: set[str] = set()
    clean_values: list[str] = []
    for value in values:
        display_value = str(value or "").strip()
        normalized_value = normalize_text(display_value)
        if not display_value or not normalized_value or normalized_value in seen_normalized:
            continue
        seen_normalized.add(normalized_value)
        clean_values.append(display_value)
    return clean_values


def rank_recommendation_candidates(
    memory: dict[str, Any],
    retrieved_chunks: list[Any],
) -> list[Any]:
    """
    Sắp xếp ứng viên recommendation để ưu tiên đa dạng category.

    Biến đầu vào:
    - memory: memory user, dùng lấy thứ tự category user quan tâm.
    - retrieved_chunks: danh sách chunks đã retrieve từ nhiều query.

    Ví dụ output:
    Nếu memory có thể thao, kiến trúc, ẩm thực thì top candidates cố gắng có cả
    ba nhóm thay vì toàn thể thao.

    Cách tự viết lại:
    Duyệt theo category trong memory trước, lấy chunk tốt nhất mỗi category,
    sau đó mới fill phần còn thiếu bằng các chunk còn lại.
    """

    memory = load_memory_json(memory)
    preferred_categories = list(memory.get("categories", []))
    selected_chunks: list[Any] = []
    selected_ids: set[str] = set()

    def chunk_key(chunk: Any) -> str:
        document = getattr(chunk, "document", chunk)
        metadata = getattr(document, "metadata", {}) or {}
        return "|".join(
            [
                str(metadata.get("category", "")),
                normalize_text(metadata.get("keyword") or metadata.get("topic") or ""),
                str(metadata.get("image_id", "")),
                str(metadata.get("question_type", "")),
            ]
        )

    def chunk_category(chunk: Any) -> str:
        document = getattr(chunk, "document", chunk)
        metadata = getattr(document, "metadata", {}) or {}
        return str(metadata.get("category", ""))

    for category in preferred_categories:
        for chunk in retrieved_chunks:
            key = chunk_key(chunk)
            if key in selected_ids or chunk_category(chunk) != category:
                continue
            selected_chunks.append(chunk)
            selected_ids.add(key)
            break

    for chunk in retrieved_chunks:
        key = chunk_key(chunk)
        if key in selected_ids:
            continue
        selected_chunks.append(chunk)
        selected_ids.add(key)

    return selected_chunks


def build_suggested_question(
    topic: str,
    raw_question: str,
    question_type: str,
) -> str:
    """
    Làm câu hỏi gợi ý cụ thể hơn, tránh câu chung chung "hình ảnh này".

    Ví dụ output:
    topic="Bánh chưng", question_type="cultural"
    -> "Ý nghĩa văn hóa của Bánh chưng là gì?"

    Cách tự viết lại:
    Nếu raw_question còn phụ thuộc ảnh/context, sinh câu hỏi template theo topic.
    Nếu raw_question đã cụ thể thì giữ lại.
    """

    normalized_question = normalize_text(raw_question)
    generic_markers = [
        "hinh anh nay",
        "hinh nay",
        "vat nay",
        "mon nay",
        "day la",
    ]
    if raw_question and not any(marker in normalized_question for marker in generic_markers):
        return raw_question

    normalized_type = normalize_text(question_type)
    if "comparison" in normalized_type:
        return f"{topic} có điểm gì đáng chú ý so với các chủ đề văn hóa liên quan?"
    if "analysis" in normalized_type:
        return f"Vì sao {topic} đáng chú ý trong văn hóa Việt Nam?"
    if "description" in normalized_type:
        return f"{topic} có những đặc điểm nổi bật nào?"
    if "identification" in normalized_type:
        return f"{topic} là gì?"
    return f"Ý nghĩa văn hóa của {topic} là gì?"


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

    topics = memory.get("topics", []) or memory.get("keywords", [])
    categories = [
        DATASET_CATEGORY_LABELS.get(category, category)
        for category in memory.get("categories", [])
    ]
    focus_terms = remove_redundant_focus_terms(
        unique_values_by_normalized(topics + categories)
    )

    if not focus_terms:
        return ""

    return " về " + ", ".join(focus_terms)


def remove_redundant_focus_terms(focus_terms: list[str]) -> list[str]:
    """
    Bỏ các cụm sở thích bị lặp nghĩa để câu mở đầu recommendation gọn hơn.

    Biến đầu vào:
    - focus_terms: danh sách sở thích đã deduplicate theo normalize_text.

    Ví dụ output:
    ["thể thao", "thể thao truyền thống", "kiến trúc"] -> ["thể thao", "kiến trúc"]

    Cách tự viết lại:
    So sánh dạng normalize của từng term. Nếu một term dài chứa term ngắn đã có,
    giữ term ngắn vì nó đại diện rộng hơn và đọc tự nhiên hơn trong câu mở đầu.
    """

    cleaned_terms: list[str] = []
    normalized_terms: list[str] = []
    for term in focus_terms:
        normalized_term = normalize_text(term)
        if any(
            existing and existing in normalized_term
            for existing in normalized_terms
        ):
            continue
        cleaned_terms.append(term)
        normalized_terms.append(normalized_term)
    return cleaned_terms


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

    matched_terms = remove_redundant_focus_terms(unique_values_by_normalized(matched_terms))
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
        "vat co truyen": "vật cổ truyền",
        "le hoi": "lễ hội",
        "am thuc": "ẩm thực",
        "banh bao": "bánh bao",
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
        "nha truyen thong": "nhà truyền thống",
        "mien tay": "miền Tây",
        "co truyen": "cổ truyền",
        "truyen thong": "truyền thống",
        "cu ta": "cử tạ",
        "the thao": "thể thao",
        "viet nam": "Việt Nam",
    }

    normalized_display = normalize_text(display_topic)
    for plain_text, vietnamese_text in replacements.items():
        normalized_display = normalized_display.replace(plain_text, vietnamese_text)

    if normalized_display != normalize_text(display_topic):
        display_topic = normalized_display

    if display_topic and display_topic[:1].islower():
        return display_topic[:1].upper() + display_topic[1:]

    return display_topic

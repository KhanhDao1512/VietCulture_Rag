"""
Rule-based intent routing, memory helpers, and grounded recommendation helpers.

Overall flow:
user message -> intent label -> optional memory update/answer -> graph route

This module intentionally avoids LLM calls. It gives the notebook a stable
baseline for routing while RAG generation can be improved separately.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# =============================================================================
# Configuration
# =============================================================================

SUPPORTED_INTENTS = {
    "rag_question",
    "followup_question",
    "recommendation_request",
    "preference_update",
    "memory_query",
    "chitchat",
    "out_of_scope",
}

DEFAULT_MEMORY = {
    "categories": [],
    "topics": [],
    "keywords": [],
    "normalized_keywords": [],
    "question_styles": [],
    "evidence": [],
    "last_updated": "",
}

DATASET_CATEGORY_LABELS = {
    "am_thuc": "ẩm thực",
    "kien_truc": "kiến trúc",
    "le_hoi": "lễ hội",
    "phong_canh": "phong cảnh",
    "trang_phuc": "trang phục",
    "doi_song_hang_ngay": "đời sống hằng ngày",
    "giao_thong": "giao thông",
    "thu_cong_my_nghe": "thủ công mỹ nghệ",
    "nhac_cu": "nhạc cụ",
    "van_hoa_dan_gian": "văn hóa dân gian",
    "tro_choi_dan_gian": "trò chơi dân gian",
    "the_thao_truyen_thong": "thể thao truyền thống",
}

CATEGORY_PATTERNS = {
    "am_thuc": ["am thuc", "mon an", "do an", "banh", "pho", "bun", "com"],
    "kien_truc": ["kien truc", "nha", "chua", "dinh", "den", "cong trinh"],
    "le_hoi": ["le hoi", "tet", "ram", "trung thu", "hoi"],
    "phong_canh": ["phong canh", "canh dep", "bien", "nui", "song", "ruong"],
    "trang_phuc": ["trang phuc", "ao dai", "non la", "quan ao"],
    "doi_song_hang_ngay": ["doi song", "hang ngay", "sinh hoat", "cho que"],
    "giao_thong": ["giao thong", "xe may", "xe om", "xe buyt", "duong pho"],
    "thu_cong_my_nghe": ["thu cong", "my nghe", "tham coi", "gom", "may tre"],
    "nhac_cu": ["nhac cu", "dan bau", "dan tranh", "dan nguyet"],
    "van_hoa_dan_gian": ["dan gian", "truyen thuyet", "co tich", "roi nuoc"],
    "tro_choi_dan_gian": ["tro choi", "keo co", "o an quan", "danh du"],
    "the_thao_truyen_thong": ["the thao", "vat", "dua thuyen", "vo co truyen"],
}

FOLLOWUP_MARKERS = [
    "no",
    "mon nay",
    "thu nay",
    "cai nay",
    "chu de nay",
    "hinh nay",
    "van de nay",
]

QUESTION_WORDS = [
    "la gi",
    "vi sao",
    "tai sao",
    "nhu the nao",
    "co y nghia gi",
    "y nghia",
    "so sanh",
    "nguon goc",
    "mo ta",
    "phan tich",
]


# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class IntentDecision:
    """A small, inspectable routing result."""

    intent: str
    reason: str


# =============================================================================
# Text helpers
# =============================================================================

def normalize_text(text: Any) -> str:
    """Normalize Vietnamese text so keyword matching is accent-insensitive."""

    if text is None:
        return ""

    raw_text = str(text).lower()
    decomposed_text = unicodedata.normalize("NFD", raw_text)
    accentless_text = "".join(
        character
        for character in decomposed_text
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"[^a-z0-9]+", " ", accentless_text).strip()


def unique_values(values: list[str]) -> list[str]:
    """Keep non-empty strings once, preserving order."""

    seen_values: set[str] = set()
    clean_values: list[str] = []

    for value in values:
        clean_value = str(value).strip()
        if not clean_value or clean_value in seen_values:
            continue

        seen_values.add(clean_value)
        clean_values.append(clean_value)

    return clean_values


def load_memory_json(memory_text: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse a user memory object and fill missing fields."""

    if isinstance(memory_text, dict):
        loaded_memory = memory_text
    elif memory_text:
        try:
            loaded_memory = json.loads(memory_text)
        except json.JSONDecodeError:
            loaded_memory = {}
    else:
        loaded_memory = {}

    memory = dict(DEFAULT_MEMORY)
    memory.update(loaded_memory)

    for field_name in [
        "categories",
        "topics",
        "keywords",
        "normalized_keywords",
        "question_styles",
        "evidence",
    ]:
        field_value = memory.get(field_name, [])
        memory[field_name] = field_value if isinstance(field_value, list) else []

    return memory


def dump_memory_json(memory: dict[str, Any]) -> str:
    """Serialize memory consistently for notebook state."""

    return json.dumps(memory, ensure_ascii=False)


def build_thread_id(user_id: str, conversation_id: str) -> str:
    """
    Build the short-term conversation key used by LangGraph.

    `user_id` is for long-term memory. `conversation_id` is for one chat thread.
    Keeping both in the thread id prevents histories from different users from
    being mixed by the graph checkpointer.
    """

    safe_user_id = normalize_text(user_id).replace(" ", "_") or "anonymous"
    safe_conversation_id = normalize_text(conversation_id).replace(" ", "_") or "default"
    return f"user:{safe_user_id}:thread:{safe_conversation_id}"


def build_langgraph_config(user_id: str, conversation_id: str) -> dict[str, dict[str, str]]:
    """Build the config object expected by LangGraph's checkpointer."""

    return {
        "configurable": {
            "thread_id": build_thread_id(
                user_id=user_id,
                conversation_id=conversation_id,
            )
        }
    }


# =============================================================================
# Intent classification
# =============================================================================

def classify_intent(user_message: str) -> IntentDecision:
    """Classify the latest user message into one graph route."""

    normalized_message = normalize_text(user_message)

    if not normalized_message:
        return IntentDecision("chitchat", "Tin nhắn rỗng hoặc quá ngắn.")

    if is_memory_query(normalized_message):
        return IntentDecision(
            "memory_query",
            "Người dùng hỏi lại thông tin đã lưu trong memory.",
        )

    if is_preference_update(normalized_message):
        return IntentDecision(
            "preference_update",
            "Người dùng đang nói về sở thích.",
        )

    if is_recommendation_request(normalized_message):
        return IntentDecision(
            "recommendation_request",
            "Người dùng muốn được gợi ý.",
        )

    if is_followup_question(normalized_message):
        return IntentDecision(
            "followup_question",
            "Câu hỏi phụ thuộc vào ngữ cảnh hội thoại trước.",
        )

    if is_dataset_question(normalized_message):
        return IntentDecision(
            "rag_question",
            "Câu hỏi phù hợp với miền dữ liệu văn hóa Việt Nam.",
        )

    if is_chitchat(normalized_message):
        return IntentDecision("chitchat", "Tin nhắn xã giao không cần retrieval.")

    return IntentDecision("out_of_scope", "Không nhận ra câu hỏi thuộc phạm vi dataset.")


def is_preference_update(normalized_message: str) -> bool:
    """Detect statements like 'tôi thích lễ hội'."""

    preference_markers = [
        "toi thich",
        "minh thich",
        "t thich",
        "toi quan tam",
        "minh quan tam",
        "toi muon tim hieu ve",
        "minh muon tim hieu ve",
    ]
    return any(marker in normalized_message for marker in preference_markers)


def is_memory_query(normalized_message: str) -> bool:
    """Detect questions that ask what the assistant remembers."""

    memory_markers = [
        "toi thich gi",
        "ban nho gi ve toi",
        "ban biet gi ve toi",
        "so thich cua toi",
        "toi quan tam gi",
    ]
    return any(marker in normalized_message for marker in memory_markers)


def is_recommendation_request(normalized_message: str) -> bool:
    """Detect recommendation requests."""

    recommendation_markers = [
        "goi y",
        "de xuat",
        "nen hoc",
        "nen tim hieu",
        "chu de hay",
        "recommend",
    ]
    return any(marker in normalized_message for marker in recommendation_markers)


def is_followup_question(normalized_message: str) -> bool:
    """Detect follow-up questions that need chat history resolution."""

    has_followup_marker = any(marker in normalized_message for marker in FOLLOWUP_MARKERS)
    has_question_shape = any(word in normalized_message for word in QUESTION_WORDS)
    return has_followup_marker and has_question_shape


def is_dataset_question(normalized_message: str) -> bool:
    """Detect ordinary RAG questions in the Vietnamese culture dataset scope."""

    has_question_shape = "?" in normalized_message or any(
        word in normalized_message
        for word in QUESTION_WORDS
    )
    has_category_hint = any(
        pattern in normalized_message
        for patterns in CATEGORY_PATTERNS.values()
        for pattern in patterns
    )
    return has_question_shape or has_category_hint


def is_chitchat(normalized_message: str) -> bool:
    """Detect common conversational messages."""

    chitchat_messages = [
        "xin chao",
        "chao",
        "hello",
        "hi",
        "cam on",
        "ok",
        "oke",
        "uh",
    ]
    return (
        normalized_message in chitchat_messages
        or normalized_message.startswith("chao ")
        or normalized_message.startswith("xin chao ")
    )


# =============================================================================
# Memory extraction and persistence
# =============================================================================

def extract_memory_updates_from_text(user_message: str) -> dict[str, list[str]]:
    """Extract categories/topics/keywords from a preference statement."""

    normalized_message = normalize_text(user_message)
    categories = detect_categories(normalized_message)
    topics = extract_topics_from_preference_text(user_message)

    return {
        "categories": categories,
        "topics": topics,
        "keywords": topics,
        "normalized_keywords": [normalize_text(topic) for topic in topics],
    }


def detect_categories(normalized_text: str) -> list[str]:
    """Map text phrases into dataset category ids."""

    detected_categories: list[str] = []

    for category_id, patterns in CATEGORY_PATTERNS.items():
        if any(pattern in normalized_text for pattern in patterns):
            detected_categories.append(category_id)

    return detected_categories


def extract_topics_from_preference_text(user_message: str) -> list[str]:
    """
    Extract the object after simple preference markers.

    Example: "Tôi thích lễ hội và ẩm thực" -> ["lễ hội", "ẩm thực"]
    """

    text = str(user_message).strip()
    marker_pattern = re.compile(
        r"(tôi|toi|mình|minh|t)\s+"
        r"(thích|thich|quan tâm|quan tam|muốn tìm hiểu về|muon tim hieu ve)\s+",
        flags=re.IGNORECASE,
    )
    cleaned_text = marker_pattern.sub("", text).strip()
    cleaned_text = re.sub(r"[.!?]+$", "", cleaned_text).strip()

    if not cleaned_text or cleaned_text == text:
        return []

    rough_topics = re.split(r",| và | với |;|/|\\|&", cleaned_text)
    return unique_values([topic.strip() for topic in rough_topics])


def merge_memory(
    current_memory: dict[str, Any],
    memory_updates: dict[str, list[str]],
    evidence_text: str = "",
) -> dict[str, Any]:
    """Merge extracted preferences into the existing memory object."""

    merged_memory = load_memory_json(current_memory)

    for field_name in ["categories", "topics", "keywords", "normalized_keywords"]:
        merged_memory[field_name] = unique_values(
            list(merged_memory.get(field_name, []))
            + memory_updates.get(field_name, [])
        )

    if evidence_text:
        merged_memory["evidence"] = unique_values(
            list(merged_memory.get("evidence", []))
            + [evidence_text]
        )[-10:]

    merged_memory["last_updated"] = datetime.now().isoformat(timespec="seconds")
    return merged_memory


def load_user_memory(memory_file: str | Path, user_id: str) -> dict[str, Any]:
    """Load one user's memory from a JSON file."""

    memory_path = Path(memory_file)
    if not memory_path.exists():
        return dict(DEFAULT_MEMORY)

    try:
        memory_db = json.loads(memory_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return dict(DEFAULT_MEMORY)

    return load_memory_json(memory_db.get(user_id, {}))


def save_user_memory(
    memory_file: str | Path,
    user_id: str,
    memory: dict[str, Any],
) -> None:
    """Save one user's memory back into the JSON memory file."""

    memory_path = Path(memory_file)
    memory_db: dict[str, Any] = {}

    if memory_path.exists():
        try:
            memory_db = json.loads(memory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            memory_db = {}

    memory_db[user_id] = memory
    memory_path.write_text(
        json.dumps(memory_db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def update_memory_from_retrieved_documents(
    current_memory: dict[str, Any],
    documents: list[Any],
    evidence_text: str = "",
) -> dict[str, Any]:
    """Update memory using metadata from retrieved documents."""

    memory_updates = {
        "categories": [],
        "topics": [],
        "keywords": [],
        "normalized_keywords": [],
    }

    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        category = str(metadata.get("category", "")).strip()
        keyword = str(metadata.get("keyword", "")).strip()
        normalized_keyword = str(metadata.get("normalized_keyword", "")).strip()

        if category:
            memory_updates["categories"].append(category)
        if keyword:
            memory_updates["topics"].append(keyword)
            memory_updates["keywords"].append(keyword)
        if normalized_keyword:
            memory_updates["normalized_keywords"].append(normalized_keyword)

    return merge_memory(current_memory, memory_updates, evidence_text=evidence_text)


# =============================================================================
# Response helpers for non-RAG intents
# =============================================================================

def build_memory_summary(memory: dict[str, Any]) -> str:
    """Build a short Vietnamese answer for 'what do I like?'."""

    categories = [
        DATASET_CATEGORY_LABELS.get(category, category)
        for category in memory.get("categories", [])
    ]
    topics = memory.get("topics", []) or memory.get("keywords", [])

    if not categories and not topics:
        return "Mình chưa lưu được sở thích rõ ràng nào của bạn."

    parts: list[str] = []
    if categories:
        parts.append("nhóm chủ đề: " + ", ".join(categories))
    if topics:
        parts.append("từ khóa/chủ đề: " + ", ".join(topics))

    return "Mình đang nhớ bạn quan tâm đến " + "; ".join(parts) + "."


def build_preference_saved_message(memory: dict[str, Any]) -> str:
    """Confirm a successful preference update."""

    summary = build_memory_summary(memory)
    return "Mình đã lưu lại sở thích này. " + summary


def build_recommendation_query(memory: dict[str, Any]) -> str:
    """Build a retrieval query from stored user preferences."""

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
    """Build a non-retrieval fallback recommendation message."""

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
    """Build detailed recommendations from retrieved dataset chunks."""

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

        recommendation = "\n".join(recommendation_lines)
        recommendations.append(recommendation)

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
    """Extract a short reason from a retrieved chunk."""

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
    """Describe the user's stored interests in one short phrase."""

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
    """Explain why a retrieved chunk fits the user's stored preference."""

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
    """Return the first matching labelled section from a chunk."""

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
    """Make a dataset excerpt readable inside a recommendation list."""

    compacted_text = re.sub(r"\s+", " ", str(text)).strip()
    if len(compacted_text) <= max_chars:
        return compacted_text

    return compacted_text[: max_chars - 3].rstrip() + "..."


def prettify_topic_for_display(topic: str) -> str:
    """
    Make legacy keyword-style topics easier to read.

    This function is deliberately small and conservative. It only fixes common
    dataset terms for display, while retrieval still uses the original metadata.
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

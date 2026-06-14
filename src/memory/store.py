"""
Các hàm xử lý long-term memory của user.

Mục đích file:
Lưu những sở thích user nói rõ, ví dụ "Tôi thích lễ hội", tách biệt với câu hỏi
RAG thông thường, ví dụ "Lễ hội này là gì?".

Ý nghĩa các field memory:
- categories: nhóm văn hóa chuẩn của dataset, ví dụ `le_hoi`, `am_thuc`.
- topics: chủ đề user nói bằng ngôn ngữ tự nhiên, ví dụ `lễ hội`, `ẩm thực`.
- keywords: hiện đang giống topics, giữ lại để tương thích code cũ.
- normalized_keywords: bản bỏ dấu của topics, dùng để match/retrieve.
- question_styles: để mở rộng sau, ví dụ user thích hỏi "so sánh" hay "nguồn gốc".
- evidence: tối đa 10 câu gốc chứng minh vì sao memory được lưu.
- last_updated: thời điểm cập nhật gần nhất.

Flow ghi memory:
preference text -> extract_memory_updates_from_text()
-> current memory + updates -> merge_memory()
-> memory file + user_id -> save_user_memory()

Flow đọc memory:
memory file + user_id -> load_user_memory() -> build_memory_summary()
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.routing.intent_router import CATEGORY_PATTERNS
from src.routing.text_utils import normalize_text, unique_values


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


def load_memory_json(memory_text: str | dict[str, Any] | None) -> dict[str, Any]:
    """
    Parse memory từ JSON string/dict và bổ sung field còn thiếu.

    Biến đầu vào:
    - memory_text: có thể là JSON string trong GraphState hoặc dict đã parse.

    Ví dụ output:
    {"categories": [], "topics": [], "keywords": [], "last_updated": ""}

    Cách tự viết lại:
    Parse JSON an toàn, merge với DEFAULT_MEMORY, rồi đảm bảo các field dạng list
    thật sự là list để các hàm sau không bị lỗi.
    """

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
    """
    Chuyển memory dict thành JSON string để lưu trong GraphState.

    Ví dụ output:
    '{"categories": ["le_hoi"], "topics": ["lễ hội"], ...}'

    Cách tự viết lại:
    Dùng `json.dumps(..., ensure_ascii=False)` để giữ tiếng Việt có dấu.
    """

    return json.dumps(memory, ensure_ascii=False)


def build_thread_id(user_id: str, conversation_id: str) -> str:
    """
    Tạo thread id cho short-term conversation của LangGraph.

    Biến đầu vào:
    - user_id: định danh user cho long-term memory.
    - conversation_id: định danh cuộc trò chuyện hiện tại.

    Ví dụ output:
    build_thread_id("demo_user_a", "demo_thread")
    -> "user:demo_user_a:thread:demo_thread"

    Cách tự viết lại:
    Normalize cả user_id và conversation_id, thay khoảng trắng bằng `_`, rồi
    ghép thành một string ổn định.
    """

    safe_user_id = normalize_text(user_id).replace(" ", "_") or "anonymous"
    safe_conversation_id = normalize_text(conversation_id).replace(" ", "_") or "default"
    return f"user:{safe_user_id}:thread:{safe_conversation_id}"


def build_langgraph_config(user_id: str, conversation_id: str) -> dict[str, dict[str, str]]:
    """
    Tạo config object mà LangGraph checkpointer cần.

    Ví dụ output:
    {"configurable": {"thread_id": "user:demo_user_a:thread:demo_thread"}}

    Cách tự viết lại:
    LangGraph cần `configurable.thread_id`, nên chỉ cần bọc build_thread_id()
    theo đúng cấu trúc dict này.
    """

    return {
        "configurable": {
            "thread_id": build_thread_id(
                user_id=user_id,
                conversation_id=conversation_id,
            )
        }
    }


def extract_memory_updates_from_text(user_message: str) -> dict[str, list[str]]:
    """
    Trích xuất phần memory update từ một câu nói sở thích.

    Biến đầu vào:
    - user_message: câu gốc, ví dụ "Tôi thích lễ hội và ẩm thực".

    Ví dụ output:
    {
        "categories": ["am_thuc", "le_hoi"],
        "topics": ["lễ hội", "ẩm thực"],
        "keywords": ["lễ hội", "ẩm thực"],
        "normalized_keywords": ["le hoi", "am thuc"]
    }

    Cách tự viết lại:
    Normalize câu để detect category, đồng thời cắt phần sau marker "tôi thích"
    để lấy topic tự nhiên user nói.
    """

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
    """
    Map text đã normalize sang category id trong dataset.

    Ví dụ output:
    detect_categories("toi thich le hoi va am thuc")
    -> ["am_thuc", "le_hoi"]

    Cách tự viết lại:
    Duyệt CATEGORY_PATTERNS, category nào có ít nhất một pattern xuất hiện trong
    câu thì thêm category đó vào kết quả.
    """

    detected_categories: list[str] = []

    for category_id, patterns in CATEGORY_PATTERNS.items():
        if any(pattern in normalized_text for pattern in patterns):
            detected_categories.append(category_id)

    return detected_categories


def extract_topics_from_preference_text(user_message: str) -> list[str]:
    """
    Lấy chủ đề nằm sau marker sở thích.

    Ví dụ output:
    extract_topics_from_preference_text("Tôi thích lễ hội và ẩm thực")
    -> ["lễ hội", "ẩm thực"]

    Cách tự viết lại:
    Dùng regex xóa prefix như "tôi thích", sau đó tách phần còn lại bằng dấu
    phẩy, "và", "với", hoặc dấu `/`.
    """

    text = str(user_message).strip()
    marker_pattern = re.compile(
        r"(tôi|toi|mình|minh|t)\s+"
        r"(thích|thich|mê|me|hứng thú|hung thu|quan tâm|quan tam|muốn tìm hiểu về|muon tim hieu ve)\s+",
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
    """
    Gộp memory cũ với phần update mới.

    Biến đầu vào:
    - current_memory: memory hiện tại của user.
    - memory_updates: dict vừa trích xuất từ câu sở thích.
    - evidence_text: câu gốc dùng làm bằng chứng.

    Ví dụ output:
    memory cũ topics=[] + update topics=["lễ hội"]
    -> topics=["lễ hội"], evidence=["Tôi thích lễ hội"]

    Cách tự viết lại:
    Với từng list field, nối list cũ + list mới rồi deduplicate. Evidence nên
    giới hạn số lượng để file memory không phình quá nhanh.
    """

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
    """
    Đọc memory của một user từ file JSON.

    Ví dụ output:
    load_user_memory("user_memories.json", "demo_user_a")
    -> {"categories": ["le_hoi"], "topics": ["lễ hội"], ...}

    Cách tự viết lại:
    Mở file JSON tổng, lấy object theo user_id, rồi đưa qua load_memory_json()
    để fill field thiếu.
    """

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
    """
    Ghi memory của một user vào file JSON tổng.

    Ví dụ output trong file:
    {
      "demo_user_a": {"categories": ["le_hoi"], "topics": ["lễ hội"]}
    }

    Cách tự viết lại:
    Đọc toàn bộ memory_db, cập nhật key user_id, rồi ghi lại bằng UTF-8 và
    `ensure_ascii=False`.
    """

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
    """
    Cập nhật memory bằng metadata của documents retrieve được.

    Lưu ý:
    Hàm này giữ lại để thử nghiệm, nhưng graph chính không gọi. Lý do: user hỏi
    về một chủ đề không có nghĩa là user thích chủ đề đó.

    Ví dụ output:
    documents có metadata category="giao_thong"
    -> memory_updates["categories"] có "giao_thong"

    Cách tự viết lại:
    Duyệt documents, lấy category/keyword từ metadata, gom thành memory_updates,
    rồi gọi merge_memory().
    """

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


def build_memory_summary(memory: dict[str, Any]) -> str:
    """
    Tạo câu trả lời cho câu hỏi "Tôi thích gì?".

    Ví dụ output:
    "Mình đang nhớ bạn quan tâm đến nhóm chủ đề: lễ hội; từ khóa/chủ đề: Tết."

    Cách tự viết lại:
    Chuyển category id sang nhãn tiếng Việt, ghép với topics, nếu chưa có gì thì
    trả fallback nói rằng chưa lưu sở thích rõ ràng.
    """

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
    """
    Tạo câu xác nhận sau khi lưu sở thích.

    Ví dụ output:
    "Mình đã lưu lại sở thích này. Mình đang nhớ bạn quan tâm đến..."

    Cách tự viết lại:
    Gọi build_memory_summary() rồi thêm prefix xác nhận đã lưu.
    """

    summary = build_memory_summary(memory)
    return "Mình đã lưu lại sở thích này. " + summary

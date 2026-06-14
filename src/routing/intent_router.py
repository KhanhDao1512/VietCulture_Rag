"""
Router rule-based cho trợ lý VietCulture.

Mục đích file:
Phân loại tin nhắn mới nhất của user trước khi agent quyết định có cần RAG,
cập nhật memory, đọc memory, gợi ý chủ đề, hay chỉ trả lời xã giao.

Luồng xử lý:
classify_intent(message)
-> normalize_text(message)
-> check memory_query
-> check preference_update
-> check recommendation_request
-> check followup_question
-> check dataset/rag_question
-> check chitchat
-> fallback out_of_scope

Cách tự viết lại file này:
1. Định nghĩa danh sách intent hệ thống hỗ trợ.
2. Viết các hàm boolean nhỏ như `is_memory_query()`.
3. Trong `classify_intent()`, kiểm tra intent theo thứ tự ưu tiên.
4. Trả về cả intent và reason để debug trên UI.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.routing.text_utils import normalize_text


SUPPORTED_INTENTS = {
    "rag_question",
    "followup_question",
    "recommendation_request",
    "preference_update",
    "memory_query",
    "chitchat",
    "out_of_scope",
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


@dataclass(frozen=True)
class IntentDecision:
    """
    Kết quả phân loại intent.

    Biến:
    - intent: nhãn route, ví dụ `rag_question` hoặc `memory_query`.
    - reason: giải thích ngắn để hiển thị trong debug UI.
    - confidence: độ chắc chắn của rule router, từ 0.0 đến 1.0.
    - source: router tạo ra quyết định, ví dụ `rule` hoặc `llm`.
    - memory_update_allowed: có cho phép node sau cập nhật long-term memory không.

    Ví dụ output:
    IntentDecision(intent="preference_update", reason="...", confidence=0.95)

    Cách tự viết lại:
    Dùng dataclass nhỏ, chỉ cần chứa nhãn intent và lý do. Không nên trả string
    intent trần vì sẽ mất thông tin debug.
    """

    intent: str
    reason: str
    confidence: float = 1.0
    source: str = "rule"
    memory_update_allowed: bool = False


def classify_intent(user_message: str) -> IntentDecision:
    """
    Phân loại một tin nhắn user thành một route duy nhất.

    Biến đầu vào:
    - user_message: câu user vừa nhập, còn nguyên dấu tiếng Việt.

    Ví dụ output:
    classify_intent("Tôi thích lễ hội")
    -> IntentDecision(intent="preference_update", reason="...")

    Cách tự viết lại:
    Normalize text trước, rồi gọi từng rule boolean theo thứ tự ưu tiên. Các
    intent liên quan memory nên check trước RAG để tránh câu "Tôi thích gì?"
    bị hiểu nhầm là câu hỏi kiến thức.
    """

    normalized_message = normalize_text(user_message)

    if not normalized_message:
        return IntentDecision(
            "chitchat",
            "Tin nhắn rỗng hoặc quá ngắn.",
            confidence=0.95,
        )

    if is_memory_query(normalized_message):
        return IntentDecision(
            "memory_query",
            "Người dùng hỏi lại thông tin đã lưu trong memory.",
            confidence=0.95,
        )

    if is_preference_update(normalized_message):
        return IntentDecision(
            "preference_update",
            "Người dùng đang nói về sở thích.",
            confidence=0.95,
            memory_update_allowed=True,
        )

    if is_recommendation_request(normalized_message):
        return IntentDecision(
            "recommendation_request",
            "Người dùng muốn được gợi ý.",
            confidence=0.92,
        )

    if is_followup_question(normalized_message):
        return IntentDecision(
            "followup_question",
            "Câu hỏi phụ thuộc vào ngữ cảnh hội thoại trước.",
            confidence=0.72,
        )

    if is_dataset_question(normalized_message):
        confidence = 0.78 if has_category_hint(normalized_message) else 0.55
        return IntentDecision(
            "rag_question",
            "Câu hỏi phù hợp với miền dữ liệu văn hóa Việt Nam.",
            confidence=confidence,
        )

    if is_chitchat(normalized_message):
        return IntentDecision(
            "chitchat",
            "Tin nhắn xã giao không cần retrieval.",
            confidence=0.93,
        )

    return IntentDecision(
        "out_of_scope",
        "Không nhận ra câu hỏi thuộc phạm vi dataset.",
        confidence=0.7,
    )


def is_preference_update(normalized_message: str) -> bool:
    """
    Kiểm tra user có đang nói rõ sở thích không.

    Biến đầu vào:
    - normalized_message: text đã bỏ dấu/lowercase, ví dụ `toi thich le hoi`.

    Ví dụ output:
    is_preference_update("toi thich le hoi") -> True

    Cách tự viết lại:
    Liệt kê các marker như `toi thich`, `minh quan tam`, sau đó kiểm tra marker
    có xuất hiện trong câu không.
    """

    preference_markers = [
        "toi thich",
        "minh thich",
        "t thich",
        "toi me",
        "minh me",
        "toi hung thu",
        "minh hung thu",
        "toi quan tam",
        "minh quan tam",
        "toi muon tim hieu ve",
        "minh muon tim hieu ve",
    ]
    return any(marker in normalized_message for marker in preference_markers)


def is_memory_query(normalized_message: str) -> bool:
    """
    Kiểm tra user có hỏi assistant đang nhớ gì về họ không.

    Ví dụ output:
    is_memory_query("toi thich gi") -> True

    Cách tự viết lại:
    Tạo danh sách câu hỏi memory phổ biến như `toi thich gi`, `ban nho gi ve
    toi`, rồi dùng `any(marker in normalized_message ...)`.
    """

    memory_markers = [
        "toi thich gi",
        "ban nho gi ve toi",
        "ban biet gi ve toi",
        "so thich cua toi",
        "toi quan tam gi",
    ]
    return any(marker in normalized_message for marker in memory_markers)


def is_recommendation_request(normalized_message: str) -> bool:
    """
    Kiểm tra user có đang yêu cầu gợi ý/chủ đề nên học không.

    Ví dụ output:
    is_recommendation_request("goi y cho toi vai chu de") -> True

    Cách tự viết lại:
    Gom các cụm như `goi y`, `de xuat`, `nen tim hieu`, `chu de hay`. Nếu muốn
    bắt tiếng Anh thì thêm `recommend`.
    """

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
    """
    Kiểm tra câu hỏi phụ thuộc ngữ cảnh trước đó.

    Ví dụ output:
    is_followup_question("mon nay co y nghia gi") -> True

    Cách tự viết lại:
    Một follow-up thường cần cả marker tham chiếu (`mon nay`, `cai nay`) và dạng
    câu hỏi (`la gi`, `y nghia`). Ghép hai điều kiện này để giảm false positive.
    """

    has_followup_marker = any(marker in normalized_message for marker in FOLLOWUP_MARKERS)
    has_question_shape = any(word in normalized_message for word in QUESTION_WORDS)
    return has_followup_marker and has_question_shape


def is_dataset_question(normalized_message: str) -> bool:
    """
    Kiểm tra câu có vẻ là câu hỏi thuộc miền dữ liệu văn hóa Việt Nam không.

    Ví dụ output:
    is_dataset_question("xe may la gi") -> True

    Cách tự viết lại:
    Check hai tín hiệu: câu có dạng hỏi, hoặc chứa keyword/category trong
    dataset. Đây là rule khá rộng, nên về sau có thể cho LLM router xử lý các
    case mơ hồ.
    """

    has_question_shape = "?" in normalized_message or any(
        word in normalized_message
        for word in QUESTION_WORDS
    )
    return has_question_shape or has_category_hint(normalized_message)


def has_category_hint(normalized_message: str) -> bool:
    """
    Kiểm tra câu có chứa keyword/category văn hóa trong dataset không.

    Ví dụ output:
    has_category_hint("toi thich le hoi") -> True

    Cách tự viết lại:
    Duyệt toàn bộ CATEGORY_PATTERNS, nếu bất kỳ pattern nào nằm trong câu đã
    normalize thì coi như có tín hiệu thuộc miền VietCulture.
    """

    return any(
        pattern in normalized_message
        for patterns in CATEGORY_PATTERNS.values()
        for pattern in patterns
    )


def is_chitchat(normalized_message: str) -> bool:
    """
    Kiểm tra tin nhắn xã giao không cần retrieval.

    Ví dụ output:
    is_chitchat("chao ban") -> True

    Cách tự viết lại:
    Tạo whitelist các câu ngắn như `chao`, `hello`, `cam on`, `ok`. Với chitchat
    nên bảo thủ để tránh nuốt nhầm câu hỏi thật.
    """

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

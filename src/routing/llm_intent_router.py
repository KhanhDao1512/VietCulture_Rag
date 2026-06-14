"""
LLM fallback router cho các intent mơ hồ.

Mục đích file:
Rule router vẫn là lớp đầu tiên vì nhanh, rẻ và dễ debug. File này chỉ dùng LLM
để phân loại lại khi rule router chưa đủ tự tin, ví dụ câu nói tự nhiên như
"Mình mê mấy thứ truyền thống kiểu hội hè ấy".

Luồng xử lý:
classify_intent_hybrid()
-> classify_intent() bằng rule
-> nếu confidence >= threshold hoặc không có LLM thì dùng rule result
-> nếu confidence thấp thì gọi classify_intent_with_llm()
-> validate intent LLM trả về
-> trả IntentDecision source="llm"

Ghi chú quan trọng:
LLM router chỉ quyết định route. Nó không tự update memory. Việc lưu memory vẫn
phải đi qua node graph và chỉ được phép khi `memory_update_allowed=True`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.routing.intent_router import SUPPORTED_INTENTS, IntentDecision, classify_intent


class LlmIntentDecision(BaseModel):
    """
    Schema output bắt buộc của LLM intent router.

    Biến:
    - intent: một trong các intent hệ thống hỗ trợ.
    - confidence: độ chắc chắn LLM tự đánh giá, từ 0.0 đến 1.0.
    - reason: giải thích ngắn bằng tiếng Việt.
    - memory_update_allowed: chỉ True khi user nói rõ sở thích.

    Ví dụ output:
    {
        "intent": "preference_update",
        "confidence": 0.86,
        "reason": "User nói thích các chủ đề hội hè truyền thống.",
        "memory_update_allowed": true
    }

    Cách tự viết lại:
    Tạo Pydantic schema nhỏ, ép LLM trả structured output. Các field nên rõ ràng
    và ít để LLM không bị phân tâm.
    """

    intent: str = Field(description="One supported intent label.")
    confidence: float = Field(description="Confidence from 0.0 to 1.0.")
    reason: str = Field(description="Short Vietnamese explanation.")
    memory_update_allowed: bool = Field(
        description="True only when the user explicitly states a preference."
    )


def classify_intent_with_llm(
    user_message: str,
    rule_decision: IntentDecision,
    llm: Any,
) -> IntentDecision:
    """
    Gọi LLM để phân loại intent khi rule router chưa đủ chắc.

    Biến đầu vào:
    - user_message: tin nhắn gốc của user.
    - rule_decision: kết quả rule router, dùng làm gợi ý/debug cho LLM.
    - llm: Chat model có hỗ trợ `with_structured_output`.

    Ví dụ output:
    IntentDecision(intent="recommendation_request", source="llm", confidence=0.82)

    Cách tự viết lại:
    Viết prompt liệt kê intent hợp lệ, nhấn mạnh luật memory, truyền cả kết quả
    rule hiện tại, rồi ép LLM trả LlmIntentDecision.
    """

    structured_llm = llm.with_structured_output(LlmIntentDecision)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là intent router cho trợ lý VietCulture.
Chọn đúng một intent trong danh sách:
- rag_question: câu hỏi kiến thức về văn hóa Việt Nam cần retrieval/RAG.
- followup_question: câu hỏi phụ thuộc ngữ cảnh trước đó như "cái này thì sao".
- recommendation_request: user muốn được gợi ý chủ đề/nội dung.
- preference_update: user nói rõ sở thích/cái họ quan tâm.
- memory_query: user hỏi assistant đang nhớ gì về họ.
- chitchat: xã giao ngắn, không cần RAG.
- out_of_scope: ngoài phạm vi văn hóa Việt Nam/dataset.

Luật memory:
- Chỉ đặt memory_update_allowed=true nếu user nói rõ sở thích, ví dụ "tôi thích lễ hội".
- Câu hỏi kiến thức như "xe máy là gì" không được coi là sở thích.
- Trả reason ngắn bằng tiếng Việt.""",
            ),
            (
                "human",
                """Tin nhắn user:
{user_message}

Kết quả rule router hiện tại:
intent={rule_intent}
confidence={rule_confidence}
reason={rule_reason}

Hãy phân loại lại nếu cần.""",
            ),
        ]
    )
    chain = prompt | structured_llm
    result = chain.invoke(
        {
            "user_message": user_message,
            "rule_intent": rule_decision.intent,
            "rule_confidence": rule_decision.confidence,
            "rule_reason": rule_decision.reason,
        }
    )

    intent = result.intent if result.intent in SUPPORTED_INTENTS else rule_decision.intent
    confidence = max(0.0, min(1.0, float(result.confidence)))
    memory_update_allowed = bool(result.memory_update_allowed)
    if intent != "preference_update":
        memory_update_allowed = False

    return IntentDecision(
        intent=intent,
        reason=f"LLM router: {result.reason}",
        confidence=confidence,
        source="llm",
        memory_update_allowed=memory_update_allowed,
    )


def classify_intent_hybrid(
    user_message: str,
    llm: Any | None = None,
    use_llm_fallback: bool = False,
    confidence_threshold: float = 0.75,
) -> IntentDecision:
    """
    Router hybrid: rule trước, LLM fallback sau.

    Biến đầu vào:
    - user_message: tin nhắn user.
    - llm: model LLM dùng cho fallback, có thể None.
    - use_llm_fallback: flag bật/tắt LLM router.
    - confidence_threshold: dưới ngưỡng này mới gọi LLM.

    Ví dụ output:
    classify_intent_hybrid("Mình mê hội hè truyền thống", llm, True)
    -> IntentDecision(intent="preference_update", source="llm", ...)

    Cách tự viết lại:
    Luôn gọi rule router trước. Nếu rule đã chắc, trả ngay. Nếu rule mơ hồ và
    có LLM thì gọi classify_intent_with_llm(). Nếu LLM lỗi thì fallback về rule.
    """

    rule_decision = classify_intent(user_message)
    if (
        not use_llm_fallback
        or llm is None
        or rule_decision.confidence >= confidence_threshold
    ):
        return rule_decision

    try:
        return classify_intent_with_llm(
            user_message=user_message,
            rule_decision=rule_decision,
            llm=llm,
        )
    except Exception as error:
        return IntentDecision(
            intent=rule_decision.intent,
            reason=f"{rule_decision.reason} LLM fallback lỗi: {error}",
            confidence=rule_decision.confidence,
            source="rule_fallback",
            memory_update_allowed=rule_decision.memory_update_allowed,
        )

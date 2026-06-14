"""
Lớp tương thích cho module routing cũ.

Mục đích file:
Code cũ đang import nhiều helper từ `src.routing.routing`. Phần implementation
đã được tách sang các module nhỏ hơn, nhưng file này re-export API cũ để
graph.py, notebook và các script cũ vẫn chạy.

Cấu trúc mới:
src.routing.text_utils         -> normalize_text(), unique_values()
src.routing.intent_router      -> classify_intent() và các rule routing
src.memory.store               -> memory JSON, thread id, trích xuất sở thích
src.recommendation.recommender -> format recommendation có căn cứ từ dataset
"""

from __future__ import annotations

from src.memory.store import (
    DATASET_CATEGORY_LABELS,
    DEFAULT_MEMORY,
    build_langgraph_config,
    build_memory_summary,
    build_preference_saved_message,
    build_thread_id,
    detect_categories,
    dump_memory_json,
    extract_memory_updates_from_text,
    extract_topics_from_preference_text,
    load_memory_json,
    load_user_memory,
    merge_memory,
    save_user_memory,
    update_memory_from_retrieved_documents,
)
from src.recommendation.recommender import (
    build_grounded_recommendation_message,
    build_recommendation_fit_reason,
    build_recommendation_message,
    build_recommendation_queries,
    build_recommendation_query,
    build_user_interest_summary,
    compact_text,
    extract_content_section,
    extract_recommendation_reason,
    prettify_topic_for_display,
    rank_recommendation_candidates,
)
from src.routing.intent_router import (
    CATEGORY_PATTERNS,
    FOLLOWUP_MARKERS,
    QUESTION_WORDS,
    SUPPORTED_INTENTS,
    IntentDecision,
    classify_intent,
    has_category_hint,
    is_chitchat,
    is_dataset_question,
    is_followup_question,
    is_memory_query,
    is_preference_update,
    is_recommendation_request,
)
from src.routing.llm_intent_router import (
    LlmIntentDecision,
    classify_intent_hybrid,
    classify_intent_with_llm,
)
from src.routing.text_utils import normalize_text, unique_values

__all__ = [
    "CATEGORY_PATTERNS",
    "DATASET_CATEGORY_LABELS",
    "DEFAULT_MEMORY",
    "FOLLOWUP_MARKERS",
    "IntentDecision",
    "LlmIntentDecision",
    "QUESTION_WORDS",
    "SUPPORTED_INTENTS",
    "build_grounded_recommendation_message",
    "build_langgraph_config",
    "build_memory_summary",
    "build_preference_saved_message",
    "build_recommendation_fit_reason",
    "build_recommendation_message",
    "build_recommendation_queries",
    "build_recommendation_query",
    "build_thread_id",
    "build_user_interest_summary",
    "classify_intent",
    "classify_intent_hybrid",
    "classify_intent_with_llm",
    "compact_text",
    "detect_categories",
    "dump_memory_json",
    "extract_content_section",
    "extract_memory_updates_from_text",
    "extract_recommendation_reason",
    "extract_topics_from_preference_text",
    "has_category_hint",
    "is_chitchat",
    "is_dataset_question",
    "is_followup_question",
    "is_memory_query",
    "is_preference_update",
    "is_recommendation_request",
    "load_memory_json",
    "load_user_memory",
    "merge_memory",
    "normalize_text",
    "prettify_topic_for_display",
    "rank_recommendation_candidates",
    "save_user_memory",
    "unique_values",
    "update_memory_from_retrieved_documents",
]

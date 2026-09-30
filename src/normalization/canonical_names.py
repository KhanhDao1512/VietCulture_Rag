"""Chuẩn hóa tên topic trước khi đưa vào câu trả lời.

Mục đích file:
Giữ tên nguồn từ dataset, tra alias xác định trước, rồi dùng LLM làm fallback
để chuyển tên không dấu/tiếng Anh sang cách hiển thị tiếng Việt dễ đọc.

Luồng xử lý:
collect_topic_names(documents)
-> apply_known_aliases()
-> ask_llm_to_normalize_unknown_names()
-> validate_proposals()
-> trả mapping tên nguồn -> tên hiển thị

Ghi chú quan trọng:
- LLM chỉ chuẩn hóa nhãn hiển thị, không bổ sung sự kiện văn hóa.
- Nếu model không chắc hoặc trả tên ngoài candidate list, giữ nguyên tên nguồn.
- Tên nguồn luôn được giữ trong mapping để truy vết.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from src.memory.store import DATASET_CATEGORY_LABELS
from src.routing.text_utils import normalize_text


KNOWN_TOPIC_ALIASES = {
    "banh chung": "Bánh chưng",
    "banh chung tet": "Bánh chưng Tết",
    "bun cha ha noi": "Bún chả Hà Nội",
    "mam ruoc": "Mắm ruốc",
    "che viet nam": "Chè Việt Nam",
    "ram thang gieng": "Rằm tháng Giêng",
    "nem con": "Ném còn",
    "tro choi dan gian viet nam": "Trò chơi dân gian Việt Nam",
}


class NameMapping(BaseModel):
    """Một tên nguồn và cách hiển thị tiếng Việt do model đề xuất."""

    source_name: str = Field(description="Tên gốc, phải trùng với candidate được cung cấp.")
    display_name: str = Field(description="Tên tiếng Việt ngắn, tự nhiên để hiển thị.")
    confidence: float = Field(ge=0.0, le=1.0, description="Độ tin cậy từ 0 đến 1.")
    is_valid_topic: bool = Field(
        default=True,
        description="Tên này có nghĩa rõ ràng và được dữ liệu hỗ trợ hay không.",
    )


class NameMappingBatch(BaseModel):
    """Danh sách mapping có cấu trúc để xác thực trước khi dùng."""

    mappings: list[NameMapping]


def collect_topic_names(documents: list[Any], limit: int = 8) -> list[str]:
    """Thu thập các tên topic duy nhất từ metadata của retrieved documents.

    Biến đầu vào:
    - documents: LangChain Documents có metadata topic/keyword.
    - limit: số tên tối đa gửi cho bước chuẩn hóa.

    Ví dụ output:
    metadata topic="Bun cha Ha Noi" -> ["Bun cha Ha Noi"]

    Cách tự viết lại:
    Duyệt metadata theo thứ tự ưu tiên canonical_topic, topic, keyword; dùng
    normalize_text làm khóa trùng nhưng giữ lại chuỗi nguồn đầu tiên.
    """

    names: list[str] = []
    seen: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        for field_name in ("canonical_topic", "topic", "retrieval_anchor", "keyword"):
            source_name = str(metadata.get(field_name) or "").strip()
            normalized_name = normalize_text(source_name)
            if not source_name or not normalized_name or normalized_name in seen:
                continue
            seen.add(normalized_name)
            names.append(source_name)
            break
        if len(names) >= limit:
            break
    return names


def collect_topic_contexts(documents: list[Any]) -> dict[str, str]:
    """Gom category, image folder và QA evidence làm ngữ cảnh chuẩn hóa tên."""

    contexts: dict[str, str] = {}
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        source_name = next(
            (
                str(metadata.get(field_name) or "").strip()
                for field_name in ("canonical_topic", "topic", "retrieval_anchor", "keyword")
                if str(metadata.get(field_name) or "").strip()
            ),
            "",
        )
        if not source_name:
            continue

        evidence = " ".join(str(getattr(document, "page_content", "")).split())[:900]
        category_id = str(metadata.get("category", ""))
        category_label = DATASET_CATEGORY_LABELS.get(category_id, category_id)
        details = [
            f"category={category_label}",
            f"image_path={metadata.get('image_path', '')}",
            f"source_question={metadata.get('question', '')}",
            f"dataset_evidence={evidence}",
        ]
        contexts[source_name] = "\n".join(details)[:1400]
    return contexts


def has_vietnamese_diacritics(value: str) -> bool:
    """Kiểm tra tên có dấu hiệu chữ Việt; nhãn ASCII không được xuất trực tiếp."""

    return bool(re.search(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", value.casefold()))


def needs_llm_normalization(source_name: str) -> bool:
    """Chỉ gửi các tên nhiều từ dạng ASCII chưa có alias đã biết sang LLM."""

    normalized_name = normalize_text(source_name)
    words = normalized_name.split()
    if normalized_name in KNOWN_TOPIC_ALIASES:
        return False
    if len(words) < 2 or "_" in source_name:
        return False
    return all(character.isascii() for character in source_name)


def validate_name_mappings(
    candidates: list[str],
    proposals: list[NameMapping],
    minimum_confidence: float = 0.82,
) -> dict[str, str]:
    """Chỉ nhận đề xuất có nguồn hợp lệ, confidence đủ cao và tên ngắn.

    Biến đầu vào:
    - candidates: tên lấy trực tiếp từ retrieved metadata.
    - proposals: output đã parse theo NameMapping.
    - minimum_confidence: ngưỡng chấp nhận đề xuất.

    Ví dụ output:
    source="Bun cha Ha Noi", proposal="Bún chả Hà Nội" -> mapping được nhận.

    Cách tự viết lại:
    Tạo lookup candidate theo dạng normalize, xác thực source/display/confidence,
    sau đó trả mapping chỉ cho các đề xuất hợp lệ.
    """

    candidate_lookup = {normalize_text(name): name for name in candidates}
    accepted: dict[str, str] = {}
    for proposal in proposals:
        source_key = normalize_text(proposal.source_name)
        source_name = candidate_lookup.get(source_key)
        display_name = proposal.display_name.strip()
        if (
            not source_name
            or not proposal.is_valid_topic
            or not display_name
            or proposal.confidence < minimum_confidence
            or not has_vietnamese_diacritics(display_name)
        ):
            continue
        if len(display_name) > 100 or "\n" in display_name or "\r" in display_name:
            continue
        accepted[source_name] = display_name
    return accepted


def normalize_topic_names(
    candidates: list[str],
    llm: Any | None,
    minimum_confidence: float = 0.82,
    contexts: dict[str, str] | None = None,
    normalize_all: bool = False,
) -> dict[str, str]:
    """Chuẩn hóa topic bằng alias cục bộ trước, LLM fallback sau.

    Hàm không gửi request nếu không có candidate cần chuẩn hóa hoặc LLM chưa
    được cấu hình. Lỗi API/schema không làm hỏng RAG; tên gốc được giữ lại.
    """

    mappings: dict[str, str] = {}
    for source_name in candidates:
        alias = KNOWN_TOPIC_ALIASES.get(normalize_text(source_name))
        if alias:
            mappings[source_name] = alias

    unknown_names = [
        name
        for name in candidates
        if normalize_text(name) not in KNOWN_TOPIC_ALIASES
        and (normalize_all or needs_llm_normalization(name))
    ]
    if llm is None or not unknown_names:
        return mappings

    prompt = (
        "Bạn chuẩn hóa nhãn chủ đề cho một chatbot văn hóa Việt Nam.\n"
        "Với mỗi candidate, dùng category, image_path và QA evidence để hiểu đúng chủ đề.\n"
        "Trả tên ngắn, tự nhiên bằng tiếng Việt có dấu; dịch các nhãn tiếng Anh.\n"
        "Sửa lỗi dấu/viết hoa khi có căn cứ. Không tự thêm địa danh hoặc sự kiện.\n"
        "Nếu tên không rõ nghĩa, sai lệch với evidence, hoặc không thể xác định chắc, "
        "đặt is_valid_topic=false. Không đoán một chủ đề khác chỉ từ ảnh.\n"
        "source_name phải giữ nguyên một candidate đã cho.\n"
        "Candidates (JSON): "
        + json.dumps(
            [
                {"source_name": name, "context": (contexts or {}).get(name, "")}
                for name in unknown_names
            ],
            ensure_ascii=False,
        )
    )

    try:
        structured_llm = llm.with_structured_output(NameMappingBatch)
        result = structured_llm.invoke(prompt)
        mappings.update(
            validate_name_mappings(
                candidates=unknown_names,
                proposals=result.mappings,
                minimum_confidence=minimum_confidence,
            )
        )
    except Exception:
        # Chuẩn hóa chỉ ảnh hưởng cách viết; lỗi model không được chặn câu trả lời.
        return mappings

    return mappings

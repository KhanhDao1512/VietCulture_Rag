"""
Helper xử lý text dùng chung cho routing, memory và recommendation.

Mục đích file:
Tập trung các thao tác text nhỏ nhưng dùng nhiều nơi, tránh copy/paste logic
normalize và deduplicate.

Luồng xử lý:
raw text -> normalize_text() -> rule matching / deduplication
list values -> unique_values() -> field memory ổn định, không bị trùng
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def normalize_text(text: Any) -> str:
    """
    Normalize tiếng Việt để match keyword không phụ thuộc dấu.

    Biến đầu vào:
    - text: chuỗi bất kỳ, có thể có dấu tiếng Việt hoặc ký tự đặc biệt.

    Ví dụ output:
    normalize_text("Tôi thích Lễ hội!") -> "toi thich le hoi"

    Cách tự viết lại:
    Chuyển text về lowercase, tách Unicode bằng NFD, bỏ các dấu thanh, rồi thay
    ký tự không phải chữ/số bằng khoảng trắng.
    """

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
    """
    Xóa giá trị rỗng/trùng nhưng vẫn giữ thứ tự xuất hiện ban đầu.

    Ví dụ output:
    unique_values(["lễ hội", "", "ẩm thực", "lễ hội"]) -> ["lễ hội", "ẩm thực"]

    Cách tự viết lại:
    Dùng set để nhớ giá trị đã gặp, nhưng append vào list kết quả để giữ order.
    """

    seen_values: set[str] = set()
    clean_values: list[str] = []

    for value in values:
        clean_value = str(value).strip()
        if not clean_value or clean_value in seen_values:
            continue

        seen_values.add(clean_value)
        clean_values.append(clean_value)

    return clean_values

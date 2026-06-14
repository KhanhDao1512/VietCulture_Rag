"""
Giao diện Streamlit cho VietCulture memory agent.

Mục đích file:
Tạo trải nghiệm 2 bước cho người dùng:
1. Welcome screen: giới thiệu VietCulture, preview chat, thử render ảnh từ Hugging Face.
2. Chat screen: trò chuyện với agent RAG + memory + recommendation.

Luồng xử lý:
if __name__ == "__main__":
    main()
-> inject_global_styles()
-> nếu chưa bắt đầu: render_welcome_screen()
-> nếu đã bắt đầu: render_chat_screen()
-> user nhập prompt
-> invoke_agent()
-> hiển thị answer + debug state

Ghi chú quan trọng:
- Ảnh nhân vật assistant được lưu trong `assets/assistant_avatar.png`.
- Ảnh bánh bao Hugging Face được dùng ở welcome để test remote image URL.
- Sidebar chỉ xuất hiện trong màn chat chính để welcome nhìn giống landing page hơn.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import streamlit as st

from src.agent.graph import create_agent_bundle, invoke_agent
from src.routing.routing import build_langgraph_config


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSISTANT_AVATAR = ASSETS_DIR / "assistant_avatar.png"
DATASET_JSON_PATH = PROJECT_ROOT / "data" / "vietnamese_vqa_dataset.json"
HF_BANH_BAO_IMAGE_URL = (
    "https://huggingface.co/datasets/Dangindev/viet-cultural-vqa/"
    "resolve/main/images/am_thuc/banh_bao/000001.jpg"
)
HF_DATASET_BASE_URL = "https://huggingface.co/datasets/Dangindev/viet-cultural-vqa/resolve/main"
IMAGE_PATH_PATTERN = re.compile(r'"image_path"\s*:\s*"([^"]+)"')


@st.cache_resource(show_spinner="Đang tải retriever và agent...")
def load_bundle():
    """
    Tạo AgentBundle một lần cho cả process Streamlit.

    Biến đầu vào:
    - Không có input trực tiếp, hàm tự đọc `.env` qua create_agent_bundle().

    Ví dụ output:
    AgentBundle(app=<CompiledGraph>, retriever=QaRetriever(...), settings=...)

    Cách tự viết lại:
    Bọc create_agent_bundle() bằng st.cache_resource để không load Chroma/model
    lại sau mỗi lần Streamlit rerun.
    """

    return create_agent_bundle()


def image_to_data_uri(image_path: Path) -> str:
    """
    Chuyển ảnh local thành data URI để nhúng vào HTML/CSS.

    Biến đầu vào:
    - image_path: đường dẫn file ảnh local trong workspace.

    Ví dụ output:
    "data:image/png;base64,iVBORw0KGgo..."

    Cách tự viết lại:
    Đọc bytes của ảnh, base64 encode, đoán mime type từ suffix, rồi ghép thành
    data URI để browser render được trong st.markdown HTML.
    """

    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    image_bytes = image_path.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def load_memory_db(memory_file: Path) -> dict[str, Any]:
    """
    Đọc toàn bộ memory JSON để hiển thị trong sidebar.

    Biến đầu vào:
    - memory_file: đường dẫn file JSON chứa memory theo user_id.

    Ví dụ output:
    {"demo_user_a": {"categories": ["le_hoi"], "topics": ["lễ hội"]}}

    Cách tự viết lại:
    Nếu file chưa tồn tại thì trả dict rỗng. Nếu JSON lỗi thì cũng trả dict rỗng
    để UI không crash.
    """

    if not memory_file.exists():
        return {}
    try:
        return json.loads(memory_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_memory_db(memory_file: Path, memory_db: dict[str, Any]) -> None:
    """
    Ghi toàn bộ memory database sau khi reset/cleanup.

    Biến đầu vào:
    - memory_file: đường dẫn file JSON memory.
    - memory_db: dict toàn bộ memory của các user.

    Ví dụ output:
    File `user_memories.json` được ghi lại bằng UTF-8 và indent=2.

    Cách tự viết lại:
    Dùng json.dumps(..., ensure_ascii=False, indent=2) rồi write_text UTF-8.
    """

    memory_file.write_text(
        json.dumps(memory_db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reset_user_memory(memory_file: Path, user_id: str) -> None:
    """
    Xóa long-term memory của một user cụ thể.

    Biến đầu vào:
    - memory_file: file JSON memory.
    - user_id: user cần xóa memory.

    Ví dụ output:
    reset_user_memory(file, "demo_user_a") -> key demo_user_a bị remove khỏi JSON.

    Cách tự viết lại:
    Load memory_db, pop user_id, rồi save lại file.
    """

    memory_db = load_memory_db(memory_file)
    memory_db.pop(user_id, None)
    save_memory_db(memory_file, memory_db)


def build_chat_history_key(user_id: str, conversation_id: str) -> str:
    """
    Tạo key riêng cho chat history trong st.session_state.

    Biến đầu vào:
    - user_id: định danh long-term memory.
    - conversation_id: định danh cuộc trò chuyện hiện tại.

    Ví dụ output:
    "chat_history::demo_user_a::demo_thread"

    Cách tự viết lại:
    Ghép user_id và conversation_id để mỗi user/thread có lịch sử UI riêng.
    """

    return f"chat_history::{user_id}::{conversation_id}"


def get_memory_summary(memory: dict[str, Any]) -> tuple[str, str]:
    """
    Tạo text ngắn để hiển thị memory trong sidebar.

    Biến đầu vào:
    - memory: memory dict của user hiện tại.

    Ví dụ output:
    ("lễ hội, ẩm thực", "Tôi thích lễ hội và ẩm thực")

    Cách tự viết lại:
    Lấy topics/categories làm dòng sở thích, lấy evidence gần nhất làm dòng bằng
    chứng. Nếu chưa có gì thì trả fallback dễ hiểu.
    """

    topics = memory.get("topics") or memory.get("keywords") or []
    categories = memory.get("categories") or []
    evidence = memory.get("evidence") or []
    interest_text = ", ".join(topics or categories) if (topics or categories) else "Chưa có sở thích"
    evidence_text = evidence[-1] if evidence else "Chưa có câu nói sở thích nào"
    return interest_text, evidence_text


def inject_global_styles() -> None:
    """
    Inject CSS global để tạo visual style VietCulture.

    Biến đầu vào:
    - Không có input, CSS được đưa trực tiếp vào Streamlit bằng st.markdown.

    Ví dụ output:
    App có nền kem, button xanh, card bo góc và chat preview giống landing page.

    Cách tự viết lại:
    Dùng st.markdown với unsafe_allow_html=True, override `.block-container`,
    button, sidebar và tạo class riêng cho hero/category/chat.
    """

    st.markdown(
        """
        <style>
        :root {
            --vc-green: #155c2f;
            --vc-green-dark: #0f4524;
            --vc-cream: #fbf5e9;
            --vc-cream-2: #fffaf1;
            --vc-ink: #202124;
            --vc-muted: #6b7280;
            --vc-border: rgba(34, 74, 42, 0.14);
            --vc-shadow: 0 22px 55px rgba(57, 43, 20, 0.14);
        }

        .stApp {
            background:
                radial-gradient(circle at 86% 24%, rgba(225, 173, 92, 0.18), transparent 28%),
                linear-gradient(180deg, #fffaf2 0%, #fbf5e9 58%, #fffaf2 100%);
            color: var(--vc-ink);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.35rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: #fffaf2;
            border-right: 1px solid var(--vc-border);
        }

        div[data-testid="stButton"] > button {
            border-radius: 14px;
            border: 1px solid rgba(21, 92, 47, 0.18);
            background: var(--vc-green);
            color: #ffffff;
            min-height: 3rem;
            font-weight: 700;
            box-shadow: 0 14px 28px rgba(21, 92, 47, 0.22);
        }

        div[data-testid="stButton"] > button:hover {
            background: var(--vc-green-dark);
            border-color: var(--vc-green-dark);
            color: #ffffff;
        }

        .vc-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 0.4rem 0 1.35rem;
            border-bottom: 1px solid rgba(55, 65, 81, 0.08);
        }

        .vc-brand {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .vc-logo {
            width: 56px;
            height: 56px;
            border-radius: 18px;
            display: grid;
            place-items: center;
            color: #fff;
            background: radial-gradient(circle at 35% 25%, #d9852c, #9d3426 56%, #155c2f);
            box-shadow: 0 10px 24px rgba(157, 52, 38, 0.25);
            font-family: Georgia, serif;
            font-size: 1.4rem;
            font-weight: 800;
        }

        .vc-brand h1 {
            margin: 0;
            color: var(--vc-green);
            font-size: 2rem;
            line-height: 1;
            font-family: Georgia, "Times New Roman", serif;
        }

        .vc-brand p {
            margin: 0.25rem 0 0;
            color: var(--vc-muted);
            font-size: 0.98rem;
        }

        .vc-nav {
            display: flex;
            align-items: center;
            gap: 2.35rem;
            color: #1f2937;
            font-weight: 650;
        }

        .vc-nav span:first-child {
            color: var(--vc-green);
            border-bottom: 3px solid var(--vc-green);
            padding-bottom: 0.55rem;
        }

        .vc-hero {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(420px, 0.95fr);
            gap: 3.5rem;
            align-items: center;
            padding: 2.9rem 0 2.6rem;
        }

        .vc-eyebrow {
            width: fit-content;
            border: 1px solid rgba(210, 154, 74, 0.32);
            background: rgba(255, 250, 241, 0.84);
            color: #80623a;
            border-radius: 999px;
            padding: 0.58rem 1rem;
            font-size: 0.92rem;
            margin-bottom: 1.5rem;
        }

        .vc-title {
            margin: 0;
            font-family: Georgia, "Times New Roman", serif;
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.06;
            font-size: clamp(3rem, 6vw, 5.1rem);
            color: #202124;
        }

        .vc-title .green {
            color: var(--vc-green);
        }

        .vc-copy {
            margin: 1.4rem 0 1.8rem;
            color: #5d6673;
            font-size: 1.16rem;
            line-height: 1.65;
            max-width: 560px;
        }

        .vc-hero-actions {
            display: flex;
            align-items: center;
            gap: 1.35rem;
        }

        .vc-secondary-link {
            color: var(--vc-green);
            font-weight: 800;
            padding-top: 0.55rem;
        }

        .vc-chat-preview {
            position: relative;
            background: rgba(255, 252, 246, 0.9);
            border: 1px solid var(--vc-border);
            border-radius: 28px;
            padding: 1.45rem;
            box-shadow: var(--vc-shadow);
            min-height: 360px;
            overflow: hidden;
        }

        .vc-chat-preview::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -84px;
            top: -70px;
            border-radius: 999px;
            border: 36px solid rgba(220, 167, 91, 0.08);
        }

        .vc-user-bubble {
            position: relative;
            z-index: 2;
            margin-left: auto;
            width: 70%;
            border-radius: 18px;
            background: #edf5df;
            border: 1px solid rgba(21, 92, 47, 0.1);
            padding: 1.1rem 1.25rem;
            box-shadow: 0 10px 18px rgba(21, 92, 47, 0.08);
            font-size: 1rem;
        }

        .vc-assistant-row {
            position: relative;
            z-index: 2;
            display: grid;
            grid-template-columns: 58px 1fr;
            gap: 0.9rem;
            align-items: start;
            margin-top: 1.2rem;
        }

        .vc-assistant-avatar {
            width: 56px;
            height: 56px;
            border-radius: 18px;
            object-fit: cover;
            border: 3px solid #fffaf1;
            box-shadow: 0 10px 18px rgba(57, 43, 20, 0.16);
        }

        .vc-assistant-bubble {
            border-radius: 18px;
            border: 1px solid rgba(55, 65, 81, 0.1);
            background: #fffdf8;
            padding: 1.1rem 1.25rem;
            line-height: 1.7;
            color: #27272a;
        }

        .vc-mini-input {
            position: relative;
            z-index: 2;
            margin: 1.4rem auto 0;
            border-radius: 999px;
            border: 1px solid rgba(55, 65, 81, 0.11);
            background: #fffdf8;
            color: #9ca3af;
            padding: 1rem 1.2rem;
            width: 92%;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }

        .vc-hf-test {
            position: relative;
            z-index: 2;
            display: grid;
            grid-template-columns: 92px 1fr;
            gap: 0.8rem;
            align-items: center;
            margin-top: 1rem;
            padding: 0.75rem;
            background: rgba(255, 250, 241, 0.78);
            border: 1px solid rgba(220, 167, 91, 0.16);
            border-radius: 18px;
        }

        .vc-hf-test img {
            width: 92px;
            height: 72px;
            object-fit: cover;
            border-radius: 14px;
        }

        .vc-hf-test p {
            margin: 0;
            color: #4b5563;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .vc-cards {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 0.8rem;
        }

        .vc-topic-card {
            min-height: 190px;
            border-radius: 18px;
            overflow: hidden;
            background-size: cover;
            background-position: center;
            position: relative;
            border: 1px solid rgba(255,255,255,0.55);
            box-shadow: 0 18px 36px rgba(57, 43, 20, 0.16);
        }

        .vc-topic-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(180deg, rgba(0,0,0,0.04), rgba(0,0,0,0.68));
        }

        .vc-topic-label {
            position: absolute;
            left: 1rem;
            right: 1rem;
            bottom: 1rem;
            color: #fff;
            font-weight: 800;
            font-size: 1.05rem;
            text-shadow: 0 2px 10px rgba(0,0,0,0.4);
        }

        .vc-footer {
            text-align: center;
            color: #7b817d;
            padding: 1.8rem 0 0.2rem;
        }

        .vc-rec-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 0.4rem;
        }

        .vc-rec-card {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(34, 74, 42, 0.12);
            background: #fffdf8;
            box-shadow: 0 12px 26px rgba(57, 43, 20, 0.1);
        }

        .vc-rec-card img {
            width: 100%;
            height: 150px;
            object-fit: cover;
            display: block;
            background: linear-gradient(135deg, #f4e7d0, #d8b26f);
        }

        .vc-rec-body {
            padding: 0.85rem 0.95rem 0.95rem;
        }

        .vc-rec-title {
            font-weight: 800;
            color: var(--vc-green);
            margin-bottom: 0.25rem;
        }

        .vc-rec-meta {
            color: #6b7280;
            font-size: 0.84rem;
            line-height: 1.45;
        }

        .vc-chat-shell {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 1rem;
        }

        .vc-chat-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 1.1rem;
            border: 1px solid var(--vc-border);
            background: rgba(255, 252, 246, 0.86);
            border-radius: 20px;
            box-shadow: 0 12px 30px rgba(57, 43, 20, 0.08);
            margin-bottom: 1rem;
        }

        .vc-chat-title {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .vc-chat-title img {
            width: 48px;
            height: 48px;
            border-radius: 16px;
            object-fit: cover;
        }

        .vc-chat-title h2 {
            margin: 0;
            color: var(--vc-green);
            font-size: 1.2rem;
        }

        .vc-chat-title p {
            margin: 0.1rem 0 0;
            color: var(--vc-muted);
            font-size: 0.9rem;
        }

        .status-pill {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .status-pill-warning {
            background: #fef3c7;
            color: #92400e;
        }

        .sidebar-note {
            color: #64748b;
            font-size: 0.82rem;
            line-height: 1.35;
        }

        @media (max-width: 980px) {
            .vc-header, .vc-nav, .vc-hero-actions {
                align-items: flex-start;
                flex-direction: column;
            }
            .vc-hero {
                grid-template-columns: 1fr;
                gap: 1.5rem;
            }
            .vc-cards {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .vc-rec-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_hf_image_url_from_path(image_path: str) -> str:
    """
    Chuyển image_path trong dataset thành URL ảnh Hugging Face.

    Biến đầu vào:
    - image_path: path kiểu `data/images/am_thuc/banh_chung_Tet/000001.jpg`.

    Ví dụ output:
    `https://huggingface.co/datasets/.../resolve/main/images/am_thuc/.../000001.jpg`

    Cách tự viết lại:
    Bỏ prefix `data/` nếu có, encode từng phần path để hỗ trợ dấu tiếng Việt,
    rồi ghép với HF_DATASET_BASE_URL.
    """

    clean_path = str(image_path or "").replace("\\", "/").lstrip("/")
    if clean_path.startswith("data/"):
        clean_path = clean_path[len("data/"):]
    encoded_path = "/".join(quote(part, safe="") for part in clean_path.split("/"))
    return f"{HF_DATASET_BASE_URL}/{encoded_path}"


@lru_cache(maxsize=1)
def load_dataset_image_path_index() -> dict[str, str]:
    """
    Tạo index image_path từ JSON dataset để biết đúng extension ảnh.

    Biến đầu vào:
    - Không có input trực tiếp, hàm đọc `data/vietnamese_vqa_dataset.json`.

    Ví dụ output:
    {"kien_truc|nha_truyen_thống_mien_Tây|000039": "data/images/.../000039.png"}

    Cách tự viết lại:
    Đọc file theo từng dòng, chỉ bắt dòng có `image_path`, tách category/folder/số ảnh,
    rồi cache dict bằng lru_cache để không scan lại ở mỗi lần recommendation.
    """

    image_index: dict[str, str] = {}
    if not DATASET_JSON_PATH.exists():
        return image_index

    with DATASET_JSON_PATH.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            match = IMAGE_PATH_PATTERN.search(line)
            if not match:
                continue
            image_path = match.group(1).replace("\\", "/")
            path_parts = image_path.split("/")
            if len(path_parts) < 5:
                continue
            category = path_parts[-3]
            keyword_folder = path_parts[-2]
            image_number = path_parts[-1].rsplit(".", 1)[0]
            image_index[f"{category}|{keyword_folder}|{image_number}"] = image_path
    return image_index


def build_hf_image_url_from_metadata(metadata: dict[str, Any]) -> str:
    """
    Dựng URL ảnh Hugging Face từ metadata của retrieved document.

    Biến đầu vào:
    - metadata: metadata Chroma, thường có `category`, `keyword`, `image_id`.

    Ví dụ output:
    category=am_thuc, keyword=banh chung Tet, image_id=000012
    -> .../images/am_thuc/banh_chung_Tet/000012.jpg

    Cách tự viết lại:
    Nếu metadata có image_path thì dùng image_path. Nếu không, chuyển keyword
    thành folder bằng cách thay space bằng `_`, rồi ghép category/image_id.
    """

    image_path = metadata.get("image_path")
    if image_path:
        return build_hf_image_url_from_path(str(image_path))

    category = str(metadata.get("category", "")).strip()
    keyword = str(metadata.get("keyword") or metadata.get("topic") or "").strip()
    image_id = str(metadata.get("image_id", "")).strip()
    if not category or not keyword or not image_id:
        return HF_BANH_BAO_IMAGE_URL

    image_number = image_id
    if "_" in image_number:
        image_number = image_number.rsplit("_", 1)[-1]
    if image_number.isdigit():
        image_number = image_number.zfill(6)

    keyword_folder = "_".join(keyword.split())
    indexed_image_path = load_dataset_image_path_index().get(
        f"{category}|{keyword_folder}|{image_number}"
    )
    if indexed_image_path:
        return build_hf_image_url_from_path(indexed_image_path)

    extension = ".png" if category == "nhac_cu" and "musical" in keyword_folder else ".jpg"
    image_path = f"images/{category}/{keyword_folder}/{image_number}{extension}"
    return build_hf_image_url_from_path(image_path)


def safe_display_topic(metadata: dict[str, Any]) -> str:
    """
    Lấy tên chủ đề ngắn từ metadata để hiển thị trên card.

    Biến đầu vào:
    - metadata: metadata của document/retrieved chunk.

    Ví dụ output:
    {"keyword": "banh chung Tet"} -> "banh chung Tet"

    Cách tự viết lại:
    Ưu tiên canonical_topic/topic/keyword, fallback về category nếu thiếu.
    """

    return str(
        metadata.get("canonical_topic")
        or metadata.get("topic")
        or metadata.get("keyword")
        or metadata.get("category")
        or "Chủ đề văn hóa"
    )


def build_recommendation_cards_html(documents: list[Any]) -> str:
    """
    Tạo HTML card ảnh cho recommendation dựa trên retrieved documents.

    Biến đầu vào:
    - documents: list LangChain Document trong state["documents"].

    Ví dụ output:
    `<div class="vc-rec-grid"><div class="vc-rec-card">...</div></div>`

    Cách tự viết lại:
    Lấy metadata của từng document, dựng URL ảnh Hugging Face, lấy topic/category
    làm text card, giới hạn 3 card để UI không quá dài.
    """

    if not documents:
        return ""

    cards: list[str] = []
    seen_keys: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        topic = safe_display_topic(metadata)
        image_id = str(metadata.get("image_id", ""))
        category = str(metadata.get("category", ""))
        dedup_key = f"{category}:{topic}:{image_id}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        image_url = build_hf_image_url_from_metadata(metadata)
        question_type = str(metadata.get("question_type", "") or "recommendation")
        keyword = str(metadata.get("keyword", "") or topic)
        topic_html = escape(topic)
        category_html = escape(category or "văn hóa")
        keyword_html = escape(keyword)
        question_type_html = escape(question_type)
        cards.append(
            '<div class="vc-rec-card">'
            f'<img src="{image_url}" alt="{topic_html}">'
            '<div class="vc-rec-body">'
            f'<div class="vc-rec-title">{topic_html}</div>'
            f'<div class="vc-rec-meta">Nhóm: {category_html}<br>'
            f'Từ khóa: {keyword_html}<br>Dạng: {question_type_html}</div>'
            "</div></div>"
        )
        if len(cards) >= 3:
            break

    if not cards:
        return ""
    return '<div class="vc-rec-grid">' + "".join(cards) + "</div>"


def render_header(show_chat_button: bool = True) -> None:
    """
    Render header thương hiệu VietCulture.

    Biến đầu vào:
    - show_chat_button: nếu True thì hiển thị nút bắt đầu ở góc phải.

    Ví dụ output:
    Header có logo, tên VietCulture, nav nhỏ và optional CTA.

    Cách tự viết lại:
    Dùng HTML/CSS cho layout cố định, còn CTA chính vẫn dùng st.button bên ngoài
    khi cần tương tác thật.
    """

    right_button = (
        '<div class="vc-nav"><span>Trang chủ</span><span>Giới thiệu</span><span>Chủ đề</span></div>'
        if not show_chat_button
        else '<div class="vc-nav"><span>Trang chủ</span><span>Giới thiệu</span><span>Chủ đề</span></div>'
    )
    st.markdown(
        f"""
        <div class="vc-header">
            <div class="vc-brand">
                <div class="vc-logo">VC</div>
                <div>
                    <h1>VietCulture</h1>
                    <p>Chatbot văn hóa Việt Nam</p>
                </div>
            </div>
            {right_button}
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_topic_cards_html() -> str:
    """
    Tạo HTML cho các card chủ đề ở welcome screen.

    Biến đầu vào:
    - Không có input, danh sách chủ đề được khai báo trong hàm.

    Ví dụ output:
    Một chuỗi HTML gồm nhiều `.vc-topic-card`.

    Cách tự viết lại:
    Tạo list dict gồm label/background, render từng card bằng loop, rồi join
    thành HTML string.
    """

    topics = [
        ("Ẩm thực", "images/am_thuc/banh_bao/000001.jpg"),
        ("Bánh chưng", "images/am_thuc/banh_chung_Tet/000001.jpg"),
        ("Lễ hội", "images/le_hoi/Vu_Lan_festival/000001.jpg"),
        ("Trang phục", "images/trang_phuc/ao_choang_lemur/000001.jpg"),
        ("Kiến trúc", "images/kien_truc/nha_hat_Lớn_Ha_Noi/000001.jpg"),
        ("Nhạc cụ", "images/nhac_cu/musical_Vietnam/000001.png"),
        ("Làng nghề", "images/thu_cong_my_nghe/đan_lat/000001.jpg"),
        ("Thể thao", "images/the_thao_truyen_thong/bóng_đa_phong_trao_Việt_Nam/000001.jpg"),
    ]

    cards: list[str] = []
    for label, image_path in topics:
        image_url = build_hf_image_url_from_path(image_path)
        label_html = escape(label)
        style = (
            "background-image: "
            "linear-gradient(180deg, rgba(0,0,0,0.03), rgba(0,0,0,0.70)), "
            f"url('{image_url}');"
        )
        cards.append(
            f'<div class="vc-topic-card" style="{style}">'
            f'<div class="vc-topic-label">{label_html}</div>'
            "</div>"
        )
    return '<div class="vc-cards">' + "".join(cards) + "</div>"


def render_welcome_screen() -> None:
    """
    Render màn chào trước khi vào chat.

    Biến đầu vào:
    - Không có input, trạng thái bắt đầu chat nằm trong st.session_state.

    Ví dụ output:
    Landing page có hero text, chat preview, ảnh bánh bao từ Hugging Face và
    các card chủ đề văn hóa.

    Cách tự viết lại:
    Tạo layout hai cột bằng HTML/CSS, dùng st.button cho CTA thật, và gọi st.rerun
    sau khi user bấm bắt đầu.
    """

    avatar_uri = image_to_data_uri(ASSISTANT_AVATAR)
    render_header(show_chat_button=False)

    left_col, right_col = st.columns([1.05, 0.95], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="vc-eyebrow">Hiểu văn hóa Việt, kết nối giá trị Việt</div>
            <h2 class="vc-title">Khám phá<br><span class="green">văn hóa Việt Nam</span><br>qua trò chuyện</h2>
            <p class="vc-copy">
                Hỏi bất cứ điều gì về ẩm thực, lễ hội, trang phục, kiến trúc,
                nhạc cụ và đời sống văn hóa Việt Nam.
            </p>
            """,
            unsafe_allow_html=True,
        )
        action_col, link_col = st.columns([0.46, 0.54], vertical_alignment="center")
        with action_col:
            if st.button("Bắt đầu trò chuyện", key="welcome_start", use_container_width=True):
                st.session_state.started_chat = True
                st.rerun()
        with link_col:
            st.markdown('<div class="vc-secondary-link">Xem chủ đề</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown(
            f"""
            <div class="vc-chat-preview">
                <div class="vc-user-bubble">
                    Bánh bao có ý nghĩa gì trong văn hóa ẩm thực Việt?
                    <div style="text-align:right;color:#60705f;font-size:0.82rem;margin-top:0.7rem;">10:24</div>
                </div>
                <div class="vc-assistant-row">
                    <img class="vc-assistant-avatar" src="{avatar_uri}" alt="VietCulture assistant">
                    <div class="vc-assistant-bubble">
                        Bánh bao là món ăn quen thuộc trong đời sống đô thị và gia đình Việt.
                        Mình có thể giúp bạn tìm hiểu nguồn gốc, cách dùng, hoặc gợi ý chủ đề
                        ẩm thực liên quan từ dataset.
                    </div>
                </div>
                <div class="vc-hf-test">
                    <img src="{HF_BANH_BAO_IMAGE_URL}" alt="Ảnh bánh bao từ Hugging Face">
                    <p>
                        Test ảnh Hugging Face: nếu bạn thấy ảnh này, Streamlit đang render được
                        remote image từ dataset.
                    </p>
                </div>
                <div class="vc-mini-input">Nhập câu hỏi của bạn...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(build_topic_cards_html(), unsafe_allow_html=True)
    st.markdown(
        '<div class="vc-footer">VietCulture - Tôn vinh và lan tỏa giá trị văn hóa Việt</div>',
        unsafe_allow_html=True,
    )


def render_debug_state(state: dict[str, Any]) -> None:
    """
    Hiển thị thông tin debug sau mỗi lượt agent chạy.

    Biến đầu vào:
    - state: GraphState sau khi invoke_agent() chạy xong.

    Ví dụ output trên UI:
    Intent: rag_question
    Route source: rule
    Intent confidence: 0.55

    Cách tự viết lại:
    Dùng st.expander, lấy các field quan trọng trong state như intent,
    transformed_question, documents, user_preferences.
    """

    with st.expander("Debug state", expanded=False):
        st.write("Intent:", state.get("intent"))
        st.write("Route source:", state.get("route_source"))
        st.write("Intent confidence:", state.get("intent_confidence"))
        st.write("Memory update allowed:", state.get("memory_update_allowed"))
        st.write("Route reason:", state.get("route_reason"))
        st.write("Transformed question:", state.get("transformed_question"))
        st.write("Retrieved documents:", len(state.get("documents", [])))
        st.code(state.get("user_preferences", "{}"), language="json")


def render_sidebar(bundle: Any) -> tuple[str, str, str]:
    """
    Render sidebar của màn chat và trả về user/thread key.

    Biến đầu vào:
    - bundle: AgentBundle chứa settings và LLM status.

    Ví dụ output:
    ("demo_user_a", "demo_thread", "chat_history::demo_user_a::demo_thread")

    Cách tự viết lại:
    Đặt input user_id/conversation_id trong sidebar, tạo chat_history_key, hiển
    thị memory hiện tại và các nút reset/clear.
    """

    settings = bundle.settings
    with st.sidebar:
        st.subheader("Hồ sơ người dùng")
        user_id = st.text_input("User ID", value="demo_user_a")
        conversation_id = st.text_input("Conversation", value="demo_thread")
        thread_id = build_langgraph_config(user_id, conversation_id)["configurable"]["thread_id"]
        st.caption(f"Thread: `{thread_id}`")

        chat_history_key = build_chat_history_key(user_id, conversation_id)

        col_a, col_b = st.columns(2)
        if col_a.button("Reset memory", use_container_width=True):
            reset_user_memory(settings.memory_file, user_id)
            st.session_state.pop(chat_history_key, None)
            st.session_state.pop("last_state", None)
            st.success(f"Đã reset memory cho {user_id}")
        if col_b.button("Clear chat", use_container_width=True):
            st.session_state.pop(chat_history_key, None)
            st.success("Đã xóa chat hiện tại")

        st.markdown("---")
        if bundle.llm_generate:
            st.markdown('<span class="status-pill">Gemini ready</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="status-pill status-pill-warning">Gemini missing</span>',
                unsafe_allow_html=True,
            )
        router_status = "Hybrid LLM router" if settings.use_llm_intent_router else "Rule router"
        st.caption(f"Router: {router_status}")
        st.caption(f"Chroma: `{settings.persist_dir.name}` / `{settings.collection_name}`")
        st.markdown(
            '<div class="sidebar-note">Long-term memory theo User ID. '
            'Chat display theo User ID + Conversation.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("Memory đã lưu")
        memory_db = load_memory_db(settings.memory_file)
        current_memory = memory_db.get(user_id, {})
        interest_text, evidence_text = get_memory_summary(current_memory)
        st.write("Sở thích:", interest_text)
        st.caption(f"Evidence: {evidence_text}")
        with st.expander("Memory JSON", expanded=False):
            st.json(current_memory or {})

        if st.button("Quay lại màn chào", use_container_width=True):
            st.session_state.started_chat = False
            st.rerun()

    return user_id, conversation_id, chat_history_key


def render_chat_screen(bundle: Any) -> None:
    """
    Render màn chat chính.

    Biến đầu vào:
    - bundle: AgentBundle đã load một lần.

    Ví dụ output:
    UI chat có topbar, sidebar memory, lịch sử chat và input prompt.

    Cách tự viết lại:
    Render sidebar để lấy user/thread, replay chat history từ session_state, rồi
    khi có prompt thì gọi invoke_agent() và append answer vào lịch sử.
    """

    avatar_uri = image_to_data_uri(ASSISTANT_AVATAR)
    user_id, conversation_id, chat_history_key = render_sidebar(bundle)

    st.markdown(
        f"""
        <div class="vc-chat-topbar">
            <div class="vc-chat-title">
                <img src="{avatar_uri}" alt="Assistant avatar">
                <div>
                    <h2>VietCulture Assistant</h2>
                    <p>Hỏi đáp văn hóa Việt Nam có RAG, memory và recommendation.</p>
                </div>
            </div>
            <span class="status-pill">Đang trò chuyện</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if chat_history_key not in st.session_state:
        st.session_state[chat_history_key] = []

    for message in st.session_state[chat_history_key]:
        avatar = str(ASSISTANT_AVATAR) if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            recommendation_cards_html = message.get("recommendation_cards_html")
            if recommendation_cards_html:
                st.markdown(recommendation_cards_html, unsafe_allow_html=True)

    prompt = st.chat_input("Nhập câu hỏi hoặc sở thích của bạn...")

    if prompt:
        st.session_state[chat_history_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=str(ASSISTANT_AVATAR)):
            with st.spinner("Agent đang xử lý..."):
                state = invoke_agent(
                    bundle=bundle,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message=prompt,
                )
            answer = state.get("answer", "")
            st.markdown(answer)
            recommendation_cards_html = ""
            if state.get("intent") == "recommendation_request":
                recommendation_cards_html = build_recommendation_cards_html(
                    state.get("documents", [])
                )
                if recommendation_cards_html:
                    st.markdown(recommendation_cards_html, unsafe_allow_html=True)
            render_debug_state(state)

        st.session_state[chat_history_key].append(
            {
                "role": "assistant",
                "content": answer,
                "recommendation_cards_html": recommendation_cards_html,
            }
        )
        st.session_state.last_state = state

    st.divider()
    with st.expander("Prompt demo", expanded=False):
        st.code(
            "\n".join(
                [
                    "Tôi thích lễ hội và ẩm thực",
                    "Tôi thích gì?",
                    "Gợi ý cho tôi vài chủ đề phù hợp",
                    "Xe máy là gì?",
                ]
            )
        )


def main() -> None:
    """
    Entry point của Streamlit app.

    Biến đầu vào:
    - Không có input trực tiếp, Streamlit chạy file từ trên xuống.

    Ví dụ output:
    Nếu chưa bắt đầu thì hiện welcome. Nếu đã bắt đầu thì load bundle và hiện chat.

    Cách tự viết lại:
    Inject CSS trước, khởi tạo session_state mặc định, render welcome hoặc chat
    theo cờ `started_chat`.
    """

    st.set_page_config(
        page_title="VietCulture",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_global_styles()
    if "started_chat" not in st.session_state:
        st.session_state.started_chat = False

    if not st.session_state.started_chat:
        render_welcome_screen()
        return

    bundle = load_bundle()
    render_chat_screen(bundle)


if __name__ == "__main__":
    main()

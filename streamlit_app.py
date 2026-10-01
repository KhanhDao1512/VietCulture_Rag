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
import uuid
from typing import Any
from urllib.parse import quote

import streamlit as st

from src.agent.graph import create_agent_bundle, invoke_agent_safely
from src.memory.conversation_store import (
    NEW_CONVERSATION_TITLE,
    append_message,
    choose_conversation_id,
    create_conversation,
    list_conversations,
    load_messages,
)
from src.memory.store import DATASET_CATEGORY_LABELS
from src.routing.text_utils import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSISTANT_AVATAR = ASSETS_DIR / "an_avatar.svg"
HF_BANH_BAO_IMAGE_URL = (
    "https://huggingface.co/datasets/Dangindev/viet-cultural-vqa/"
    "resolve/main/images/am_thuc/banh_bao/000001.jpg"
)
HF_DATASET_BASE_URL = "https://huggingface.co/datasets/Dangindev/viet-cultural-vqa/resolve/main"
HOMEPAGE_TOPICS = [
    ("Ẩm thực", "Hương vị ba miền", "images/am_thuc/banh_chung_Tet/000001.jpg"),
    ("Kiến trúc", "Dấu ấn kiến trúc Việt", "images/kien_truc/nha_hat_Lớn_Ha_Noi/000001.jpg"),
    ("Lễ hội", "Sắc màu lễ hội", "images/le_hoi/Vu_Lan_festival/000001.jpg"),
    ("Phong cảnh", "Non nước Việt Nam", "images/phong_canh/sông_Hồng/000001.jpg"),
    ("Trang phục", "Trang phục qua thời kỳ", "images/trang_phuc/ao_choang_lemur/000001.jpg"),
    ("Đời sống hằng ngày", "Nếp sống thường ngày", "images/doi_song_hang_ngay/thu_hoạch_lúa/000001.jpg"),
    ("Giao thông", "Giao thông xưa và nay", "images/giao_thong/xe_bò/000001.jpg"),
    ("Thủ công mỹ nghệ", "Nghề thủ công truyền thống", "images/thu_cong_my_nghe/đan_lat/000001.jpg"),
    ("Nhạc cụ", "Thanh âm dân tộc", "images/nhac_cu/musical_Vietnam/000001.png"),
    ("Văn hóa dân gian", "Di sản văn hóa dân gian", "images/van_hoa_dan_gian/xẩm_singing/000001.jpg"),
    ("Trò chơi dân gian", "Trò chơi và sinh hoạt cộng đồng", "images/tro_choi_dan_gian/wrestling_traditional/000001.jpg"),
    ("Thể thao truyền thống", "Tinh thần thượng võ", "images/the_thao_truyen_thong/bóng_đa_phong_trao_Việt_Nam/000001.jpg"),
]


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
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(suffix, "image/png")
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
            cursor: pointer;
            text-decoration: none;
            display: block;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .vc-topic-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 22px 40px rgba(57, 43, 20, 0.25);
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

        /* Tokens và component styles khớp với Design/01 và Design/02. */
        :root {
            --vc-green: #126044;
            --vc-green-dark: #0d4d37;
            --vc-cream: #f7f3e9;
            --vc-cream-2: #ffffff;
            --vc-ink: #173d32;
            --vc-muted: #74877d;
            --vc-border: rgba(18, 96, 68, 0.10);
            --vc-shadow: 0 14px 36px rgba(24, 65, 49, 0.08);
        }

        .stApp { background: #f5f8f5; color: var(--vc-ink); }
        .block-container { max-width: 1240px; padding-top: 1.4rem; padding-bottom: 2rem; }
        .vc-header {
            min-height: 76px; padding: 0.25rem 0 0.8rem; border-bottom: 1px solid #e9eeea;
            align-items: center; overflow: visible; box-sizing: border-box;
        }
        .vc-brand { gap: 0.7rem; }
        .vc-logo {
            width: 44px; height: 44px; border-radius: 14px; background: var(--vc-green);
            box-shadow: none; font-size: 1.35rem; line-height: 1; overflow: visible;
        }
        .vc-brand { flex-shrink: 0; }
        .vc-brand h1 { color: var(--vc-green); font: 750 clamp(1.3rem, 2vw, 1.55rem)/1.2 "Segoe UI", Arial, sans-serif; letter-spacing: -0.025em; }
        .vc-nav { gap: clamp(0.65rem, 1.5vw, 1.4rem); margin-left: auto; margin-right: 0.8rem; }
        .vc-nav a { white-space: nowrap; }
        .vc-nav a { color: #65776d; text-decoration: none; font-weight: 500; }
        .vc-nav a.active { color: var(--vc-green); font-weight: 700; }
        .vc-nav-cta, .vc-primary-link {
            display: inline-flex; align-items: center; gap: 0.75rem; border-radius: 12px;
            background: var(--vc-green); color: #fff !important; padding: 0.78rem 1.25rem;
            min-height: 44px; box-sizing: border-box; white-space: nowrap;
            font-weight: 700; line-height: 1.4; text-decoration: none !important; align-self: center;
        }
        .vc-home-hero {
            display: grid; grid-template-columns: 1.08fr 0.92fr; align-items: center;
            gap: 2rem; margin: 1.6rem 0 1.8rem; padding: 2rem;
            border-radius: 24px; background: #e8f2eb;
        }
        .vc-home-copy { padding: 0.4rem 0.8rem; }
        .vc-eyebrow { border: 0; background: transparent; color: var(--vc-green); padding: 0; margin-bottom: 1rem; font-size: 0.78rem; letter-spacing: 0.04em; }
        .vc-home-copy h2 { margin: 0; color: #173d32; font: 750 clamp(2.4rem, 4.2vw, 3.35rem)/1.16 "Segoe UI", Arial, sans-serif; letter-spacing: -0.035em; }
        .vc-home-copy p { max-width: 570px; color: var(--vc-muted); font-size: 1rem; line-height: 1.55; margin: 1rem 0 1.25rem; }
        .vc-primary-link { padding: 0.72rem 1rem; }
        .vc-welcome-card { display: flex; align-items: center; gap: 1.1rem; background: #fff; border-radius: 20px; padding: 1.4rem; min-height: 180px; }
        .vc-welcome-card img { width: 96px; height: 112px; object-fit: cover; border-radius: 18px; }
        .vc-welcome-card strong { color: var(--vc-green); font-size: 1.05rem; }
        .vc-welcome-card p { color: var(--vc-muted); line-height: 1.5; margin: 0.45rem 0; }
        .vc-welcome-card span { color: var(--vc-green); font-size: 0.9rem; }
        .vc-section-heading { display: flex; align-items: center; justify-content: space-between; margin: 0.7rem 0 0.5rem; }
        .vc-section-heading h2 { margin: 0; font: 750 1.65rem/1.25 "Segoe UI", Arial, sans-serif; letter-spacing: -0.02em; color: #173d32; }
        .vc-section-heading p { margin: 0.3rem 0 0; color: var(--vc-muted); font-size: 0.9rem; }
        .vc-section-heading > span { color: var(--vc-green); font-size: 0.82rem; }
        .vc-cards { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin-top: 1rem; }
        .vc-topic-card { min-height: 0; background: white; border: 0; border-radius: 16px; box-shadow: none; color: var(--vc-ink); text-decoration: none !important; }
        .vc-topic-card::before { display: none; }
        .vc-topic-card img { display: block; width: 100%; height: 132px; object-fit: cover; background: #e8f2eb; }
        .vc-topic-card-copy { display: flex; flex-direction: column; gap: 0.35rem; padding: 0.8rem 0.9rem 0.9rem; }
        .vc-topic-card-copy strong { color: var(--vc-ink); font-size: 1.02rem; text-decoration: none; }
        .vc-topic-card-copy span { color: var(--vc-muted); font-size: 0.8rem; }
        .vc-topic-card:hover { transform: translateY(-3px); box-shadow: var(--vc-shadow); }
        .vc-footer { color: var(--vc-muted); border-top: 1px solid #e9eeea; margin-top: 1.2rem; font-size: 0.82rem; }
        .vc-chat-layout { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 1.5rem; }
        .vc-topic-nav-title { color: var(--vc-green); font-weight: 700; margin: 0.4rem 0 0.8rem; }
        .vc-chat-topbar { background: #fff; border: 0; border-radius: 0; box-shadow: none; border-bottom: 1px solid #edf0ed; margin: -0.5rem -0.5rem 1rem; padding: 0.9rem 1.2rem; }
        .vc-chat-title h2 { color: var(--vc-green); }
        .vc-chat-title img { border-radius: 50%; }
        [data-testid="stChatMessage"] { border-radius: 16px; }
        [data-testid="stChatInput"] { border: 1px solid #bcd5c5; border-radius: 18px; background: #dcebe1; box-shadow: 0 8px 24px rgba(24,65,49,0.10); }
        [data-testid="stChatInput"] textarea { color: var(--vc-ink); font-size: 1.08rem; line-height: 1.55; }
        [data-testid="stChatMessage"] { font-size: 1.08rem; line-height: 1.7; }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { font-size: 1.08rem; line-height: 1.7; }
        [data-testid="stChatMessage"]:has(img[alt="An"]) [data-testid="stMarkdownContainer"] { background: #fff; border: 1px solid #edf0ed; border-radius: 16px; padding: 0.75rem 1rem; }
        .vc-user-message { width: fit-content; max-width: min(86%, 760px); margin-left: auto; background: #e0eee5; border: 1px solid #d3e5d9; border-radius: 18px; padding: 0.85rem 1.1rem; color: #173d32; font-size: 1.08rem; line-height: 1.65; overflow-wrap: anywhere; }
        div[data-testid="stButton"] > button { background: #fff; border: 0; color: var(--vc-green); box-shadow: none; text-align: left; min-height: 2.7rem; }
        div[data-testid="stButton"] > button:hover { background: #e8f2eb; color: var(--vc-green-dark); border: 0; }
        [data-testid="stSidebar"] { background: #fff; }
        .vc-chat-note { background: #eff6f0; border-radius: 14px; padding: 0.85rem; color: var(--vc-muted); font-size: 0.8rem; line-height: 1.45; }
        @media (max-width: 900px) {
            .vc-home-hero { grid-template-columns: 1fr; padding: 1.25rem; }
            .vc-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .vc-nav { display: none; }
        }
        @media (max-width: 620px) {
            .vc-cards { grid-template-columns: 1fr 1fr; gap: 0.6rem; }
            .vc-topic-card img { height: 96px; }
        }
        @media (max-width: 760px) {
            .vc-header { gap: 0.6rem; }
            .vc-brand h1 { font-size: 1.2rem; }
            .vc-nav-cta { min-height: 40px; padding: 0.55rem 0.75rem; }
            .vc-home-copy h2 { font-size: 2.15rem; }
            .vc-section-heading h2 { font-size: 1.4rem; }
            .vc-user-message { max-width: 94%; }
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
    # Runtime không đọc raw dataset chỉ để tìm extension. Các index mới đã lưu
    # image_path trong metadata; index legacy dùng fallback theo category.
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
        metadata.get("display_topic")
        or metadata.get("canonical_topic")
        or metadata.get("topic")
        or metadata.get("keyword")
        or metadata.get("category")
        or "Chủ đề văn hóa"
    )


def build_recommendation_cards_html(documents: list[Any], max_cards: int = 5) -> str:
    """
    Tạo HTML card ảnh cho recommendation dựa trên retrieved documents.

    Số card hiển thị khớp với giới hạn 3–5 mục của graph/formatter.

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
        if "display_topic" in metadata and not metadata.get("display_topic"):
            continue
        topic = safe_display_topic(metadata)
        category = str(metadata.get("category", ""))
        dedup_key = f"{category}:{normalize_text(topic)}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        image_url = build_hf_image_url_from_metadata(metadata)
        question_type = str(metadata.get("question_type", "") or "recommendation")
        keyword = topic
        topic_html = escape(topic)
        category_html = escape(DATASET_CATEGORY_LABELS.get(category, category or "văn hóa"))
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
        if len(cards) >= min(5, max(3, int(max_cards))):
            break

    if not cards:
        return ""
    return '<div class="vc-rec-grid">' + "".join(cards) + "</div>"


def _conversation_query_suffix() -> str:
    """Carry the active user/chat through the app's full-page HTML links."""

    user_id = str(st.session_state.get("vc_user_id", "") or "").strip()
    conversation_id = str(
        st.session_state.get("vc_active_conversation", "") or ""
    ).strip()
    params = []
    if user_id:
        params.append("user_id=" + quote(user_id, safe=""))
    if conversation_id:
        params.append("conversation_id=" + quote(conversation_id, safe=""))
    return ("&" + "&".join(params)) if params else ""


def render_header(active_page: str = "home") -> None:
    """
    Render header thương hiệu VietCulture.

    Biến đầu vào:
    - active_page: màn hiện tại để đánh dấu điều hướng.

    Ví dụ output:
    Header có logo, tên VietCulture, nav nhỏ và optional CTA.

    Cách tự viết lại:
    Dùng HTML/CSS cho layout cố định, còn CTA chính vẫn dùng st.button bên ngoài
    khi cần tương tác thật.
    """

    nav_items = [
        ("home", "Trang chủ", "?home=true" + _conversation_query_suffix()),
        ("topics", "Chủ đề", "#topics"),
        ("about", "Giới thiệu", "#about"),
    ]
    nav_links = "".join(
        f'<a class="{"active" if key == active_page else ""}" href="{href}">{label}</a>'
        for key, label, href in nav_items
    )
    st.markdown(
        f"""
        <div class="vc-header">
            <div class="vc-brand">
                <div class="vc-logo">V</div>
                <h1>VietCulture</h1>
            </div>
            <nav class="vc-nav">{nav_links}</nav>
            <a class="vc-nav-cta" href="?start_chat=true{_conversation_query_suffix()}">Trò chuyện</a>
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

    cards: list[str] = []
    for label, description, image_path in HOMEPAGE_TOPICS:
        image_url = build_hf_image_url_from_path(image_path)
        label_html = escape(label)
        topic_param = quote(label)  # Mã hóa string để đưa lên URL an toàn

        # Đổi thành thẻ <a> truyền query param
        description_html = escape(description)
        cards.append(
            f'<a href="?topic={topic_param}{_conversation_query_suffix()}" target="_self" class="vc-topic-card">'
            f'<img src="{image_url}" alt="{label_html}" loading="lazy">'
            f'<div class="vc-topic-card-copy"><strong>{label_html}</strong>'
            f'<span>{description_html} &nbsp;→</span></div></a>'
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
    render_header(active_page="home")
    st.markdown(
        f"""
        <section class="vc-home-hero">
          <div class="vc-home-copy">
            <div class="vc-eyebrow">HIỂU VĂN HÓA VIỆT · KẾT NỐI GIÁ TRỊ VIỆT</div>
            <h2>Mỗi câu hỏi,<br>một nét đẹp Việt Nam.</h2>
            <p>Cùng khám phá những câu chuyện về ẩm thực, con người và di sản Việt Nam qua cuộc trò chuyện với An.</p>
            <a class="vc-primary-link" href="?start_chat=true">Bắt đầu trò chuyện <span>→</span></a>
          </div>
          <div class="vc-welcome-card">
            <img src="{avatar_uri}" alt="An, trợ lý văn hóa Việt Nam">
            <div><strong>Xin chào, mình là An!</strong>
              <p>Người bạn đồng hành cùng bạn tìm hiểu văn hóa Việt.</p>
              <span>Bạn muốn khám phá điều gì?</span>
            </div>
          </div>
        </section>
        <section class="vc-topic-section" id="topics">
          <div class="vc-section-heading"><div><h2>Hôm nay, bạn muốn khám phá gì?</h2>
          <p>Chọn một chủ đề để bắt đầu trò chuyện cùng An.</p></div><span>{len(HOMEPAGE_TOPICS)} chủ đề văn hóa</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(build_topic_cards_html(), unsafe_allow_html=True)
    st.markdown(
        '<div class="vc-footer">VietCulture · Văn hóa gần hơn qua mỗi cuộc trò chuyện</div>',
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
        st.write("Tên chuẩn hóa:", state.get("canonical_names", []))
        st.write("Retrieved documents:", len(state.get("documents", [])))
        if state.get("recommendation_debug"):
            st.write("Recommendation pipeline:", state["recommendation_debug"])
        if state.get("agent_error"):
            st.write("Agent error type:", state["agent_error"])
        for index, document in enumerate(state.get("documents", []), start=1):
            metadata = getattr(document, "metadata", {}) or {}
            with st.expander(
                f"Tài liệu {index}: "
                f"{metadata.get('canonical_topic') or metadata.get('topic') or metadata.get('keyword', 'không tên')}"
            ):
                st.write("Category:", metadata.get("category"))
                st.write("Question type:", metadata.get("question_type"))
                st.write("Image path:", metadata.get("image_path"))
                st.code(getattr(document, "page_content", ""))
        st.code(state.get("user_preferences", "{}"), language="json")


def render_sidebar(bundle: Any) -> tuple[str, str]:
    """Render lịch sử hội thoại và memory trong sidebar native của Streamlit."""

    settings = bundle.settings
    with st.sidebar:
        st.markdown("### VietCulture")
        requested_user_id = str(st.query_params.get("user_id", "") or "").strip()
        if requested_user_id and "vc_user_id" not in st.session_state:
            st.session_state["vc_user_id"] = requested_user_id
        user_id = st.text_input("User ID", value="demo_user_a", key="vc_user_id").strip()
        user_id = user_id or "anonymous"
        memory_db = load_memory_db(settings.memory_file)
        current_memory = memory_db.get(user_id, {})
        conversations = list_conversations(settings.conversation_db, user_id)

        if not conversations:
            conversation_id = f"chat_{uuid.uuid4().hex[:10]}"
            create_conversation(settings.conversation_db, user_id, conversation_id)
            if not (current_memory.get("categories") or current_memory.get("topics")):
                append_message(
                    settings.conversation_db,
                    user_id,
                    conversation_id,
                    "assistant",
                    "Chào bạn! Có vẻ đây là lần đầu bạn trò chuyện với mình. "
                    "Bạn hứng thú với lĩnh vực văn hóa nào?",
                )
            conversations = list_conversations(settings.conversation_db, user_id)

        conversation_ids = [item["conversation_id"] for item in conversations]
        user_changed = st.session_state.get("vc_active_user") != user_id
        requested_conversation_id = str(
            st.query_params.get("conversation_id", "") or ""
        ).strip()
        selected_conversation_id = choose_conversation_id(
            conversation_ids,
            requested_id=(
                requested_conversation_id
                if user_changed or not st.session_state.get("vc_active_conversation")
                else None
            ),
            stored_id=(
                None
                if user_changed
                else st.session_state.get("vc_active_conversation")
            ),
        )
        st.session_state["vc_active_user"] = user_id
        st.session_state["vc_active_conversation"] = selected_conversation_id

        if st.button("＋ Cuộc trò chuyện mới", key="vc_new_conversation", use_container_width=True):
            conversation_id = f"chat_{uuid.uuid4().hex[:10]}"
            create_conversation(settings.conversation_db, user_id, conversation_id)
            if not (current_memory.get("categories") or current_memory.get("topics")):
                append_message(
                    settings.conversation_db,
                    user_id,
                    conversation_id,
                    "assistant",
                    "Chào bạn! Có vẻ đây là lần đầu bạn trò chuyện với mình. "
                    "Bạn hứng thú với lĩnh vực văn hóa nào?",
                )
            st.session_state["vc_active_user"] = user_id
            st.session_state["vc_active_conversation"] = conversation_id
            st.query_params["user_id"] = user_id
            st.query_params["conversation_id"] = conversation_id
            st.rerun()

        title_by_id = {item["conversation_id"]: item["title"] for item in conversations}
        # Hội thoại vừa tạo có thể chưa nằm trong snapshot trước khi rerun.
        selected_id = st.radio(
            "Lịch sử trò chuyện",
            options=conversation_ids,
            key="vc_active_conversation",
            format_func=lambda item: title_by_id.get(item, NEW_CONVERSATION_TITLE),
            label_visibility="collapsed",
        )
        if st.query_params.get("user_id") != user_id:
            st.query_params["user_id"] = user_id
        if st.query_params.get("conversation_id") != selected_id:
            st.query_params["conversation_id"] = selected_id

        st.markdown("---")
        st.caption(f"User đang xem: `{user_id}`")
        if bundle.llm_generate:
            provider_label = f"{settings.llm_provider} · {settings.llm_model}"
            st.markdown(f'<span class="status-pill">{escape(provider_label)}</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<span class="status-pill status-pill-warning">LLM chưa cấu hình</span>',
                unsafe_allow_html=True,
            )
        router_status = "Hybrid LLM router" if settings.use_llm_intent_router else "Rule router"
        st.caption(f"Router: {router_status}")
        st.caption(f"Chroma: `{settings.persist_dir.name}` / `{settings.collection_name}`")

        st.markdown("---")
        st.subheader("Memory")
        interest_text, evidence_text = get_memory_summary(current_memory)
        st.write("Sở thích:", interest_text)
        st.caption(f"Evidence: {evidence_text}")
        with st.expander("Memory JSON", expanded=False):
            st.json(current_memory or {})
        if st.button("Reset memory", key="vc_reset_memory", use_container_width=True):
            reset_user_memory(settings.memory_file, user_id)
            st.rerun()

    return user_id, selected_id


def _render_chat_message(message: dict[str, Any]) -> None:
    """Vẽ user/assistant với bề mặt màu riêng và transcript SQLite."""

    if message["role"] == "user":
        safe_text = escape(str(message["content"])).replace("\n", "<br>")
        st.markdown(f'<div class="vc-user-message">{safe_text}</div>', unsafe_allow_html=True)
        return

    with st.chat_message("assistant", avatar=str(ASSISTANT_AVATAR)):
        st.markdown(message["content"])
        recommendation_cards_html = message.get("recommendation_cards_html") or ""
        if recommendation_cards_html:
            st.markdown(recommendation_cards_html, unsafe_allow_html=True)


def render_chat_screen(bundle: Any) -> None:
    """Vẽ transcript từ SQLite và giữ chat input ở cuối vùng nội dung chính."""

    user_id, conversation_id = render_sidebar(bundle)
    current_memory = load_memory_db(bundle.settings.memory_file).get(user_id, {})
    messages = load_messages(bundle.settings.conversation_db, user_id, conversation_id)

    render_header(active_page="chat")
    st.markdown(
        '<div class="vc-chat-topbar"><div class="vc-chat-title">'
        '<img src="' + image_to_data_uri(ASSISTANT_AVATAR) + '" alt="An">'
        '<div><h2>An · Hướng dẫn văn hóa</h2>'
        '<p>Cùng khám phá những câu chuyện văn hóa Việt Nam</p></div></div>'
        '<span class="status-pill">● Sẵn sàng trò chuyện</span></div>',
        unsafe_allow_html=True,
    )

    last_state = st.session_state.get(f"last_state::{user_id}::{conversation_id}")
    for index, message in enumerate(messages):
        _render_chat_message(message)
        if index == len(messages) - 1 and message["role"] == "assistant" and last_state:
            render_debug_state(last_state)

    has_memory = bool(current_memory.get("categories") or current_memory.get("topics"))
    if not messages or (len(messages) == 1 and messages[0]["role"] == "assistant"):
        st.markdown('<div class="vc-chat-note"><strong>Cứ hỏi điều bạn tò mò</strong><br>'
                    'An sẽ cùng bạn tìm hiểu từng câu chuyện văn hóa.</div>',
                    unsafe_allow_html=True)
        if not has_memory:
            starter_columns = st.columns(4)
            for index, topic in enumerate(["Ẩm thực", "Lễ hội", "Kiến trúc", "Trang phục"]):
                if starter_columns[index].button(topic, key=f"vc_coldstart_{topic}"):
                    st.session_state["pending_prompt"] = f"Tôi quan tâm đến {topic.lower()}"
                    st.rerun()

    pending_prompt = st.session_state.pop("pending_prompt", None)
    if not pending_prompt:
        pending_prompt = st.session_state.pop("welcome_quick_prompt", None)
    user_input = st.chat_input("Bạn muốn tìm hiểu điều gì về văn hóa Việt?")
    final_prompt = user_input or pending_prompt
    if final_prompt:
        append_message(
            bundle.settings.conversation_db, user_id, conversation_id, "user", final_prompt
        )
        with st.chat_message("user"):
            safe_text = escape(final_prompt).replace("\n", "<br>")
            st.markdown(f'<div class="vc-user-message">{safe_text}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant", avatar=str(ASSISTANT_AVATAR)):
            with st.spinner("An đang tìm câu trả lời..."):
                state = invoke_agent_safely(
                    bundle=bundle,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    message=final_prompt,
                    history=messages,
                )
            answer = state.get("answer", "")
            st.markdown(answer)
            recommendation_cards_html = ""
            if state.get("intent") == "recommendation_request":
                recommendation_cards_html = build_recommendation_cards_html(
                    state.get("documents", []),
                    max_cards=bundle.settings.recommendation_count,
                )
                if recommendation_cards_html:
                    st.markdown(recommendation_cards_html, unsafe_allow_html=True)

        append_message(
            bundle.settings.conversation_db,
            user_id,
            conversation_id,
            "assistant",
            answer,
            recommendation_cards_html,
        )
        st.session_state[f"last_state::{user_id}::{conversation_id}"] = state
        st.rerun()

    st.caption("An có thể nhầm lẫn. Hãy kiểm chứng thông tin quan trọng.")


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

    # Preserve the selected identity before query params are cleared by the
    # home/start-chat/topic navigation handlers below.
    query_user_id = str(st.query_params.get("user_id", "") or "").strip()
    query_conversation_id = str(
        st.query_params.get("conversation_id", "") or ""
    ).strip()
    if query_user_id and "vc_user_id" not in st.session_state:
        st.session_state["vc_user_id"] = query_user_id
    if query_conversation_id and "vc_active_conversation" not in st.session_state:
        st.session_state["vc_active_conversation"] = query_conversation_id

    if "home" in st.query_params:
        st.session_state.started_chat = False
        st.query_params.clear()
        st.rerun()
    if "start_chat" in st.query_params:
        st.session_state.started_chat = True
        st.query_params.clear()

    if "topic" in st.query_params:
        selected_topic = st.query_params["topic"]
        st.session_state.started_chat = True
        st.session_state.welcome_quick_prompt = f"Tôi thích {selected_topic.lower()}"
        st.query_params.clear()  # Xóa URL param để giữ UI sạch sẽ

    if "started_chat" not in st.session_state:
        st.session_state.started_chat = False

    if not st.session_state.started_chat:
        render_welcome_screen()
        return

    bundle = load_bundle()
    render_chat_screen(bundle)


if __name__ == "__main__":
    main()

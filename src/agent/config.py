"""
Cấu hình runtime cho VietCulture memory agent.

Mục đích file:
Đọc `.env`, chọn Chroma index baseline đã build trên Kaggle, cấu hình
embedding/runtime, và ép cache HuggingFace nằm trong thư mục project để dễ quản
lý khi demo.

Luồng xử lý:
load_agent_settings()
-> load `.env`
-> configure_huggingface_cache()
-> chọn Chroma baseline: chroma_db / langchain
-> trả AgentSettings cho graph.py và retriever.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentSettings:
    """
    Toàn bộ config cần để build agent graph.

    Biến quan trọng:
    - project_root: thư mục gốc project.
    - memory_file: file JSON lưu long-term memory.
    - retriever_profile: hiện giữ để tương thích, mặc định dùng `legacy`.
    - persist_dir: thư mục Chroma thực tế.
    - collection_name: tên collection trong Chroma.
    - embed_model/device: model embedding dùng lúc query runtime.
    - top_k/fetch_k: số kết quả retrieval cuối/candidate ban đầu.
    - google_api_key/gemini_model: cấu hình Gemini.
    - use_llm_intent_router: bật/tắt LLM fallback cho intent mơ hồ.
    - llm_intent_confidence_threshold: rule confidence dưới ngưỡng này mới gọi LLM.

    Ví dụ output:
    AgentSettings(retriever_profile="legacy", persist_dir=Path("D:/Ds107/chroma_db"))

    Cách tự viết lại:
    Gom toàn bộ biến môi trường vào một dataclass để các file khác không phải
    tự đọc `.env` lặp lại.
    """

    project_root: Path
    memory_file: Path
    retriever_profile: str
    persist_dir: Path
    collection_name: str
    embed_model: str
    device: str
    top_k: int
    fetch_k: int
    lexical_weight: float
    retrieval_max_score: float | None
    max_retries: int
    gemini_model: str
    google_api_key: str | None
    use_llm_intent_router: bool
    llm_intent_confidence_threshold: float


def configure_huggingface_cache(project_root: Path) -> None:
    """
    Đặt cache HuggingFace vào trong workspace.

    Ví dụ output:
    HF_HOME=D:/Ds107/.cache/huggingface

    Cách tự viết lại:
    Tạo thư mục cache, rồi set các biến môi trường HF_HOME,
    HUGGINGFACE_HUB_CACHE, SENTENCE_TRANSFORMERS_HOME.
    """

    cache_dir = project_root / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def load_agent_settings(project_root: str | Path | None = None) -> AgentSettings:
    """
    Đọc toàn bộ settings từ `.env` và biến môi trường.

    Ví dụ output hiện tại:
        retriever_profile="legacy"
        persist_dir="D:/Ds107/chroma_db"
        collection_name="langchain"

    Cách tự viết lại:
    Load `.env`, đặt default index là `chroma_db/langchain`, cho phép env
    override khi cần test index khác, rồi ép kiểu int/float cho các tham số
    retrieval.
    """

    root = Path(project_root or Path.cwd()).resolve()
    load_dotenv(root / ".env")
    configure_huggingface_cache(root)

    retriever_profile = os.getenv("QA_RETRIEVER_PROFILE", "legacy").strip().lower()
    default_index_dir = root / "chroma_db"
    default_collection_name = "langchain"

    persist_dir = Path(os.getenv("QA_CHROMA_DIR", default_index_dir))
    collection_name = os.getenv("QA_CHROMA_COLLECTION", default_collection_name)
    test_index_dir = root / "chroma_db_qa_test"

    if not persist_dir.exists() and test_index_dir.exists():
        persist_dir = test_index_dir
        if "QA_CHROMA_COLLECTION" not in os.environ:
            collection_name = "qa_chunks"

    raw_max_score = os.getenv("QA_RETRIEVAL_MAX_SCORE", "").strip()
    use_llm_intent_router = os.getenv("USE_LLM_INTENT_ROUTER", "false").strip().lower()

    return AgentSettings(
        project_root=root,
        memory_file=Path(os.getenv("MEMORY_FILE", root / "user_memories.json")),
        retriever_profile=retriever_profile,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embed_model=os.getenv("QA_EMBED_MODEL", "intfloat/multilingual-e5-base"),
        device=os.getenv("QA_RETRIEVER_DEVICE", "cpu"),
        top_k=int(os.getenv("QA_TOP_K", "5")),
        fetch_k=int(os.getenv("QA_FETCH_K", "50")),
        lexical_weight=float(os.getenv("QA_LEXICAL_WEIGHT", "0.06")),
        retrieval_max_score=float(raw_max_score) if raw_max_score else None,
        max_retries=int(os.getenv("QA_MAX_RETRIES", "3")),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        use_llm_intent_router=use_llm_intent_router in {"1", "true", "yes", "on"},
        llm_intent_confidence_threshold=float(
            os.getenv("LLM_INTENT_CONFIDENCE_THRESHOLD", "0.75")
        ),
    )

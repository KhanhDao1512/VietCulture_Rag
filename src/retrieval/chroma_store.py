"""
Helper load Chroma và embedding E5 dùng chung cho ingestion và runtime.

Mục đích file:
Tách phần khởi tạo embedding/load Chroma ra khỏi `src.ingestion` để app runtime
không phụ thuộc ngược vào script build dữ liệu offline.

Flow sử dụng:
QaRetriever -> load_chroma_index() -> E5Embeddings -> Chroma(...)
build_chroma_index.py -> E5Embeddings -> Chroma.from_documents(...)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"


def load_hf_token_from_env_file(env_file: str | Path = DEFAULT_ENV_FILE) -> None:
    """
    Đọc HF_TOKEN từ file `.env` nếu process hiện tại chưa có token.

    Biến đầu vào:
    - env_file: đường dẫn file `.env`, mặc định là project `.env`.

    Ví dụ output:
    os.environ["HF_TOKEN"] được set, nhưng token không bị print ra terminal.

    Cách tự viết lại:
    Set cache folder trước, kiểm tra env đã có HF_TOKEN chưa. Nếu chưa, đọc từng
    dòng `.env`, tìm key HF_TOKEN và đưa value vào os.environ.
    """

    DEFAULT_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(DEFAULT_HF_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(DEFAULT_HF_CACHE_DIR / "hub")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(DEFAULT_HF_CACHE_DIR / "sentence_transformers")
    os.environ["TRANSFORMERS_CACHE"] = str(DEFAULT_HF_CACHE_DIR / "transformers")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    if os.environ.get("HF_TOKEN"):
        return

    env_path = Path(env_file)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() != "HF_TOKEN":
            continue

        os.environ["HF_TOKEN"] = value.strip().strip('"').strip("'")
        return


class E5Embeddings:
    """
    Wrapper cho HuggingFaceEmbeddings theo format E5.

    E5 model được train với prefix:
    - `passage: ...` cho documents lưu vào Chroma.
    - `query: ...` cho câu hỏi của user.

    Ví dụ output:
    embed_query("Bánh tét là gì?") -> list[float] vector embedding.

    Cách tự viết lại:
    Bọc model embedding gốc trong class có hai hàm `embed_documents()` và
    `embed_query()`, thêm prefix đúng trước khi gọi model thật.
    """

    def __init__(self, model_name: str, device: str) -> None:
        # Cần set HF_TOKEN/cache trước khi import hoặc khởi tạo model.
        load_hf_token_from_env_file()

        from langchain_huggingface import HuggingFaceEmbeddings

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed các QA chunks trước khi lưu vào Chroma.

        Biến đầu vào:
        - texts: list nội dung Document cần index.

        Ví dụ output:
        ["Question: ... Answer: ..."] -> [[0.01, -0.02, ...], ...]

        Cách tự viết lại:
        Thêm prefix `passage: ` cho mỗi text rồi gọi embed_documents của model.
        """

        passages = [f"passage: {text}" for text in texts]
        return self.embedding_model.embed_documents(passages)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query runtime của user.

        Biến đầu vào:
        - text: câu hỏi/search query.

        Ví dụ output:
        "query: Xe máy là gì?" -> [0.03, 0.12, ...]

        Cách tự viết lại:
        Thêm prefix `query: ` cho câu hỏi rồi gọi embed_query của model.
        """

        return self.embedding_model.embed_query(f"query: {text}")


def load_chroma_index(
    persist_directory: str | Path,
    collection_name: str,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cpu",
) -> Any:
    """
    Load Chroma index đã tồn tại để search runtime.

    Biến đầu vào:
    - persist_directory: thư mục Chroma đang có sẵn.
    - collection_name: tên collection trong Chroma.
    - model_name/device: embedding model và thiết bị query runtime.

    Ví dụ output:
    Chroma(persist_directory="D:/Ds107/chroma_db", collection_name="langchain")

    Cách tự viết lại:
    Khởi tạo cùng embedding function như lúc build index, rồi tạo object Chroma
    trỏ vào persist_directory cũ.
    """

    from langchain_chroma import Chroma

    embeddings = E5Embeddings(model_name=model_name, device=device)

    return Chroma(
        persist_directory=str(persist_directory),
        collection_name=collection_name,
        embedding_function=embeddings,
    )

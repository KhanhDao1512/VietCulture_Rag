"""
Build hoặc load Chroma index từ QA chunks sạch.

Mục đích file:
Biến raw Vietnamese VQA dataset thành vector database lưu trên disk. Đây là bước
embedding nặng, nên có thể chạy file này trên Kaggle/GPU rồi copy thư mục
Chroma về project local.

Flow build embedding:
raw JSON dataset
-> load_clean_documents()
-> clean_qa_chunks.build_qa_chunks()
-> E5Embeddings.embed_documents()
-> Chroma.from_documents()
-> thư mục Chroma DB đã persist

Flow load runtime:
QaRetriever
-> load_chroma_index()
-> Chroma(..., embedding_function=E5Embeddings)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.clean_qa_chunks import (
    build_qa_chunks,
    convert_to_langchain_documents,
    load_dataset,
    print_summary,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_COLLECTION_NAME = "qa_chunks"
DEFAULT_PERSIST_DIR = "chroma_db_qa"
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


def select_category_balanced_records(
    dataset_records: list[dict[str, Any]],
    sample_size: int | None,
) -> list[dict[str, Any]]:
    """
    Chọn sample nhỏ nhưng vẫn cân bằng category.

    Biến đầu vào:
    - dataset_records: list raw records từ dataset JSON.
    - sample_size: số record muốn lấy để test index nhỏ.

    Ví dụ output:
    sample_size=24 với 12 category -> mỗi category được lấy khoảng 2 records.

    Cách tự viết lại:
    Group records theo category, rồi vòng qua từng category để lấy lần lượt.
    Không nên chỉ lấy N record đầu vì dataset có thể đang grouped theo category.
    """

    if sample_size is None or sample_size >= len(dataset_records):
        return dataset_records

    category_groups: dict[str, list[dict[str, Any]]] = {}
    for record in dataset_records:
        category = str(record.get("category", "unknown"))
        category_groups.setdefault(category, []).append(record)

    selected_records: list[dict[str, Any]] = []
    categories = sorted(category_groups)

    while len(selected_records) < sample_size:
        made_progress = False

        for category in categories:
            records = category_groups[category]
            selected_count_for_category = sum(
                1 for record in selected_records
                if record.get("category", "unknown") == category
            )

            if selected_count_for_category >= len(records):
                continue

            selected_records.append(records[selected_count_for_category])
            made_progress = True

            if len(selected_records) >= sample_size:
                break

        if not made_progress:
            break

    return selected_records


# =============================================================================
# Embedding wrapper
# =============================================================================

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
        # Set HF_TOKEN/HF cache path before importing or initializing the model.
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

        Ví dụ input/output:
        ["Question: ... Answer: ..."] -> [[0.01, -0.02, ...], ...]

        Cách tự viết lại:
        Thêm prefix `passage: ` cho mỗi text rồi gọi embed_documents của model.
        """

        passages = [f"passage: {text}" for text in texts]
        return self.embedding_model.embed_documents(passages)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed query runtime của user.

        Ví dụ output:
        "query: Xe máy là gì?" -> [0.03, 0.12, ...]

        Cách tự viết lại:
        Thêm prefix `query: ` cho câu hỏi rồi gọi embed_query của model.
        """

        return self.embedding_model.embed_query(f"query: {text}")


# =============================================================================
# Dataset preparation
# =============================================================================

def load_clean_documents(
    dataset_path: str | Path,
    sample_size: int | None,
) -> list[Any]:
    """
    Load dataset và chuyển QA chunks sạch thành LangChain Documents.

    Biến đầu vào:
    - dataset_path: đường dẫn `vietnamese_vqa_dataset.json`.
    - sample_size: giới hạn số raw records, không phải số chunks cuối.

    Ví dụ output:
    [Document(page_content="Question: ...", metadata={"category": "am_thuc"})]

    Cách tự viết lại:
    Load raw JSON, chọn sample cân bằng nếu cần, build QA chunks, in summary để
    kiểm tra nhanh, rồi convert sang LangChain Document.
    """

    dataset_records = load_dataset(dataset_path)

    dataset_records = select_category_balanced_records(
        dataset_records=dataset_records,
        sample_size=sample_size,
    )

    qa_chunks = build_qa_chunks(dataset_records)
    print_summary(qa_chunks)

    return convert_to_langchain_documents(qa_chunks)


# =============================================================================
# Chroma build/load
# =============================================================================

def build_chroma_index(
    dataset_path: str | Path,
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cpu",
    sample_size: int | None = None,
) -> Any:
    """
    Build và persist Chroma index mới từ clean QA documents.

    Biến đầu vào:
    - dataset_path: raw JSON dataset.
    - persist_directory: thư mục output Chroma, ví dụ `chroma_db`.
    - collection_name: tên collection, ví dụ `langchain`.
    - model_name/device: embedding model và thiết bị chạy.
    - sample_size: dùng để build test index nhỏ.

    Ví dụ output:
    Một thư mục Chroma DB được tạo trên disk và object Chroma được trả về.

    Cách tự viết lại:
    Load clean documents, khởi tạo E5Embeddings, rồi gọi
    `Chroma.from_documents(..., persist_directory=...)`.
    """

    from langchain_chroma import Chroma

    documents = load_clean_documents(
        dataset_path=dataset_path,
        sample_size=sample_size,
    )
    embeddings = E5Embeddings(model_name=model_name, device=device)

    print("Persist directory:", persist_directory)
    print("Collection:", collection_name)
    print("Embedding model:", model_name)
    print("Device:", device)
    print("Documents to index:", len(documents))

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def load_chroma_index(
    persist_directory: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cpu",
) -> Any:
    """
    Load Chroma index đã tồn tại để search runtime.

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


# =============================================================================
# Retrieval preview
# =============================================================================

def preview_search(vectorstore: Any, query: str, top_k: int = 5) -> None:
    """
    In thử kết quả search để kiểm tra chất lượng index.

    Ví dụ output:
    RANK 1 | SCORE ...
    topic: bánh tét
    category: am_thuc

    Cách tự viết lại:
    Gọi `similarity_search_with_score()`, rồi in metadata quan trọng và một đoạn
    page_content để con người kiểm tra retrieval có đúng không.
    """

    results = vectorstore.similarity_search_with_score(query, k=top_k)

    print("=" * 80)
    print("QUERY:", query)

    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata

        print("=" * 80)
        print(f"RANK {rank} | SCORE {score}")
        print("topic:", metadata.get("topic"))
        print("topic_source:", metadata.get("topic_source"))
        print("keyword:", metadata.get("keyword"))
        print("aliases:", metadata.get("aliases"))
        print("category:", metadata.get("category"))
        print("question_type:", metadata.get("question_type"))
        print("question:", metadata.get("question"))
        print()
        print(document.page_content[:900])
        print()


# =============================================================================
# Command line interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse tham số CLI khi chạy file trực tiếp.

    Ví dụ command:
    python src/ingestion/build_chroma_index.py --dataset vietnamese_vqa_dataset.json

    Cách tự viết lại:
    Dùng argparse, mỗi tham số CLI nên map với một biến của build_chroma_index().
    """

    parser = argparse.ArgumentParser(
        description="Build and preview a Chroma QA index."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to vietnamese_vqa_dataset.json.",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help="Directory where the Chroma index will be saved.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help="HuggingFace embedding model name.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device: cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Raw dataset records to index. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--preview-query",
        default="Ý nghĩa văn hóa của bánh chưng là gì?",
        help="Query to run after building the index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieval results to print.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point CLI: build index rồi preview một query mẫu.

    Ví dụ output:
    In summary số chunks, thông tin persist directory, rồi top-k search results.

    Cách tự viết lại:
    Parse args, đổi sample_size=0 thành None, gọi build_chroma_index(), sau đó
    gọi preview_search() để kiểm tra nhanh.
    """

    args = parse_args()
    sample_size = None if args.sample_size == 0 else args.sample_size

    vectorstore = build_chroma_index(
        dataset_path=args.dataset,
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        device=args.device,
        sample_size=sample_size,
    )

    preview_search(
        vectorstore=vectorstore,
        query=args.preview_query,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

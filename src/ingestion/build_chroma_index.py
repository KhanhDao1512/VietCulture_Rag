"""
Build a Chroma index from clean QA chunks.

Overall flow:
raw JSON dataset -> clean QA chunks -> LangChain Documents -> embeddings -> Chroma DB

Use `--sample-size` for a small local test before building the full index.
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
    Load HF_TOKEN from a local .env file when it is not already in the process.

    This keeps HuggingFace requests authenticated without printing or exposing
    the token value.
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
    Select a small preview dataset with category coverage.

    The raw dataset is grouped by category, so taking the first N records can
    produce a misleading test index that only contains one category.
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
    Wrap HuggingFaceEmbeddings with the prefixes expected by E5 models.

    E5 models are trained with:
    - `passage: ...` for stored documents.
    - `query: ...` for user queries.
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
        """Embed stored QA chunks as E5 passages."""

        passages = [f"passage: {text}" for text in texts]
        return self.embedding_model.embed_documents(passages)

    def embed_query(self, text: str) -> list[float]:
        """Embed a user query as an E5 query."""

        return self.embedding_model.embed_query(f"query: {text}")


# =============================================================================
# Dataset preparation
# =============================================================================

def load_clean_documents(
    dataset_path: str | Path,
    sample_size: int | None,
) -> list[Any]:
    """
    Load the dataset and convert clean QA chunks into LangChain Documents.

    `sample_size` limits raw dataset records, not final chunks. This is useful
    for fast smoke tests before building the full Chroma index.
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
    """Build and persist a new Chroma index from clean QA documents."""

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
    """Load an existing Chroma index for preview/search."""

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
    """Print search results with the metadata needed for human inspection."""

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
    """Parse CLI arguments."""

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
    """Build a Chroma index and immediately preview retrieval results."""

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

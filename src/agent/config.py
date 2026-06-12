"""Configuration helpers for the memory-agent baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentSettings:
    """All runtime settings needed to build the agent graph."""

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


def configure_huggingface_cache(project_root: Path) -> None:
    """Keep Hugging Face model cache inside the project workspace."""

    cache_dir = project_root / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def load_agent_settings(project_root: str | Path | None = None) -> AgentSettings:
    """Load settings from `.env` and environment variables."""

    root = Path(project_root or Path.cwd()).resolve()
    load_dotenv(root / ".env")
    configure_huggingface_cache(root)

    retriever_profile = os.getenv("QA_RETRIEVER_PROFILE", "legacy").strip().lower()
    if retriever_profile == "hybrid":
        default_index_dir = root / "chroma_db_qa_hybrid"
        default_collection_name = "qa_hybrid_chunks"
    else:
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
    )

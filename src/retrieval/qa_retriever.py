"""
Retriever Chroma cho database `chroma_db` được build trên Kaggle.

Mục đích file:
Load thư mục `chroma_db` đã copy từ Kaggle, lấy các QA chunks liên quan bằng
vector search của Chroma, rồi rerank nhẹ bằng keyword/question_type có trong
metadata thật của baseline Kaggle.

Flow runtime:
QaRetriever.retrieve(query)
-> Chroma similarity_search_with_score()
-> score_lexical_match()
-> final_score = vector_score - lexical_weight * lexical_score
-> trả top-k RetrievedQaChunk

Metadata baseline Kaggle đang có:
image_id, category, keyword, normalized_keyword, question_type.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.chroma_store import (
    DEFAULT_EMBED_MODEL,
    load_chroma_index,
)


# =============================================================================
# Cấu hình
# =============================================================================

DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db"
DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_FETCH_MULTIPLIER = 8
DEFAULT_LEXICAL_WEIGHT = 0.06
DEFAULT_NO_LEXICAL_PENALTY = 0.03

VIETNAMESE_STOPWORDS = {
    "anh",
    "bang",
    "ban",
    "biet",
    "cho",
    "co",
    "con",
    "cua",
    "duoc",
    "gi",
    "hay",
    "hien",
    "la",
    "lam",
    "mot",
    "nao",
    "nay",
    "nguoi",
    "nhu",
    "nhung",
    "noi",
    "o",
    "tai",
    "the",
    "thi",
    "toi",
    "trong",
    "ve",
    "voi",
    "y",
    "van",
    "hoa",
    "nghia",
    "mo",
    "ta",
    "so",
    "sanh",
    "va",
}

BASELINE_METADATA_FIELDS_FOR_MATCHING = [
    "keyword",
    "normalized_keyword",
    "category",
    "question_type",
]


# =============================================================================
# Helper lexical matching
# =============================================================================

def normalize_for_matching(text: Any) -> str:
    """
    Normalize text thành dạng không dấu để so khớp lexical.

    Biến đầu vào:
    - text: chuỗi bất kỳ từ query, metadata hoặc page_content.

    Ví dụ output:
    normalize_for_matching("Bánh bao!") -> "banh bao"

    Cách tự viết lại:
    Ép về lowercase, bỏ dấu Unicode, thay ký tự không phải chữ/số bằng khoảng
    trắng rồi collapse khoảng trắng.
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


def extract_query_tokens(query: str) -> list[str]:
    """
    Lấy token quan trọng trong query, bỏ từ quá chung.

    Biến đầu vào:
    - query: câu user hỏi hoặc câu đã được transform_query rewrite.

    Ví dụ output:
    "Ý nghĩa văn hóa của bánh bao là gì?" -> ["banh", "bao"]

    Cách tự viết lại:
    Normalize query, split token, bỏ token quá ngắn và stopwords để lexical
    score tập trung vào object/chủ đề user hỏi.
    """

    normalized_query = normalize_for_matching(query)
    return [
        token
        for token in normalized_query.split()
        if len(token) >= 2 and token not in VIETNAMESE_STOPWORDS
    ]


def extract_query_phrases(query: str) -> list[str]:
    """
    Lấy cụm 2 từ quan trọng từ query.

    Biến đầu vào:
    - query: câu search.

    Ví dụ output:
    "so sánh bánh chưng và bánh tét" -> ["banh chung", "banh tet"]

    Cách tự viết lại:
    Tạo bigram từ query đã normalize. Với object như bánh/áo/nón/xe, giữ cụm 2
    từ vì token đầu thường quá rộng.
    """

    raw_tokens = normalize_for_matching(query).split()
    phrases: list[str] = []

    for index in range(len(raw_tokens) - 1):
        first_token, second_token = raw_tokens[index:index + 2]
        phrase = f"{first_token} {second_token}"
        if all(token not in VIETNAMESE_STOPWORDS for token in phrase.split()):
            phrases.append(phrase)
        if first_token in {"banh", "ao", "non", "xe"} and second_token:
            phrases.append(phrase)

    return list(dict.fromkeys(phrases))


def detect_query_intent(query: str) -> str:
    """
    Nhận diện dạng câu hỏi đơn giản để match `question_type`.

    Biến đầu vào:
    - query: câu search.

    Ví dụ output:
    detect_query_intent("so sánh bánh chưng và bánh tét") -> "comparison"

    Cách tự viết lại:
    Check một số marker rõ như "so sánh", "nguồn gốc", "là gì"; nếu không có
    marker thì trả "general".
    """

    normalized_query = normalize_for_matching(query)
    if any(marker in normalized_query for marker in ["so sanh", "khac nhau", "khac biet"]):
        return "comparison"
    if any(marker in normalized_query for marker in ["nguon goc", "xuat xu", "lich su", "truyen thuyet"]):
        return "origin"
    if any(marker in normalized_query for marker in ["la gi", "day la gi"]):
        return "identification"
    return "general"


def metadata_text_for_matching(metadata: dict[str, Any]) -> str:
    """
    Ghép metadata thật của baseline Kaggle thành text để lexical match.

    Biến đầu vào:
    - metadata: metadata trong Chroma document.

    Ví dụ output:
    {"keyword": "bánh bao", "category": "am_thuc"} -> "bánh bao am_thuc ..."

    Cách tự viết lại:
    Chỉ lấy các field có thật trong Chroma Kaggle baseline. Không dùng
    `topic_card`, `retrieval_anchor`, `canonical_topic` vì database hiện tại
    không tạo các field đó.
    """

    return " ".join(
        str(metadata.get(field_name, ""))
        for field_name in BASELINE_METADATA_FIELDS_FOR_MATCHING
    )


def build_searchable_text(document: Any) -> str:
    """
    Tạo text dùng để so khớp query với một document.

    Biến đầu vào:
    - document: LangChain Document lấy từ Chroma.

    Ví dụ output:
    metadata + page_content -> "banh bao topic banh bao question ..."

    Cách tự viết lại:
    Ghép metadata baseline với page_content rồi normalize. page_content của
    Kaggle chunk đã có Topic, Question và Answer nên đủ thông tin để match.
    """

    metadata = getattr(document, "metadata", {}) or {}
    page_content = getattr(document, "page_content", "")
    return normalize_for_matching(
        metadata_text_for_matching(metadata) + " " + page_content
    )


def score_question_type_match(query: str, metadata: dict[str, Any]) -> float:
    """
    Cộng điểm khi query hợp với `question_type` trong metadata.

    Biến đầu vào:
    - query: câu search.
    - metadata: metadata baseline có field `question_type`.

    Ví dụ output:
    query có "so sánh", question_type="comparison" -> 0.5

    Cách tự viết lại:
    Detect intent từ query, so với question_type. Giữ điểm nhỏ để nó chỉ rerank
    trong các ứng viên Chroma đã lấy, không lấn át vector search.
    """

    query_intent = detect_query_intent(query)
    question_type = normalize_for_matching(metadata.get("question_type", ""))

    if query_intent == "comparison" and question_type == "comparison":
        return 0.5
    if query_intent == "identification" and question_type == "identification":
        return 0.4
    if query_intent == "origin" and question_type in {"origin", "history", "cultural"}:
        return 0.3
    return 0.0


def score_phrase_match(query: str, searchable_text: str) -> float:
    """
    Cộng điểm cho exact phrase match như "bánh bao", "áo dài".

    Biến đầu vào:
    - query: câu search.
    - searchable_text: document text đã normalize.

    Ví dụ output:
    query có "bánh bao" và document có "banh bao" -> 0.8

    Cách tự viết lại:
    Lấy các bigram quan trọng trong query, đếm số phrase xuất hiện trong
    searchable_text rồi cộng điểm vừa phải.
    """

    matched_phrases = [
        phrase
        for phrase in extract_query_phrases(query)
        if phrase in searchable_text
    ]
    if not matched_phrases:
        return 0.0
    return min(0.8, 0.4 * len(matched_phrases))


def score_lexical_match(query: str, document: Any) -> float:
    """
    Tính lexical score phù hợp với Chroma baseline từ Kaggle.

    Điểm gồm:
    - token overlap giữa query và metadata/page_content.
    - exact phrase match cho tên object như bánh bao, áo dài.
    - question_type match cho comparison/identification/origin.

    Ví dụ output:
    score_lexical_match("bánh bao là gì", doc_banh_bao) -> 1.7

    Cách tự viết lại:
    Giữ scoring nhỏ, dễ giải thích và chỉ dựa trên field thật của baseline.
    Nếu sau này build thêm topic_card hoặc metadata mới thì mới mở rộng hàm này.
    """

    query_tokens = extract_query_tokens(query)
    if not query_tokens:
        return 0.0

    metadata = getattr(document, "metadata", {}) or {}
    searchable_text = build_searchable_text(document)
    searchable_tokens = set(searchable_text.split())

    matched_tokens = [
        token
        for token in query_tokens
        if token in searchable_tokens
    ]
    token_overlap_score = len(matched_tokens) / len(query_tokens)
    phrase_score = score_phrase_match(query, searchable_text)
    question_type_score = score_question_type_match(query, metadata)

    return token_overlap_score + phrase_score + question_type_score


# =============================================================================
# Data container
# =============================================================================

@dataclass(frozen=True)
class RetrievedQaChunk:
    """
    Một kết quả retrieval sau rerank.

    Biến:
    - document: LangChain Document lấy từ Chroma.
    - vector_score: distance gốc của Chroma, thấp hơn là tốt hơn.
    - lexical_score: điểm match keyword/question_type, cao hơn là tốt hơn.
    - final_score: điểm sort cuối, thấp hơn là tốt hơn.
    - rank: thứ hạng sau rerank.

    Ví dụ output:
    RetrievedQaChunk(document=Document(...), vector_score=0.42, lexical_score=1.2, final_score=0.35, rank=1)

    Cách tự viết lại:
    Dùng dataclass để debug retrieval dễ hơn thay vì trả mỗi Document trần.
    """

    document: Any
    vector_score: float
    lexical_score: float
    final_score: float
    rank: int

    @property
    def score(self) -> float:
        """
        Alias tương thích với notebook/code cũ.

        Ví dụ output:
        chunk.score -> chunk.final_score

        Cách tự viết lại:
        Trả final_score để các đoạn code cũ gọi `.score` vẫn hoạt động.
        """

        return self.final_score

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Trả metadata của document.

        Ví dụ output:
        {"keyword": "bánh bao", "category": "am_thuc"}

        Cách tự viết lại:
        Expose `self.document.metadata` qua property để code format/debug ngắn hơn.
        """

        return getattr(self.document, "metadata", {}) or {}

    @property
    def page_content(self) -> str:
        """
        Trả text chunk đã lưu trong Chroma.

        Ví dụ output:
        "Topic: bánh bao\\nQuestion: ...\\nAnswer: ..."

        Cách tự viết lại:
        Expose `self.document.page_content` qua property để code graph/UI dùng dễ hơn.
        """

        return getattr(self.document, "page_content", "")


# =============================================================================
# Retriever
# =============================================================================

class QaRetriever:
    """
    Retriever load `chroma_db/langchain` và trả QA chunks đã rerank.

    Ví dụ khởi tạo hiện tại:
    QaRetriever(persist_directory="D:/Ds107/chroma_db", collection_name="langchain")

    Cách tự viết lại:
    Trong __init__, load Chroma bằng embedding E5. Trong retrieve(), lấy nhiều
    candidates bằng vector search, chấm lại lexical, sort rồi trả top_k.
    """

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_name: str = DEFAULT_EMBED_MODEL,
        device: str = "cpu",
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        no_lexical_penalty: float = DEFAULT_NO_LEXICAL_PENALTY,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self.lexical_weight = lexical_weight
        self.no_lexical_penalty = no_lexical_penalty

        self.vectorstore = load_chroma_index(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
            model_name=self.model_name,
            device=self.device,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int | None = None,
        max_score: float | None = None,
        use_rerank: bool = True,
    ) -> list[RetrievedQaChunk]:
        """
        Retrieve top QA chunks cho một query.

        Biến đầu vào:
        - query: câu search.
        - top_k: số chunks cuối cùng trả về.
        - fetch_k: số candidates lấy từ Chroma trước rerank.
        - max_score: ngưỡng lọc final_score, thường để None.
        - use_rerank: bật/tắt lexical rerank.

        Ví dụ output:
        [RetrievedQaChunk(rank=1, metadata={"keyword": "bánh bao"}, ...), ...]

        Cách tự viết lại:
        Gọi Chroma similarity_search_with_score, tính score cho từng candidate,
        sort theo final_score tăng dần, rồi lấy top_k.
        """

        candidate_count = fetch_k or max(top_k * DEFAULT_FETCH_MULTIPLIER, top_k)
        raw_results = self.vectorstore.similarity_search_with_score(
            query,
            k=candidate_count,
        )

        scored_candidates = [
            self._score_candidate(
                query=query,
                document=document,
                vector_score=float(vector_score),
                use_rerank=use_rerank,
            )
            for document, vector_score in raw_results
        ]
        scored_candidates.sort(key=lambda chunk: chunk.final_score)

        reranked_chunks = [
            RetrievedQaChunk(
                document=chunk.document,
                vector_score=chunk.vector_score,
                lexical_score=chunk.lexical_score,
                final_score=chunk.final_score,
                rank=rank,
            )
            for rank, chunk in enumerate(scored_candidates[:top_k], start=1)
        ]

        if max_score is None:
            return reranked_chunks

        return [
            chunk
            for chunk in reranked_chunks
            if chunk.final_score <= max_score
        ]

    def format_context(
        self,
        retrieved_chunks: list[RetrievedQaChunk],
        max_chars_per_chunk: int = 1200,
    ) -> str:
        """
        Format retrieved chunks thành context block cho LLM/debug.

        Biến đầu vào:
        - retrieved_chunks: list RetrievedQaChunk.
        - max_chars_per_chunk: số ký tự tối đa mỗi chunk.

        Ví dụ output:
        [Source 1]
        Keyword: bánh bao
        Question Type: cultural
        ...

        Cách tự viết lại:
        In score + metadata thật của Kaggle baseline, rồi nối page_content đã cắt.
        """

        context_blocks: list[str] = []

        for chunk in retrieved_chunks:
            metadata = chunk.metadata
            clipped_content = chunk.page_content[:max_chars_per_chunk].strip()

            context_blocks.append(
                "\n".join(
                    [
                        f"[Source {chunk.rank}]",
                        f"Final Score: {chunk.final_score:.4f}",
                        f"Vector Score: {chunk.vector_score:.4f}",
                        f"Lexical Score: {chunk.lexical_score:.4f}",
                        f"Keyword: {metadata.get('keyword', '')}",
                        f"Normalized Keyword: {metadata.get('normalized_keyword', '')}",
                        f"Category: {metadata.get('category', '')}",
                        f"Question Type: {metadata.get('question_type', '')}",
                        f"Trace: image_id={metadata.get('image_id', '')}",
                        "",
                        clipped_content,
                    ]
                )
            )

        return "\n\n".join(context_blocks)

    def print_results(
        self,
        query: str,
        retrieved_chunks: list[RetrievedQaChunk],
        content_preview_chars: int = 700,
    ) -> None:
        """
        In kết quả retrieval ra terminal để debug bằng mắt.

        Biến đầu vào:
        - query: query đang test.
        - retrieved_chunks: kết quả retrieve.
        - content_preview_chars: số ký tự preview content.

        Ví dụ output:
        RANK 1 | FINAL 0.123 | VECTOR 0.200 | LEXICAL 1.200

        Cách tự viết lại:
        In index/collection, score, metadata baseline và preview page_content.
        """

        print("=" * 80)
        print("QUERY:", query)
        print("Index:", self.persist_directory)
        print("Collection:", self.collection_name)

        for chunk in retrieved_chunks:
            metadata = chunk.metadata

            print("=" * 80)
            print(
                f"RANK {chunk.rank} | "
                f"FINAL {chunk.final_score:.6f} | "
                f"VECTOR {chunk.vector_score:.6f} | "
                f"LEXICAL {chunk.lexical_score:.3f}"
            )
            print("keyword:", metadata.get("keyword"))
            print("normalized_keyword:", metadata.get("normalized_keyword"))
            print("category:", metadata.get("category"))
            print("question_type:", metadata.get("question_type"))
            print("image_id:", metadata.get("image_id"))
            print()
            print(chunk.page_content[:content_preview_chars])
            print()

    def _score_candidate(
        self,
        query: str,
        document: Any,
        vector_score: float,
        use_rerank: bool,
    ) -> RetrievedQaChunk:
        """
        Kết hợp vector distance và lexical score thành final_score.

        Biến đầu vào:
        - query: query đang retrieve.
        - document: candidate từ Chroma.
        - vector_score: distance gốc của Chroma.
        - use_rerank: False thì giữ nguyên vector_score.

        Ví dụ output:
        vector_score=0.5, lexical_score=1.0, lexical_weight=0.06
        -> final_score=0.44

        Cách tự viết lại:
        Vì Chroma distance thấp hơn là tốt hơn, lexical match tốt thì trừ bớt
        final_score. Nếu query có token mà lexical_score bằng 0 thì cộng penalty nhẹ.
        """

        if not use_rerank:
            return RetrievedQaChunk(
                document=document,
                vector_score=vector_score,
                lexical_score=0.0,
                final_score=vector_score,
                rank=0,
            )

        lexical_score = score_lexical_match(query, document)
        missing_lexical_penalty = (
            self.no_lexical_penalty
            if extract_query_tokens(query) and lexical_score == 0.0
            else 0.0
        )
        final_score = (
            vector_score
            - (self.lexical_weight * lexical_score)
            + missing_lexical_penalty
        )

        return RetrievedQaChunk(
            document=document,
            vector_score=vector_score,
            lexical_score=lexical_score,
            final_score=final_score,
            rank=0,
        )


# =============================================================================
# CLI preview
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse CLI args để preview retrieval.

    Ví dụ command:
    python src/retrieval/qa_retriever.py --query "Bánh bao có ý nghĩa gì?"

    Cách tự viết lại:
    Dùng argparse và expose persist-dir, collection, model, device, query, top-k.
    """

    parser = argparse.ArgumentParser(
        description="Preview retrieval from the Kaggle-built Chroma baseline."
    )
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Directory containing the Chroma index.",
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
        "--query",
        required=True,
        help="Question to retrieve context for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final reranked chunks to return.",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help="Number of vector candidates to fetch before reranking.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=None,
        help="Optional maximum final score. Lower is better.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable lexical reranking and show vector-only results.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the final context block after retrieval preview.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point CLI: load retriever và in kết quả cho một query.

    Ví dụ output:
    QUERY: Bánh bao có ý nghĩa gì?
    RANK 1 | FINAL ...

    Cách tự viết lại:
    Parse args, tạo QaRetriever, gọi retrieve(), rồi print_results().
    """

    args = parse_args()

    retriever = QaRetriever(
        persist_directory=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        device=args.device,
    )
    retrieved_chunks = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        max_score=args.max_score,
        use_rerank=not args.no_rerank,
    )

    retriever.print_results(
        query=args.query,
        retrieved_chunks=retrieved_chunks,
    )

    if args.show_context:
        print("=" * 80)
        print("FORMATTED CONTEXT")
        print("=" * 80)
        print(retriever.format_context(retrieved_chunks))


if __name__ == "__main__":
    main()

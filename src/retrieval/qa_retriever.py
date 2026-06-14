"""
Retriever Chroma cho các QA chunks của VietCulture.

Mục đích file:
Load Chroma index đã build sẵn và lấy các QA chunks liên quan nhất cho RAG hoặc
recommendation cá nhân hóa.

Flow runtime:
QaRetriever.retrieve(query)
-> embed query bằng E5
-> Chroma similarity_search_with_score()
-> score_lexical_match()
-> kết hợp vector score + lexical rerank
-> trả top-k RetrievedQaChunk

Ghi chú project hiện tại:
App mặc định dùng `chroma_db` với collection `langchain` khi
`QA_RETRIEVER_PROFILE=legacy`. Phần chunking/embedding có thể chạy trên Kaggle
hoặc máy GPU, sau đó copy thư mục Chroma về project local.
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

from src.ingestion.build_chroma_index import (
    DEFAULT_EMBED_MODEL,
    E5Embeddings,
    load_chroma_index,
)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PERSIST_DIR = PROJECT_ROOT / "chroma_db_qa_hybrid"
DEFAULT_COLLECTION_NAME = "qa_hybrid_chunks"
DEFAULT_FETCH_MULTIPLIER = 8
DEFAULT_LEXICAL_WEIGHT = 0.06
DEFAULT_NO_LEXICAL_PENALTY = 0.03

# These words are too generic for matching Vietnamese culture topics. They are
# normalized without accents because `normalize_for_matching` strips accents.
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

METADATA_FIELDS_FOR_MATCHING = [
    "retrieval_anchor",
    "canonical_topic",
    "topic",
    "aliases",
    "keyword",
    "normalized_keyword",
    "question",
    "category",
    "question_type",
]


# =============================================================================
# Text matching helpers
# =============================================================================

def normalize_for_matching(text: Any) -> str:
    """
    Normalize text thành token ASCII để so khớp lexical.

    Ví dụ output:
    normalize_for_matching("Thảm cói!") -> "tham coi"

    Cách tự viết lại:
    Lowercase, bỏ dấu Unicode, thay ký tự không phải chữ/số bằng khoảng trắng.
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
    Lấy các token quan trọng trong query, bỏ stopwords.

    Ví dụ output:
    "Y nghia van hoa cua tham coi la gi?" -> ["tham", "coi"]

    Cách tự viết lại:
    Normalize query, split thành token, bỏ token quá ngắn và token trong
    VIETNAMESE_STOPWORDS.
    """

    normalized_query = normalize_for_matching(query)
    tokens = normalized_query.split()

    return [
        token
        for token in tokens
        if len(token) >= 2 and token not in VIETNAMESE_STOPWORDS
    ]


def metadata_text_for_matching(metadata: dict[str, Any]) -> str:
    """
    Ghép các field metadata quan trọng thành text để rerank lexical.

    Ví dụ output:
    "Bánh tét banh tet am_thuc cultural ..."

    Cách tự viết lại:
    Chọn các metadata field liên quan topic/question/category, lấy value và join
    thành một chuỗi.
    """

    field_values = [
        str(metadata.get(field_name, ""))
        for field_name in METADATA_FIELDS_FOR_MATCHING
    ]
    return " ".join(field_values)


def build_searchable_text(document: Any) -> str:
    """
    Tạo text surface để match query với document.

    Ví dụ output:
    metadata + page_content -> "banh tet am thuc y nghia..."

    Cách tự viết lại:
    Ghép metadata_text_for_matching(metadata) với document.page_content, rồi
    normalize_for_matching().
    """

    metadata = document.metadata or {}
    return normalize_for_matching(
        " ".join(
            [
                metadata_text_for_matching(metadata),
                document.page_content,
            ]
        )
    )


def extract_query_phrases(query: str) -> list[str]:
    """
    Lấy các cụm 2 từ quan trọng trong query.

    Ví dụ output:
    "so sánh bánh chưng và bánh tét" -> ["banh chung", "banh tet"]

    Cách tự viết lại:
    Tạo bigram từ token query, bỏ bigram toàn stopword. Với object phổ biến như
    "banh", "ao", "non", "xe", giữ cụm 2 từ vì token đầu thường quá chung.
    """

    normalized_query = normalize_for_matching(query)
    raw_tokens = normalized_query.split()
    phrases: list[str] = []

    for index in range(len(raw_tokens) - 1):
        phrase = " ".join(raw_tokens[index:index + 2])
        if all(token not in VIETNAMESE_STOPWORDS for token in phrase.split()):
            phrases.append(phrase)

    # Food/object names often use a generic first token plus a specific second
    # token, so keep these phrases even though "banh" itself is too broad.
    for index in range(len(raw_tokens) - 1):
        first_token, second_token = raw_tokens[index:index + 2]
        if first_token in {"banh", "ao", "non", "xe"} and second_token:
            phrases.append(f"{first_token} {second_token}")

    return list(dict.fromkeys(phrases))


def detect_query_intent(query: str) -> str:
    """
    Nhận diện dạng câu hỏi để rerank phù hợp.

    Ví dụ output:
    detect_query_intent("So sánh bánh chưng và bánh tét") -> "comparison"

    Cách tự viết lại:
    Normalize query, check các marker như "so sanh", "nguon goc", "la gi", rồi
    trả nhãn intent retrieval đơn giản.
    """

    normalized_query = normalize_for_matching(query)

    if any(phrase in normalized_query for phrase in ["so sanh", "khac nhau", "khac biet"]):
        return "comparison"

    if any(phrase in normalized_query for phrase in ["nguon goc", "xuat xu", "lich su", "truyen thuyet"]):
        return "origin"

    if any(phrase in normalized_query for phrase in ["la gi", "day la gi"]):
        return "identification"

    return "general"


def score_entity_coverage(query: str, document: Any) -> float:
    """
    Cộng điểm cho document bao phủ đủ entity trong query nhiều chủ đề.

    Ví dụ output:
    Query có "bánh chưng" và "bánh tét", document có cả hai -> score 1.2.

    Cách tự viết lại:
    Extract phrase quan trọng từ query, kiểm tra phrase nào xuất hiện trong
    metadata/content. Càng phủ đủ entity thì cộng điểm càng cao.
    """

    query_phrases = extract_query_phrases(query)
    if len(query_phrases) < 2:
        return 0.0

    metadata = document.metadata or {}
    searchable_metadata = normalize_for_matching(metadata_text_for_matching(metadata))
    searchable_text = build_searchable_text(document)
    matched_phrases = [
        phrase
        for phrase in query_phrases
        if phrase in searchable_metadata
    ]

    if len(matched_phrases) == len(query_phrases):
        return 1.2

    if len(matched_phrases) >= 2:
        return 0.8

    # A weaker fallback for chunks whose metadata contains one queried entity
    # and whose content mentions another. This keeps related chunks available
    # without letting off-topic comparison chunks dominate.
    if len(matched_phrases) == 1:
        has_second_phrase_in_content = any(
            phrase in searchable_text and phrase not in matched_phrases
            for phrase in query_phrases
        )
        if has_second_phrase_in_content:
            return 0.4

    return 0.0


def score_intent_match(query: str, document: Any) -> float:
    """
    Cộng điểm nếu document hợp dạng câu hỏi của user.

    Ví dụ output:
    Query dạng comparison và metadata question_type="comparison" -> cộng điểm.

    Cách tự viết lại:
    Detect query intent, đọc metadata question_type, rồi cộng điểm theo rule
    riêng cho comparison/origin/identification.
    """

    metadata = document.metadata or {}
    question_type = str(metadata.get("question_type", ""))
    searchable_metadata = normalize_for_matching(metadata_text_for_matching(metadata))
    searchable_text = build_searchable_text(document)
    query_intent = detect_query_intent(query)
    query_phrases = extract_query_phrases(query)
    has_metadata_topic_match = any(
        phrase in searchable_metadata
        for phrase in query_phrases
    )
    matched_metadata_phrases = [
        phrase
        for phrase in query_phrases
        if phrase in searchable_metadata
    ]
    covers_all_query_phrases = (
        len(query_phrases) >= 2
        and len(matched_metadata_phrases) == len(query_phrases)
    )

    if query_intent == "comparison":
        comparison_score = 0.0
        if question_type == "comparison" and covers_all_query_phrases:
            comparison_score += 0.7
        elif question_type == "comparison" and has_metadata_topic_match:
            comparison_score += 0.15
        elif question_type == "comparison":
            comparison_score += 0.1
        if covers_all_query_phrases and any(term in searchable_text for term in ["khac nhau", "khac biet", "so sanh"]):
            comparison_score += 0.4
        return comparison_score

    if query_intent == "origin":
        origin_score = 0.0
        if question_type in {"origin", "history"}:
            origin_score += 0.8
        if any(term in searchable_text for term in ["nguon goc", "xuat xu", "lang lieu", "truyen thuyet", "vua hung"]):
            origin_score += 0.8
        return origin_score

    if query_intent == "identification" and question_type == "identification":
        return 0.8

    return 0.0


def score_lexical_match(query: str, document: Any) -> float:
    """
    Tính điểm lexical/topic match giữa query và document.

    Điểm gồm:
    - overlap token query với metadata/page_content.
    - exact phrase match trong metadata/content.
    - topic-card boost nhỏ.
    - entity coverage score.
    - intent match score.

    Ví dụ output:
    score_lexical_match("xe máy là gì", doc_xe_may) -> 2.3

    Cách tự viết lại:
    Tính từng điểm con thật dễ giải thích, sau đó cộng lại. Không nên làm quá
    phức tạp nếu mục tiêu là baseline dễ debug.
    """

    query_tokens = extract_query_tokens(query)
    if not query_tokens:
        return 0.0

    metadata = document.metadata or {}
    normalized_metadata = normalize_for_matching(metadata_text_for_matching(metadata))
    normalized_content = normalize_for_matching(document.page_content)

    metadata_tokens = set(normalized_metadata.split())
    content_tokens = set(normalized_content.split())
    matched_tokens = [
        token
        for token in query_tokens
        if token in metadata_tokens or token in content_tokens
    ]

    overlap_score = len(matched_tokens) / len(query_tokens)
    query_phrase = " ".join(query_tokens)

    phrase_score = 0.0
    if query_phrase and query_phrase in normalized_metadata:
        phrase_score += 1.0
    elif query_phrase and query_phrase in normalized_content:
        phrase_score += 0.4

    anchor_text = normalize_for_matching(
        " ".join(
            [
                str(metadata.get("retrieval_anchor", "")),
                str(metadata.get("canonical_topic", "")),
                str(metadata.get("topic", "")),
            ]
        )
    )
    anchor_token_score = 0.5 if any(token in anchor_text for token in query_tokens) else 0.0

    chunk_type = str(metadata.get("chunk_type", ""))
    topic_card_score = 0.15 if chunk_type == "topic_card" else 0.0

    entity_coverage_score = score_entity_coverage(query, document)
    intent_score = score_intent_match(query, document)

    return (
        overlap_score
        + phrase_score
        + anchor_token_score
        + topic_card_score
        + entity_coverage_score
        + intent_score
    )


# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class RetrievedQaChunk:
    """
    Một kết quả retrieval sau khi đã tính score.

    Biến:
    - document: LangChain Document lấy từ Chroma.
    - vector_score: score/distance gốc từ Chroma, thấp hơn là tốt hơn.
    - lexical_score: điểm match keyword/topic do mình tự tính, cao hơn là tốt hơn.
    - final_score: score cuối để sort, thấp hơn là tốt hơn.
    - rank: thứ hạng sau rerank.

    Ví dụ output:
    RetrievedQaChunk(document=Document(...), vector_score=0.42, lexical_score=1.5, final_score=0.33, rank=1)

    Cách tự viết lại:
    Dùng dataclass để gom document và các score lại, giúp debug retrieval dễ hơn
    thay vì chỉ trả Document trần.
    """

    document: Any
    vector_score: float
    lexical_score: float
    final_score: float
    rank: int

    @property
    def score(self) -> float:
        """Backward-compatible score used by older notebook cells."""

        return self.final_score

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Trả metadata của document.

        Ví dụ output:
        {"category": "am_thuc", "topic": "bánh tét"}

        Cách tự viết lại:
        Expose `self.document.metadata` qua property để code debug ngắn hơn.
        """

        return self.document.metadata

    @property
    def page_content(self) -> str:
        """
        Trả nội dung text đã lưu trong Chroma.

        Ví dụ output:
        "Question:\\n...\\nAnswer:\\n..."

        Cách tự viết lại:
        Expose `self.document.page_content` qua property để code format context
        đọc dễ hơn.
        """

        return self.document.page_content


# =============================================================================
# Retriever
# =============================================================================

class QaRetriever:
    """
    Retriever load Chroma index và trả QA chunks đã rerank.

    Ví dụ khởi tạo hiện tại:
    QaRetriever(persist_directory="D:/Ds107/chroma_db", collection_name="langchain")

    Cách tự viết lại:
    Trong __init__, load Chroma với embedding function. Trong retrieve(), search
    rộng bằng vector, chấm lại bằng lexical score, sort và trả top_k.
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
        - max_score: lọc bỏ chunk có final_score quá cao.
        - use_rerank: bật/tắt lexical rerank.

        Ví dụ output:
        [RetrievedQaChunk(rank=1, metadata={"topic": "xe máy"}, ...), ...]

        Cách tự viết lại:
        Gọi Chroma lấy nhiều candidates, tính _score_candidate() cho từng
        document, sort theo final_score tăng dần, rồi lấy top_k.
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
        Format retrieved chunks thành context block cho LLM.

        Ví dụ output:
        [Source 1]
        Final Score: ...
        Category: am_thuc
        ...

        Cách tự viết lại:
        Lặp qua chunks, in score + metadata trace nguồn + page_content đã cắt
        ngắn để prompt không quá dài.
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
                        f"Chunk Type: {metadata.get('chunk_type', '')}",
                        f"Retrieval Anchor: {metadata.get('retrieval_anchor', '')}",
                        f"Canonical Topic: {metadata.get('canonical_topic', '')}",
                        f"Keyword: {metadata.get('keyword', '')}",
                        f"Normalized Keyword: {metadata.get('normalized_keyword', '')}",
                        f"Category: {metadata.get('category', '')}",
                        f"Question Type: {metadata.get('question_type', '')}",
                        f"Question: {metadata.get('question', '')}",
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

        Ví dụ output:
        RANK 1 | FINAL 0.123 | VECTOR 0.200 | LEXICAL 1.200

        Cách tự viết lại:
        In query/index/collection, rồi với từng chunk in score, metadata quan
        trọng và preview page_content.
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
            print("chunk_type:", metadata.get("chunk_type"))
            print("retrieval_anchor:", metadata.get("retrieval_anchor"))
            print("canonical_topic:", metadata.get("canonical_topic"))
            print("topic:", metadata.get("topic"))
            print("keyword:", metadata.get("keyword"))
            print("normalized_keyword:", metadata.get("normalized_keyword"))
            print("aliases:", metadata.get("aliases"))
            print("category:", metadata.get("category"))
            print("question_type:", metadata.get("question_type"))
            print("question:", metadata.get("question"))
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

        Công thức hiện tại:
        final_score = vector_score - lexical_weight * lexical_score + penalty

        Ví dụ output:
        vector_score=0.5, lexical_score=2.0, lexical_weight=0.06
        -> final_score khoảng 0.38

        Cách tự viết lại:
        Vì Chroma distance thấp hơn là tốt hơn, lexical match tốt thì phải làm
        final_score thấp xuống. Nếu không có lexical match thì thêm penalty nhẹ.
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
# Command line preview
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse CLI args để preview retrieval.

    Ví dụ command:
    python src/retrieval/qa_retriever.py --query "Xe máy là gì?"

    Cách tự viết lại:
    Dùng argparse và expose các tham số quan trọng: persist-dir, collection,
    model, device, query, top-k, fetch-k.
    """

    parser = argparse.ArgumentParser(
        description="Preview retrieval from the hybrid Chroma QA index."
    )
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Directory containing the Chroma QA index.",
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
    QUERY: Xe máy là gì?
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

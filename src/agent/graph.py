"""
Pipeline LangGraph cho trợ lý VietCulture cá nhân hóa.

Mục đích file:
File này nối các khối xử lý chính thành một agent hoàn chỉnh:
memory -> routing intent -> retrieval -> Gemini generation -> chấm groundedness.

Flow tổng quát:
invoke_agent()
-> load_memory_node(): đọc memory dài hạn theo user_id
-> route_intent_node(): phân loại tin nhắn mới nhất
-> non_rag_intent_node(): xử lý memory/recommendation/chitchat nếu không cần RAG
-> transform_query(): viết lại câu hỏi để retrieve tốt hơn
-> rag_node(): lấy documents từ Chroma
-> generate(): gọi Gemini sinh câu trả lời từ documents
-> check_hallucination_and_evaluate(): kiểm tra câu trả lời có grounded/useful không
-> update_memory_node(): kết thúc RAG path, không tự ý lưu sở thích mới

Luật memory quan trọng:
Chỉ câu nói rõ sở thích mới được lưu vào long-term memory. Ví dụ "Xe máy là gì?"
không được hiểu thành "user thích giao thông".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agent.config import AgentSettings, load_agent_settings
from src.retrieval.qa_retriever import QaRetriever
from src.routing.routing import (
    build_grounded_recommendation_message,
    build_langgraph_config,
    build_memory_summary,
    build_preference_saved_message,
    build_recommendation_message,
    build_recommendation_queries,
    build_recommendation_query,
    classify_intent_hybrid,
    dump_memory_json,
    extract_memory_updates_from_text,
    load_memory_json,
    load_user_memory,
    merge_memory,
    rank_recommendation_candidates,
    save_user_memory,
)


class GraphState(TypedDict):
    """
    State chung được truyền qua tất cả node của LangGraph.

    Các biến chính:
    - messages: lịch sử message trong thread hiện tại. Tin nhắn user mới nhất
      thường nằm ở `messages[-1]`. LangGraph dùng field này để giữ short-term
      conversation theo `thread_id`.
    - intent: nhãn route sau khi router phân loại, ví dụ `rag_question`,
      `preference_update`, `memory_query`.
    - route_reason: giải thích ngắn vì sao router chọn intent đó, dùng để debug.
    - intent_confidence: độ chắc chắn của router, từ 0.0 đến 1.0.
    - route_source: nguồn quyết định route, ví dụ `rule`, `llm`, `rule_fallback`.
    - memory_update_allowed: chỉ True khi user nói rõ sở thích và được phép lưu.
    - transformed_question: câu hỏi đã được rewrite cho retrieval. Nếu không có
      Gemini thì thường bằng câu hỏi gốc.
    - documents: danh sách LangChain Document lấy từ Chroma, dùng làm context.
    - answer: câu trả lời cuối cùng để UI hiển thị.
    - user_id: khóa định danh long-term memory của từng user.
    - user_preferences: memory của user ở dạng JSON string để dễ đưa vào prompt.
    - retry_count: số lần thử generate/rewrite khi grader thấy answer chưa ổn.

    Ví dụ state sau câu "Xe máy là gì?":
    {
        "intent": "rag_question",
        "transformed_question": "Xe máy trong giao thông Việt Nam",
        "documents": [Document(...), Document(...)],
        "answer": "Xe máy là ...",
        "user_id": "demo_user_a",
    }

    Cách tự viết lại:
    1. Liệt kê mọi dữ liệu mà các node cần dùng chung.
    2. Đặt mỗi dữ liệu thành một field trong TypedDict.
    3. Field nào là message history thì dùng `Annotated[..., add_messages]`.
    4. Giữ state vừa đủ, tránh nhét object quá lớn nếu không cần debug.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    route_reason: str
    intent_confidence: float
    route_source: str
    memory_update_allowed: bool
    transformed_question: str
    documents: list[Any]
    answer: str
    user_id: str
    user_preferences: str
    retry_count: int


class GradeHallucinations(BaseModel):
    """
    Schema để Gemini chấm câu trả lời có bám vào documents không.

    Ví dụ output:
    {"binary_score": true, "reasoning": "Câu trả lời được hỗ trợ bởi tài liệu."}

    Cách tự viết lại:
    Tạo một Pydantic model có field boolean cho quyết định chính và field text
    để model giải thích ngắn. Sau đó dùng `llm.with_structured_output(Model)`.
    """

    binary_score: bool = Field(
        description="True if all main claims are supported by the documents."
    )
    reasoning: str = Field(description="Short Vietnamese explanation.")


class GradeAnswer(BaseModel):
    """
    Schema để Gemini chấm câu trả lời có trả lời đúng câu hỏi gốc không.

    Ví dụ output:
    {"is_relevant": true, "reasoning": "Câu trả lời định nghĩa đúng xe máy."}

    Cách tự viết lại:
    Xác định tiêu chí cần chấm, tạo field boolean cho tiêu chí đó, rồi yêu cầu
    grader trả thêm reasoning để debug khi answer bị đánh rớt.
    """

    is_relevant: bool = Field(
        description="True if the answer directly addresses the original question."
    )
    reasoning: str = Field(description="Short Vietnamese explanation.")


@dataclass
class AgentBundle:
    """
    Gói các object đắt tiền để Streamlit/notebook tái sử dụng.

    Các biến:
    - app: LangGraph đã compile, dùng để `.invoke()`.
    - settings: toàn bộ config runtime.
    - retriever: object load Chroma + embedding model.
    - llm_generate: Gemini sinh answer.
    - llm_grader: Gemini chấm groundedness/usefulness.

    Ví dụ output của create_agent_bundle():
    AgentBundle(app=<CompiledGraph>, settings=AgentSettings(...), retriever=QaRetriever(...))

    Cách tự viết lại:
    Gom những dependency cần khởi tạo một lần vào dataclass, rồi truyền bundle
    này vào UI thay vì khởi tạo model/retriever nhiều lần.
    """

    app: Any
    settings: AgentSettings
    retriever: QaRetriever
    llm_generate: ChatGoogleGenerativeAI | None
    llm_grader: ChatGoogleGenerativeAI | None


def build_context(docs: list[Any]) -> str:
    """
    Ghép các retrieved documents thành một block context cho prompt Gemini.

    Biến đầu vào:
    - docs: list LangChain Document, mỗi Document có `page_content` và `metadata`.

    Ví dụ output rút gọn:
    [Document 1]
    Category: giao_thong
    Question: Xe máy có vai trò gì?

    Answer:
    Xe máy là ...

    Cách tự viết lại:
    Lặp qua từng document, lấy metadata quan trọng để trace nguồn, rồi nối với
    page_content. Nên giữ format rõ ràng để LLM dễ đọc và debug dễ hơn.
    """

    context_blocks: list[str] = []

    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        context_blocks.append(
            "\n".join(
                [
                    f"[Document {index}]",
                    f"Chunk Type: {metadata.get('chunk_type', '')}",
                    f"Retrieval Anchor: {metadata.get('retrieval_anchor', '')}",
                    f"Canonical Topic: {metadata.get('canonical_topic', '')}",
                    f"Keyword: {metadata.get('keyword', '')}",
                    f"Normalized Keyword: {metadata.get('normalized_keyword', '')}",
                    f"Topic: {metadata.get('topic', '')}",
                    f"Category: {metadata.get('category', '')}",
                    f"Question Type: {metadata.get('question_type', '')}",
                    f"Question: {metadata.get('question', '')}",
                    f"Trace: image_id={metadata.get('image_id', '')}",
                    "",
                    getattr(doc, "page_content", ""),
                ]
            )
        )

    return "\n\n".join(context_blocks)


def create_llms(settings: AgentSettings) -> tuple[ChatGoogleGenerativeAI | None, ChatGoogleGenerativeAI | None]:
    """
    Khởi tạo Gemini model cho generation và grading.

    Biến đầu vào:
    - settings.google_api_key: nếu rỗng thì không tạo LLM.
    - settings.gemini_model: tên model Gemini cần dùng.

    Ví dụ output:
    (ChatGoogleGenerativeAI(...), ChatGoogleGenerativeAI(...))
    hoặc (None, None) nếu chưa có GOOGLE_API_KEY.

    Cách tự viết lại:
    Kiểm tra API key trước, tạo một model temperature thấp cho answer và một
    model temperature 0 cho grader để kết quả chấm ổn định hơn.
    """

    if not settings.google_api_key:
        return None, None

    llm_generate = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.2,
        max_retries=2,
    )
    llm_grader = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0,
        max_retries=2,
    )
    return llm_generate, llm_grader


def create_retriever(settings: AgentSettings) -> QaRetriever:
    """
    Load retriever theo config hiện tại.

    Biến đầu vào:
    - settings.persist_dir: thư mục Chroma, hiện tại thường là `chroma_db`.
    - settings.collection_name: collection, hiện tại thường là `langchain`.
    - settings.embed_model/device: model embedding khi query runtime.

    Ví dụ output:
    QaRetriever(persist_directory="D:/Ds107/chroma_db", collection_name="langchain")

    Cách tự viết lại:
    Tạo wrapper retriever riêng, truyền vào đường dẫn Chroma + embedding model,
    rồi để graph gọi một hàm `.retrieve(query)` duy nhất.
    """

    return QaRetriever(
        persist_directory=settings.persist_dir,
        collection_name=settings.collection_name,
        model_name=settings.embed_model,
        device=settings.device,
        lexical_weight=settings.lexical_weight,
    )


def create_agent_bundle(project_root: str | None = None) -> AgentBundle:
    """
    Tạo toàn bộ agent: settings, retriever, LLM, prompt, node và graph.

    Biến đầu vào:
    - project_root: thư mục project. Nếu None thì dùng current working directory.

    Ví dụ output:
    AgentBundle chứa graph đã compile, retriever đã load, settings đã đọc từ `.env`.

    Cách tự viết lại:
    1. Load config.
    2. Khởi tạo retriever và LLM.
    3. Viết prompt/chain.
    4. Định nghĩa từng node nhận state và trả dict update.
    5. Nối node bằng StateGraph.
    6. Compile graph và trả về bundle.
    """

    settings = load_agent_settings(project_root)
    retriever = create_retriever(settings)
    llm_generate, llm_grader = create_llms(settings)

    answer_prompt = ChatPromptTemplate.from_template(
        """You are a professional assistant for Vietnamese culture and tourism.
Rules:
- Respond only in Vietnamese.
- Use only the provided documents to answer the user question.
- If the question asks for a definition, start with a short definition grounded in the documents.
- Do not add outside knowledge. If the documents are insufficient, say: "Tôi không có đủ thông tin để trả lời câu hỏi này".
- Use User Preferences only to personalize focus and tone. Do not use preferences as factual evidence.

User Preferences:
{user_preferences}

Documents:
{context}

Question: {question}

Answer:"""
    )
    rag_chain = None
    if llm_generate is not None:
        rag_chain = answer_prompt | llm_generate | StrOutputParser()

    def retrieve_documents(query: str, top_k: int | None = None) -> list[Any]:
        """
        Lấy documents từ Chroma và bỏ lớp score wrapper.

        Biến đầu vào:
        - query: câu search cuối cùng, có thể là câu gốc hoặc câu đã rewrite.
        - top_k: số document muốn lấy. Nếu None thì dùng settings.top_k.

        Ví dụ output:
        [Document(page_content="...", metadata={"category": "giao_thong"}), ...]

        Cách tự viết lại:
        Gọi retriever.retrieve(), lấy `chunk.document` từ mỗi kết quả, vì node
        generate chỉ cần Document chứ không cần score/rank.
        """

        retrieved_chunks = retriever.retrieve(
            query=query,
            top_k=top_k or settings.top_k,
            fetch_k=settings.fetch_k,
            max_score=settings.retrieval_max_score,
        )
        return [chunk.document for chunk in retrieved_chunks]

    def load_memory_node(state: GraphState) -> dict[str, Any]:
        """
        Node đọc long-term memory theo user_id.

        Biến sử dụng:
        - state["user_id"]: user hiện tại, ví dụ `demo_user_a`.
        - settings.memory_file: file JSON lưu memory.

        Ví dụ output:
        {"user_preferences": "{\"categories\": [\"le_hoi\"], ...}"}

        Cách tự viết lại:
        Lấy user_id từ state, đọc memory store theo user đó, serialize thành
        JSON string rồi trả dict update cho LangGraph.
        """

        user_id = state.get("user_id", "anonymous")
        memory = load_user_memory(settings.memory_file, user_id)
        return {"user_preferences": dump_memory_json(memory)}

    def route_intent_node(state: GraphState) -> dict[str, str]:
        """
        Node phân loại intent của message mới nhất.

        Biến sử dụng:
        - state["messages"]: danh sách message trong thread.
        - latest_message: nội dung tin nhắn cuối cùng của user.

        Ví dụ output:
        {
            "intent": "rag_question",
            "route_reason": "Câu hỏi phù hợp với dataset.",
            "intent_confidence": 0.55,
            "route_source": "rule"
        }

        Cách tự viết lại:
        Lấy `messages[-1].content`, đưa vào hybrid router. Rule router luôn chạy
        trước; LLM chỉ fallback nếu settings bật và confidence thấp.
        """

        messages = state.get("messages", [])
        latest_message = messages[-1].content if messages else ""
        decision = classify_intent_hybrid(
            user_message=latest_message,
            llm=llm_grader or llm_generate,
            use_llm_fallback=settings.use_llm_intent_router,
            confidence_threshold=settings.llm_intent_confidence_threshold,
        )
        return {
            "intent": decision.intent,
            "route_reason": decision.reason,
            "intent_confidence": decision.confidence,
            "route_source": decision.source,
            "memory_update_allowed": decision.memory_update_allowed,
        }

    def route_after_intent(state: GraphState) -> str:
        """
        Quyết định nhánh graph sau khi có intent.

        Biến sử dụng:
        - state["intent"]: nhãn từ router.

        Ví dụ output:
        "needs_rag" nếu intent là `rag_question`.
        "no_rag_needed" nếu intent là `memory_query`.

        Cách tự viết lại:
        Gom các intent cần retrieval vào một set, intent còn lại đi nhánh
        non-RAG. Hàm conditional edge chỉ cần trả tên nhánh dạng string.
        """

        intent = state.get("intent", "rag_question")
        if intent in {"rag_question", "followup_question"}:
            return "needs_rag"
        return "no_rag_needed"

    def non_rag_intent_node(state: GraphState) -> dict[str, Any]:
        """
        Node xử lý các intent không cần RAG answer generation.

        Biến sử dụng:
        - intent: route hiện tại.
        - latest_message: text user vừa nhập.
        - current_memory: memory JSON đã parse thành dict.

        Ví dụ output cho preference_update:
        {
            "answer": "Mình đã lưu lại sở thích này...",
            "user_preferences": "{...}",
            "messages": [AIMessage(...)]
        }

        Cách tự viết lại:
        Tạo từng nhánh if theo intent. Mỗi nhánh trả `answer`, có thể kèm
        `documents` và `user_preferences`. Với preference_update thì phải save
        memory; với recommendation thì build query từ memory rồi retrieve.
        """

        intent = state.get("intent", "chitchat")
        user_id = state.get("user_id", "anonymous")
        messages = state.get("messages", [])
        latest_message = messages[-1].content if messages else ""
        current_memory = load_memory_json(state.get("user_preferences", ""))

        if intent == "preference_update":
            if not state.get("memory_update_allowed", False):
                answer = (
                    "Mình thấy câu này có thể liên quan đến sở thích, nhưng chưa đủ rõ "
                    "để lưu vào memory. Bạn có thể nói kiểu: 'Tôi thích lễ hội' hoặc "
                    "'Tôi quan tâm đến ẩm thực' nhé."
                )
                return {
                    "answer": answer,
                    "messages": [AIMessage(content=answer)],
                }

            memory_updates = extract_memory_updates_from_text(latest_message)
            updated_memory = merge_memory(
                current_memory,
                memory_updates,
                evidence_text=latest_message,
            )
            save_user_memory(settings.memory_file, user_id, updated_memory)
            answer = build_preference_saved_message(updated_memory)
            return {
                "answer": answer,
                "user_preferences": dump_memory_json(updated_memory),
                "messages": [AIMessage(content=answer)],
            }

        if intent == "memory_query":
            answer = build_memory_summary(current_memory)
            documents: list[Any] = []
        elif intent == "recommendation_request":
            recommendation_queries = build_recommendation_queries(current_memory)
            if recommendation_queries:
                retrieved_chunks = []
                seen_doc_keys: set[str] = set()
                per_query_top_k = max(3, settings.top_k)
                for recommendation_query in recommendation_queries:
                    query_chunks = retriever.retrieve(
                        query=recommendation_query,
                        top_k=per_query_top_k,
                        fetch_k=settings.fetch_k,
                    )
                    for chunk in query_chunks:
                        metadata = getattr(chunk.document, "metadata", {}) or {}
                        doc_key = "|".join(
                            [
                                str(metadata.get("category", "")),
                                str(metadata.get("keyword", "")),
                                str(metadata.get("image_id", "")),
                                str(metadata.get("question_type", "")),
                            ]
                        )
                        if doc_key in seen_doc_keys:
                            continue
                        seen_doc_keys.add(doc_key)
                        retrieved_chunks.append(chunk)
                answer = build_grounded_recommendation_message(
                    current_memory,
                    retrieved_chunks,
                )
                selected_chunks = rank_recommendation_candidates(
                    current_memory,
                    retrieved_chunks,
                )[:3]
                documents = [chunk.document for chunk in selected_chunks]
            else:
                answer = build_recommendation_message(current_memory)
                documents = []
        elif intent == "out_of_scope":
            answer = (
                "Mình chỉ có dữ liệu về văn hóa Việt Nam trong dataset hiện tại. "
                "Bạn thử hỏi về ẩm thực, lễ hội, trang phục, giao thông hoặc thủ công mỹ nghệ nhé."
            )
            documents = []
        else:
            answer = "Chào bạn, mình sẵn sàng hỗ trợ các câu hỏi về văn hóa Việt Nam."
            documents = []

        return {
            "answer": answer,
            "documents": documents,
            "messages": [AIMessage(content=answer)],
        }

    def transform_query(state: GraphState) -> dict[str, str]:
        """
        Node rewrite câu hỏi để retrieval tốt hơn.

        Biến sử dụng:
        - original_question: câu user vừa hỏi.
        - user_preferences: memory dùng để cá nhân hóa nhẹ query.
        - retry_count: nếu answer bị chấm chưa ổn, query có thể rewrite lại.

        Ví dụ output:
        {"transformed_question": "vai trò của xe máy trong giao thông Việt Nam"}

        Cách tự viết lại:
        Nếu không có LLM thì trả câu hỏi gốc. Nếu có LLM, viết prompt yêu cầu
        model trả đúng một search query ngắn, không giải thích thêm.
        """

        messages = state.get("messages", [])
        original_question = messages[-1].content if messages else ""
        if llm_generate is None:
            return {"transformed_question": original_question}

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a RAG query optimization expert for Vietnamese culture and tourism.
Rewrite the user's latest question into one concise Vietnamese search query for a vector database.
Resolve references from chat history. Add dataset keywords only when useful.
Return only the rewritten query.""",
                ),
                *messages[:-1],
                (
                    "human",
                    """User Historical Preferences: {user_preferences}
Previous Query Used: {current_search_query}
Current Retry Count: {retry_count}
User's Latest Question: {original_question}
REWRITTEN QUERY FOR DATABASE:""",
                ),
            ]
        )
        chain = prompt | llm_generate | StrOutputParser()
        better_query = chain.invoke(
            {
                "user_preferences": state.get("user_preferences", ""),
                "current_search_query": state.get("transformed_question", ""),
                "retry_count": state.get("retry_count", 0),
                "original_question": original_question,
            }
        ).strip()
        return {"transformed_question": better_query or original_question}

    def rag_node(state: GraphState) -> dict[str, list[Any]]:
        """
        Node lấy documents từ Chroma bằng query đã chuẩn bị.

        Biến sử dụng:
        - state["transformed_question"]: query ưu tiên.
        - messages[-1].content: fallback nếu chưa có transformed_question.

        Ví dụ output:
        {"documents": [Document(...), Document(...)]}

        Cách tự viết lại:
        Chọn search_query, gọi hàm retrieve_documents(), rồi trả list documents
        vào state để node generate dùng làm context.
        """

        messages = state.get("messages", [])
        search_query = state.get("transformed_question") or (
            messages[-1].content if messages else ""
        )
        return {"documents": retrieve_documents(search_query)}

    def generate(state: GraphState) -> dict[str, Any]:
        """
        Node gọi Gemini sinh câu trả lời dựa trên retrieved documents.

        Biến sử dụng:
        - documents: context lấy từ Chroma.
        - original_question: câu hỏi user.
        - user_preferences: memory để chỉnh focus/tone, không dùng làm evidence.
        - retry_count: tăng sau mỗi lần generate.

        Ví dụ output:
        {"answer": "Xe máy là phương tiện...", "retry_count": 1}

        Cách tự viết lại:
        Build context từ documents, đưa context + question vào prompt, gọi LLM.
        Bọc try/except để UI không crash khi API lỗi.
        """

        messages = state.get("messages", [])
        original_question = messages[-1].content if messages else ""
        retry_count = state.get("retry_count", 0) + 1

        if rag_chain is None:
            return {
                "answer": (
                    "Chưa cấu hình LLM để sinh câu trả lời RAG. "
                    "Hãy thiết lập GOOGLE_API_KEY trước khi demo generation."
                ),
                "retry_count": retry_count,
            }

        try:
            answer = rag_chain.invoke(
                {
                    "context": build_context(state.get("documents", [])),
                    "question": original_question,
                    "user_preferences": state.get("user_preferences", ""),
                }
            )
        except Exception as error:
            answer = f"Lỗi khi gọi Gemini: {error}"

        return {"answer": answer, "retry_count": retry_count}

    def check_hallucination(state: GraphState) -> bool:
        """
        Kiểm tra answer có được documents hỗ trợ không.

        Biến sử dụng:
        - documents: nguồn retrieved.
        - answer: câu trả lời vừa generate.

        Ví dụ output:
        True nếu các claim chính có trong documents, False nếu có dấu hiệu bịa.

        Cách tự viết lại:
        Tạo grader prompt, truyền documents + answer, ép LLM trả structured
        output theo schema GradeHallucinations.
        """

        if llm_grader is None:
            return True

        structured_grader = llm_grader.with_structured_output(GradeHallucinations)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a groundedness grader for a RAG system.
Return True only when the answer is supported by the documents.
Write the reasoning in Vietnamese.""",
                ),
                ("human", "DOCUMENTS:\n\n{documents}\n\nANSWER:\n{answer}"),
            ]
        )
        chain = prompt | structured_grader
        try:
            score = chain.invoke(
                {
                    "documents": build_context(state.get("documents", [])),
                    "answer": state.get("answer", ""),
                }
            )
        except Exception:
            return False
        return bool(score.binary_score)

    def evaluate_answer(state: GraphState) -> bool:
        """
        Kiểm tra answer có trả lời đúng câu hỏi gốc không.

        Biến sử dụng:
        - original_question: câu hỏi user.
        - transformed_question: query retrieval, dùng để debug.
        - answer: câu trả lời vừa sinh.

        Ví dụ output:
        True nếu answer giải quyết đúng câu hỏi; False nếu answer lạc đề.

        Cách tự viết lại:
        Tạo grader prompt khác với groundedness: prompt này chấm relevance giữa
        câu hỏi gốc và câu trả lời cuối cùng.
        """

        if llm_grader is None:
            return True

        messages = state.get("messages", [])
        original_question = messages[-1].content if messages else ""
        structured_judge = llm_grader.with_structured_output(GradeAnswer)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a strict answer-quality grader.
Evaluate whether the final answer directly solves the original question.
Write the reasoning in Vietnamese.""",
                ),
                (
                    "human",
                    """ORIGINAL QUESTION: {original_question}
REWRITTEN QUESTION FOR RETRIEVAL: {transformed_question}

FINAL ANSWER:
{answer}""",
                ),
            ]
        )
        chain = prompt | structured_judge
        try:
            result = chain.invoke(
                {
                    "original_question": original_question,
                    "transformed_question": state.get("transformed_question", ""),
                    "answer": state.get("answer", ""),
                }
            )
        except Exception:
            return False
        return bool(result.is_relevant)

    def check_hallucination_and_evaluate(state: GraphState) -> str:
        """
        Chọn bước tiếp theo sau khi generate.

        Biến sử dụng:
        - retry_count: tránh retry vô hạn.
        - check_hallucination(): chấm groundedness.
        - evaluate_answer(): chấm usefulness/relevance.

        Ví dụ output:
        "is_useful" -> đi update_memory/end.
        "is_not_useful" -> quay lại transform query.
        "is_hallucinated" -> generate lại.
        "max_retries" -> kết thúc.

        Cách tự viết lại:
        Đặt max retry trước, rồi kiểm tra groundedness, rồi kiểm tra relevance.
        Mỗi kết quả trả về phải khớp key trong `add_conditional_edges`.
        """

        if state.get("retry_count", 0) >= settings.max_retries:
            return "max_retries"
        if not check_hallucination(state):
            return "is_hallucinated"
        if evaluate_answer(state):
            return "is_useful"
        return "is_not_useful"

    def update_memory_node(state: GraphState) -> dict[str, Any]:
        """
        Kết thúc RAG path mà không tự ý sửa long-term memory.

        Biến sử dụng:
        - answer: câu trả lời cuối cùng.
        - user_preferences: memory hiện tại.

        Ví dụ output:
        {"user_preferences": "{...}", "messages": [AIMessage(content=answer)]}

        Cách tự viết lại:
        Parse memory hiện tại, serialize lại để giữ state sạch, rồi append
        AIMessage. Không gọi save_user_memory ở đây, vì RAG question không phải
        explicit preference.
        """

        current_memory = load_memory_json(state.get("user_preferences", ""))
        return {
            "user_preferences": dump_memory_json(current_memory),
            "messages": [AIMessage(content=state.get("answer", ""))],
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("route_intent", route_intent_node)
    workflow.add_node("non_rag_intent", non_rag_intent_node)
    workflow.add_node("transform", transform_query)
    workflow.add_node("rag", rag_node)
    workflow.add_node("generate", generate)
    workflow.add_node("update_memory", update_memory_node)

    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "route_intent")
    workflow.add_conditional_edges(
        "route_intent",
        route_after_intent,
        {
            "needs_rag": "transform",
            "no_rag_needed": "non_rag_intent",
        },
    )
    workflow.add_edge("non_rag_intent", END)
    workflow.add_edge("transform", "rag")
    workflow.add_edge("rag", "generate")
    workflow.add_edge("update_memory", END)
    workflow.add_conditional_edges(
        "generate",
        check_hallucination_and_evaluate,
        {
            "is_hallucinated": "generate",
            "is_useful": "update_memory",
            "is_not_useful": "transform",
            "max_retries": END,
        },
    )

    app = workflow.compile(checkpointer=MemorySaver())
    return AgentBundle(
        app=app,
        settings=settings,
        retriever=retriever,
        llm_generate=llm_generate,
        llm_grader=llm_grader,
    )


def invoke_agent(
    bundle: AgentBundle,
    user_id: str,
    conversation_id: str,
    message: str,
) -> dict[str, Any]:
    """
    Hàm public để UI/notebook gửi một message vào agent.

    Biến đầu vào:
    - bundle: AgentBundle đã tạo sẵn bằng create_agent_bundle().
    - user_id: khóa long-term memory.
    - conversation_id: khóa short-term thread history.
    - message: tin nhắn user vừa nhập.

    Ví dụ output rút gọn:
    {
        "intent": "memory_query",
        "answer": "Mình đang nhớ bạn quan tâm đến...",
        "documents": []
    }

    Cách tự viết lại:
    Tạo input state ban đầu gồm HumanMessage + user_id, rồi gọi
    `bundle.app.invoke(input_state, config=build_langgraph_config(...))`.
    """

    return bundle.app.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        },
        config=build_langgraph_config(user_id, conversation_id),
    )

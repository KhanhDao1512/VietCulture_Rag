"""LangGraph memory-agent baseline packaged as reusable Python code."""

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
    build_recommendation_query,
    classify_intent,
    dump_memory_json,
    extract_memory_updates_from_text,
    load_memory_json,
    load_user_memory,
    merge_memory,
    save_user_memory,
)


class GraphState(TypedDict):
    """State shared between LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    route_reason: str
    transformed_question: str
    documents: list[Any]
    answer: str
    user_id: str
    user_preferences: str
    retry_count: int


class GradeHallucinations(BaseModel):
    """Binary groundedness score for an answer against retrieved documents."""

    binary_score: bool = Field(
        description="True if all main claims are supported by the documents."
    )
    reasoning: str = Field(description="Short Vietnamese explanation.")


class GradeAnswer(BaseModel):
    """Evaluate whether the answer solves the original question."""

    is_relevant: bool = Field(
        description="True if the answer directly addresses the original question."
    )
    reasoning: str = Field(description="Short Vietnamese explanation.")


@dataclass
class AgentBundle:
    """Objects created once and reused by notebooks or Streamlit."""

    app: Any
    settings: AgentSettings
    retriever: QaRetriever
    llm_generate: ChatGoogleGenerativeAI | None
    llm_grader: ChatGoogleGenerativeAI | None


def build_context(docs: list[Any]) -> str:
    """Format retrieved documents into a readable prompt context."""

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
    """Create Gemini models only when GOOGLE_API_KEY is available."""

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
    """Load the configured Chroma retriever."""

    return QaRetriever(
        persist_directory=settings.persist_dir,
        collection_name=settings.collection_name,
        model_name=settings.embed_model,
        device=settings.device,
        lexical_weight=settings.lexical_weight,
    )


def create_agent_bundle(project_root: str | None = None) -> AgentBundle:
    """Build the full LangGraph app and its shared dependencies."""

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
        """Retrieve plain LangChain documents for graph nodes."""

        retrieved_chunks = retriever.retrieve(
            query=query,
            top_k=top_k or settings.top_k,
            fetch_k=settings.fetch_k,
            max_score=settings.retrieval_max_score,
        )
        return [chunk.document for chunk in retrieved_chunks]

    def load_memory_node(state: GraphState) -> dict[str, Any]:
        """Load long-term memory for the current user."""

        user_id = state.get("user_id", "anonymous")
        memory = load_user_memory(settings.memory_file, user_id)
        return {"user_preferences": dump_memory_json(memory)}

    def route_intent_node(state: GraphState) -> dict[str, str]:
        """Classify latest user message before deciding whether RAG is needed."""

        messages = state.get("messages", [])
        latest_message = messages[-1].content if messages else ""
        decision = classify_intent(latest_message)
        return {
            "intent": decision.intent,
            "route_reason": decision.reason,
        }

    def route_after_intent(state: GraphState) -> str:
        """Map intent labels to graph branches."""

        intent = state.get("intent", "rag_question")
        if intent in {"rag_question", "followup_question"}:
            return "needs_rag"
        return "no_rag_needed"

    def non_rag_intent_node(state: GraphState) -> dict[str, Any]:
        """Handle memory, recommendation, chitchat, and out-of-scope intents."""

        intent = state.get("intent", "chitchat")
        user_id = state.get("user_id", "anonymous")
        messages = state.get("messages", [])
        latest_message = messages[-1].content if messages else ""
        current_memory = load_memory_json(state.get("user_preferences", ""))

        if intent == "preference_update":
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
            recommendation_query = build_recommendation_query(current_memory)
            if recommendation_query:
                retrieved_chunks = retriever.retrieve(
                    query=recommendation_query,
                    top_k=3,
                    fetch_k=settings.fetch_k,
                )
                answer = build_grounded_recommendation_message(
                    current_memory,
                    retrieved_chunks,
                )
                documents = [chunk.document for chunk in retrieved_chunks]
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
        """Rewrite latest user question for retrieval when Gemini is available."""

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
        """Retrieve documents using transformed query, with raw-question fallback."""

        messages = state.get("messages", [])
        search_query = state.get("transformed_question") or (
            messages[-1].content if messages else ""
        )
        return {"documents": retrieve_documents(search_query)}

    def generate(state: GraphState) -> dict[str, Any]:
        """Generate an answer from retrieved documents."""

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
        """Check whether generated answer is grounded in retrieved documents."""

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
        """Check whether the answer solves the original user question."""

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
        """Route after generation based on grading result."""

        if state.get("retry_count", 0) >= settings.max_retries:
            return "max_retries"
        if not check_hallucination(state):
            return "is_hallucinated"
        if evaluate_answer(state):
            return "is_useful"
        return "is_not_useful"

    def update_memory_node(state: GraphState) -> dict[str, Any]:
        """
        Keep explicit preferences separate from ordinary RAG questions.

        A user asking "Xe máy là gì?" is not the same as saying
        "Tôi thích giao thông". For the Streamlit/demo module, long-term
        preference memory is updated only in the `preference_update` branch.
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
    """Invoke the graph with the standard multi-user config."""

    return bundle.app.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
        },
        config=build_langgraph_config(user_id, conversation_id),
    )

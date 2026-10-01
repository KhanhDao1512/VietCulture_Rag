"""Unit tests for deterministic routing, memory, normalization, and deduplication."""

from __future__ import annotations

import unittest
import re
from unittest.mock import patch
from types import SimpleNamespace

from src.agent.graph import deduplicate_answer_paragraphs, invoke_agent_safely
from src.agent.config import load_agent_settings
from src.memory.store import merge_memory
from src.memory.conversation_store import (
    append_message,
    choose_conversation_id,
    create_conversation,
    list_conversations,
    load_messages,
)
from src.normalization.canonical_names import (
    NameMapping,
    NameMappingBatch,
    normalize_topic_names,
    validate_name_mappings,
)
from src.recommendation.recommender import (
    build_grounded_recommendation_message,
    rank_recommendation_candidates,
)
from src.retrieval.qa_retriever import RetrievedQaChunk, deduplicate_scored_candidates
from src.routing.intent_router import classify_intent


def fake_document_chunk(category: str, topic: str, image_id: str, question_type: str = "cultural"):
    document = SimpleNamespace(
        metadata={
            "category": category,
            "canonical_topic": topic,
            "image_id": image_id,
            "question_type": question_type,
        },
        page_content=f"Question: {topic} có ý nghĩa gì?\nAnswer: {topic} là chủ đề văn hóa.",
    )
    return SimpleNamespace(document=document)


class RoutingTests(unittest.TestCase):
    def test_routes_memory_query_before_rag(self):
        decision = classify_intent("sở thích của tôi là gì")
        self.assertEqual(decision.intent, "memory_query")
        self.assertFalse(decision.memory_update_allowed)

    def test_only_explicit_interest_allows_memory_update(self):
        explicit = classify_intent("Tôi thích lễ hội")
        factual = classify_intent("Lễ hội Chùa Hương là gì?")
        self.assertEqual(explicit.intent, "preference_update")
        self.assertTrue(explicit.memory_update_allowed)
        self.assertFalse(factual.memory_update_allowed)


class MemoryTests(unittest.TestCase):
    def test_merge_memory_deduplicates_and_preserves_evidence(self):
        merged = merge_memory(
            {"categories": ["am_thuc"], "topics": ["bún chả"], "evidence": []},
            {"categories": ["am_thuc"], "topics": ["bún chả", "bánh chưng"]},
            evidence_text="Tôi thích bún chả và bánh chưng",
        )
        self.assertEqual(merged["categories"], ["am_thuc"])
        self.assertEqual(merged["topics"], ["bún chả", "bánh chưng"])
        self.assertEqual(merged["evidence"], ["Tôi thích bún chả và bánh chưng"])


class AgentSettingsTests(unittest.TestCase):
    def test_recommendation_count_is_clamped_and_pool_cannot_be_smaller(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_directory:
            with patch.dict(
                "os.environ",
                {
                    "QA_RECOMMENDATION_COUNT": "8",
                    "QA_RECOMMENDATION_CANDIDATE_POOL": "2",
                },
            ):
                settings = load_agent_settings(Path(temporary_directory))
        self.assertEqual(settings.recommendation_count, 5)
        self.assertEqual(settings.recommendation_candidate_pool, 5)


class NameNormalizationTests(unittest.TestCase):
    def test_known_aliases_normalize_without_llm(self):
        names = normalize_topic_names(["Bun cha Ha Noi"], llm=None)
        self.assertEqual(names["Bun cha Ha Noi"], "Bún chả Hà Nội")

    def test_llm_mapping_rejects_unknown_source_and_low_confidence(self):
        candidates = ["Hoi Lim", "Bun cha Ha Noi"]
        proposals = [
            NameMapping(source_name="Hoi Lim", display_name="Hội Lim", confidence=0.9),
            NameMapping(source_name="Bun cha Ha Noi", display_name="Bún chả Hà Nội", confidence=0.5),
            NameMapping(source_name="invented", display_name="Tên bịa", confidence=0.99),
        ]
        accepted = validate_name_mappings(candidates, proposals)
        self.assertEqual(accepted, {"Hoi Lim": "Hội Lim"})

    def test_english_topic_gets_contextual_vietnamese_name(self):
        class FakeLlm:
            def with_structured_output(self, schema):
                self.schema = schema
                return self

            def invoke(self, prompt):
                self.prompt = prompt
                return NameMappingBatch(
                    mappings=[
                        NameMapping(
                            source_name="lacquer Viet Nam",
                            display_name="Sơn mài Việt Nam",
                            confidence=0.96,
                        )
                    ]
                )

        llm = FakeLlm()
        names = normalize_topic_names(
            ["lacquer Viet Nam"],
            llm,
            contexts={"lacquer Viet Nam": "category=thu_cong_my_nghe; QA về tranh sơn mài"},
            normalize_all=True,
        )
        self.assertEqual(names, {"lacquer Viet Nam": "Sơn mài Việt Nam"})
        self.assertIn("QA về tranh sơn mài", llm.prompt)

    def test_english_or_rejected_topic_is_not_displayed(self):
        proposals = [
            NameMapping(source_name="silk painting", display_name="Silk painting", confidence=0.99),
            NameMapping(
                source_name="lam huong",
                display_name="Làm hương",
                confidence=0.99,
                is_valid_topic=False,
            ),
        ]
        self.assertEqual(
            validate_name_mappings(["silk painting", "lam huong"], proposals),
            {},
        )

    def test_normalization_timeout_does_not_prevent_source_name_fallback(self):
        class FailingLlm:
            def with_structured_output(self, schema):
                raise TimeoutError("normalizer timeout")

        names = normalize_topic_names(
            ["silk painting"], FailingLlm(), normalize_all=True
        )
        answer = build_grounded_recommendation_message(
            memory={"categories": ["thu_cong_my_nghe"]},
            retrieved_chunks=[
                fake_document_chunk("thu_cong_my_nghe", "silk painting", "000001")
            ],
            display_names=names,
        )
        self.assertIn("1. Silk painting", answer)
        self.assertIn("ý nghĩa gì", answer)


class RecommendationCountTests(unittest.TestCase):
    def test_zero_one_three_and_five_retrieved_topics(self):
        for result_count in (0, 1, 3, 5):
            with self.subTest(result_count=result_count):
                chunks = [
                    fake_document_chunk(
                        "kien_truc", f"Công trình kiến trúc số {index}", f"{index:06}"
                    )
                    for index in range(1, result_count + 1)
                ]
                answer = build_grounded_recommendation_message(
                    memory={"categories": ["kien_truc"]},
                    retrieved_chunks=chunks,
                    max_recommendations=5,
                )
                rendered_count = len(re.findall(r"(?m)^\d+\. ", answer))
                self.assertEqual(rendered_count, result_count)

    def test_partial_name_mapping_keeps_unmapped_candidate(self):
        chunks = [
            fake_document_chunk("kien_truc", "Cầu Thê Húc", "000001"),
            fake_document_chunk("kien_truc", "Hoi An ancient town", "000002"),
        ]
        answer = build_grounded_recommendation_message(
            memory={"categories": ["kien_truc"]},
            retrieved_chunks=chunks,
            display_names={"Cầu Thê Húc": "Cầu Thê Húc"},
            max_recommendations=5,
        )
        self.assertIn("1. Cầu Thê Húc", answer)
        self.assertIn("2. Hoi An ancient town", answer)

    def test_formatter_deduplicates_same_topic_and_fills_to_five(self):
        chunks = [
            fake_document_chunk("kien_truc", "Cầu Thê Húc", f"{index:06}")
            for index in range(1, 4)
        ] + [
            fake_document_chunk("kien_truc", f"Địa điểm số {index}", f"{index + 10:06}")
            for index in range(1, 5)
        ]
        answer = build_grounded_recommendation_message(
            memory={"categories": ["kien_truc"]},
            retrieved_chunks=chunks,
            max_recommendations=5,
        )
        self.assertEqual(len(re.findall(r"(?m)^\d+\. ", answer)), 5)
        self.assertEqual(
            len(re.findall(r"(?m)^\d+\. Cầu Thê Húc$", answer)),
            1,
        )

    def test_ui_cards_use_same_five_item_limit_and_keep_fallback_titles(self):
        from streamlit_app import build_recommendation_cards_html

        documents = []
        for index in range(1, 7):
            documents.append(
                SimpleNamespace(
                    metadata={
                        "category": "kien_truc",
                        "display_topic": f"Topic {index}",
                        "topic": f"Topic {index}",
                        "keyword": f"Topic {index}",
                        "image_id": f"{index:06}",
                    },
                    page_content="",
                )
            )
        cards = build_recommendation_cards_html(documents, max_cards=5)
        self.assertEqual(cards.count('class="vc-rec-card"'), 5)
        self.assertIn("Topic 1", cards)


class ConversationStoreTests(unittest.TestCase):
    def test_active_conversation_prefers_url_and_keeps_per_tab_selection(self):
        conversation_ids = ["chat_latest", "chat_older"]
        first_tab = choose_conversation_id(
            conversation_ids, requested_id="chat_older", stored_id="chat_latest"
        )
        second_tab = choose_conversation_id(
            conversation_ids, requested_id="chat_latest", stored_id="chat_older"
        )
        self.assertEqual(first_tab, "chat_older")
        self.assertEqual(second_tab, "chat_latest")
        self.assertEqual(
            choose_conversation_id(conversation_ids, requested_id="unknown"),
            "chat_latest",
        )

    def test_transcripts_are_persistent_and_separated_by_user(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "chat.sqlite3"
            create_conversation(database_path, "user_a", "thread_1")
            append_message(database_path, "user_a", "thread_1", "user", "Tôi thích lễ hội")
            append_message(
                database_path,
                "user_a",
                "thread_1",
                "assistant",
                "Mình đã ghi nhớ.",
                "<div>cards</div>",
            )
            create_conversation(database_path, "user_b", "thread_2")

            conversations = list_conversations(database_path, "user_a")
            messages = load_messages(database_path, "user_a", "thread_1")
            other_user_messages = load_messages(database_path, "user_b", "thread_1")

        self.assertEqual(conversations[0]["title"], "Tôi thích lễ hội")
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(messages[1]["recommendation_cards_html"], "<div>cards</div>")
        self.assertEqual(other_user_messages, [])


class LangGraphHistoryRestoreTests(unittest.TestCase):
    def test_replays_history_only_when_thread_checkpoint_is_empty(self):
        from src.agent.graph import invoke_agent

        class FakeApp:
            def __init__(self):
                self.stored_messages = []

            def get_state(self, config):
                return SimpleNamespace(values={"messages": self.stored_messages})

            def invoke(self, input_state, config):
                return input_state

        app = FakeApp()
        bundle = SimpleNamespace(app=app)
        result = invoke_agent(
            bundle,
            "user_a",
            "thread_1",
            "Câu hỏi mới",
            history=[
                {"role": "user", "content": "Tôi thích lễ hội"},
                {"role": "assistant", "content": "Mình đã ghi nhớ."},
            ],
        )
        self.assertEqual(
            [message.content for message in result["messages"]],
            ["Tôi thích lễ hội", "Mình đã ghi nhớ.", "Câu hỏi mới"],
        )

        app.stored_messages = [SimpleNamespace(content="đã có checkpoint")]
        result = invoke_agent(bundle, "user_a", "thread_1", "Lượt kế tiếp", history=[])
        self.assertEqual([message.content for message in result["messages"]], ["Lượt kế tiếp"])

    def test_agent_exception_returns_a_persistable_failure_turn(self):
        with self.assertLogs("src.agent.graph", level="ERROR"):
            with patch(
                "src.agent.graph.invoke_agent", side_effect=RuntimeError("api down")
            ):
                result = invoke_agent_safely(
                    SimpleNamespace(), "user_a", "thread_1", "Câu hỏi được giữ lại"
                )
        self.assertEqual(result["intent"], "agent_error")
        self.assertEqual(result["agent_error"], "RuntimeError")
        self.assertIn("Câu hỏi vẫn được lưu", result["answer"])


class RecommendationNameTests(unittest.TestCase):
    def test_validated_name_is_preferred_and_failed_name_uses_source_fallback(self):
        chunks = [
            fake_document_chunk("thu_cong_my_nghe", "lacquer Vietnam", "000001"),
            fake_document_chunk("thu_cong_my_nghe", "silk painting", "000002"),
        ]
        answer = build_grounded_recommendation_message(
            memory={"categories": ["thu_cong_my_nghe"]},
            retrieved_chunks=chunks,
            requested_categories=["thu_cong_my_nghe"],
            display_names={"lacquer Vietnam": "Sơn mài Việt Nam"},
        )
        self.assertIn("Sơn mài Việt Nam", answer)
        self.assertIn("Silk painting", answer)
        self.assertEqual(len(re.findall(r"(?m)^\d+\. ", answer)), 2)

class DuplicateHandlingTests(unittest.TestCase):
    def test_reranker_candidates_remove_exact_duplicate_content(self):
        document_one = SimpleNamespace(page_content="Bánh chưng là món ăn ngày Tết.")
        document_two = SimpleNamespace(page_content="Banh chung la mon an ngay Tet!")
        document_three = SimpleNamespace(page_content="Lễ hội gắn với sinh hoạt cộng đồng.")
        candidates = [
            RetrievedQaChunk(document_one, 0.2, 0.8, 0.15, 1),
            RetrievedQaChunk(document_two, 0.3, 0.6, 0.25, 2),
            RetrievedQaChunk(document_three, 0.4, 0.1, 0.39, 3),
        ]
        deduplicated = deduplicate_scored_candidates(candidates)
        self.assertEqual(len(deduplicated), 2)
        self.assertIs(deduplicated[0].document, document_one)

    def test_recommendations_deduplicate_topic_across_images(self):
        chunks = [
            fake_document_chunk("am_thuc", "Bánh chưng", "000001", "cultural"),
            fake_document_chunk("am_thuc", "Banh chung", "000002", "analysis"),
            fake_document_chunk("am_thuc", "Bánh tét", "000003", "cultural"),
        ]
        selected = rank_recommendation_candidates(
            memory={"categories": ["am_thuc"]},
            retrieved_chunks=chunks,
        )
        self.assertEqual(len(selected), 2)

    def test_answer_removes_exact_duplicate_paragraph_but_keeps_distinct_content(self):
        answer = (
            "Bánh chưng gắn với Tết.\n\n"
            "Banh chung gan voi Tet!\n\n"
            "Bánh chưng còn là dịp để gia đình sum họp."
        )
        cleaned = deduplicate_answer_paragraphs(answer)
        self.assertEqual(cleaned.count("Bánh chưng gắn với Tết."), 1)
        self.assertIn("gia đình sum họp", cleaned)


if __name__ == "__main__":
    unittest.main()

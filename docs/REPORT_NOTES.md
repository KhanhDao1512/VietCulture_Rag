# Gợi Ý Viết Báo Cáo

File này là dàn ý để viết báo cáo cho project.

## 1. Đặt Vấn Đề

Project xây dựng một trợ lý hỏi đáp về văn hóa Việt Nam.

Trợ lý cần:

- Trả lời câu hỏi dựa trên dataset.
- Ghi nhớ sở thích rõ ràng của người dùng.
- Gợi ý chủ đề phù hợp với sở thích.
- Hỗ trợ nhiều người dùng khác nhau.

Vấn đề quan trọng: không phải câu nào cũng nên đi qua RAG.

Ví dụ:

```text
Xe máy là gì?              -> RAG
Tôi thích lễ hội           -> update memory
Tôi thích gì?              -> query memory
Gợi ý cho tôi chủ đề hay   -> recommendation
```

## 2. Mục Tiêu

Mục tiêu của project:

1. Xây dựng baseline RAG cho câu hỏi văn hóa Việt Nam.
2. Thêm intent router trước RAG.
3. Thêm long-term memory theo `user_id`.
4. Thêm recommendation dựa trên memory nhưng vẫn grounded bằng dataset.
5. Demo multi-user bằng Streamlit.

## 3. Dữ Liệu Và Retrieval

Dataset gồm các câu hỏi/trả lời về văn hóa Việt Nam.

Hệ thống dùng Chroma làm vector database.

Retriever hiện tại:

- Tìm kiếm bằng embedding.
- Kết hợp metadata/topic matching.
- Rerank bằng lexical score.

File liên quan:

```text
src/retrieval/qa_retriever.py
src/ingestion/clean_qa_chunks.py
src/ingestion/build_chroma_index.py
```

## 4. LangGraph Pipeline

Các node chính:

```text
load_memory
route_intent
non_rag_intent
transform
rag
generate
update_memory
```

Luồng RAG:

```text
load_memory -> route_intent -> transform -> rag -> generate -> grade -> update_memory
```

Luồng không cần RAG:

```text
load_memory -> route_intent -> non_rag_intent
```

## 5. Intent Router

Các intent đang hỗ trợ:

```text
rag_question
followup_question
recommendation_request
preference_update
memory_query
chitchat
out_of_scope
```

Router giúp hệ thống biết khi nào cần retrieve, khi nào cần update memory, khi nào chỉ cần đọc memory.

## 6. Memory

Memory lưu theo `user_id`.

Ví dụ:

```json
{
  "demo_user_a": {
    "categories": ["le_hoi"],
    "topics": ["lễ hội"],
    "keywords": ["lễ hội"]
  }
}
```

Quy tắc quan trọng:

```text
Chỉ lưu memory khi user nói rõ sở thích.
```

Ví dụ:

- `Tôi thích giao thông` -> lưu.
- `Xe máy là gì?` -> không lưu thành sở thích.

## 7. Multi-User

Hệ thống tách:

- `user_id`: long-term memory.
- `conversation_id`: một cuộc hội thoại.
- `thread_id`: lịch sử ngắn hạn của LangGraph.

Format:

```python
thread_id = f"user:{user_id}:thread:{conversation_id}"
```

Nhờ vậy nhiều user không bị trộn memory và history.

## 8. Recommendation

Recommendation dùng cả memory và retrieval:

```text
saved preferences -> build recommendation query -> retrieve chunks -> grounded recommendation
```

Câu trả lời recommendation có:

- chủ đề,
- category,
- dạng câu hỏi,
- lý do phù hợp với sở thích,
- câu hỏi nên thử,
- nội dung gợi ý từ dataset.

## 9. Evaluation

Baseline evaluator kiểm tra:

- intent router,
- retrieval top-k.

Kết quả hiện tại:

```text
Intent:    5/5
Retrieval: 4/4
Total:     9/9
```

## 10. Demo

Prompt demo:

```text
Tôi thích lễ hội và ẩm thực
Tôi thích gì?
Gợi ý cho tôi vài chủ đề phù hợp
Xe máy là gì?
```

Demo multi-user:

1. Chọn `demo_user_a`, nhập sở thích.
2. Chọn `demo_user_b`, nhập sở thích khác.
3. Hỏi `Tôi thích gì?` ở từng user.
4. Kết quả phải khác nhau.

## 11. Hạn Chế

- Memory đang dùng JSON, chưa phải database.
- Chroma DB chưa upload lên GitHub do dung lượng lớn.
- Metadata cũ có thể lẫn tiếng Anh, tiếng Việt không dấu.
- Baseline phù hợp demo/học thuật, chưa phải production.

## 12. Hướng Phát Triển

- Chuyển memory từ JSON sang SQLite.
- Tách `explicit_preferences`, `recent_topics`, `conversation_summary`.
- Chuẩn hóa canonical topic trong Chroma index.
- Tạo evaluation set lớn hơn.
- Deploy với vector database và database memory ổn định hơn.

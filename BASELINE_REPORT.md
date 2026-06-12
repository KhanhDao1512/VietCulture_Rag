# Báo Cáo Baseline v1

Ngày cập nhật: 2026-06-12

## Phạm Vi Baseline

Baseline hiện tại là một trợ lý hỏi đáp văn hóa Việt Nam có:

- Intent router trước RAG.
- Long-term memory theo `user_id`.
- Short-term conversation theo `thread_id`.
- Retrieval từ Chroma.
- Gemini sinh câu trả lời dựa trên tài liệu retrieve được.
- Recommendation dựa trên sở thích đã lưu và grounded bằng dataset.
- Streamlit demo để test multi-user.

## Index Đang Dùng

Mặc định project dùng Chroma index cũ:

```text
Persist directory: D:\Ds107\chroma_db
Collection: langchain
Embedding model: intfloat/multilingual-e5-base
Device: cpu
```

Hybrid index vẫn giữ lại để thử nghiệm:

```text
Persist directory: D:\Ds107\chroma_db_qa_hybrid
Collection: qa_hybrid_chunks
```

## Flow Chính

RAG path:

```text
load_memory
-> route_intent
-> transform_query
-> rag
-> generate
-> groundedness/usefulness check
-> update_memory
```

Non-RAG path:

```text
load_memory
-> route_intent
-> non_rag_intent
```

## Intent Đang Hỗ Trợ

```text
rag_question
followup_question
recommendation_request
preference_update
memory_query
chitchat
out_of_scope
```

## Quy Ước Multi-User

Long-term memory:

```python
user_id = "demo_user_a"
```

Short-term thread:

```python
thread_id = f"user:{user_id}:thread:{conversation_id}"
```

Helper:

```python
build_langgraph_config(user_id, conversation_id)
```

## Chính Sách Memory

Baseline hiện tại chỉ lưu memory khi user nói rõ sở thích.

Ví dụ:

```text
Tôi thích lễ hội -> lưu vào memory.
Xe máy là gì?    -> không lưu thành sở thích.
```

Điều này giúp profile người dùng sạch hơn và tránh hiểu nhầm rằng user thích mọi chủ đề họ từng hỏi.

## Test Notebook

Chạy theo thứ tự:

```text
test_00_setup
test_01_preference_update
test_02_memory_query
test_03_recommendation
test_04_direct_retrieval
test_05_rag_smoke
test_06_cleanup
```

## Test Evaluator

Command:

```powershell
D:\anaconda\envs\rag\python.exe src\evaluation\baseline_eval.py --persist-dir D:\Ds107\chroma_db --collection langchain --device cpu --top-k 5 --fetch-k 40
```

Kết quả hiện tại:

```text
Intent:    5/5
Retrieval: 4/4
Total:     9/9
```

## Streamlit Demo

Command:

```powershell
D:\anaconda\envs\rag\python.exe -m streamlit run streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

URL:

```text
http://127.0.0.1:8501
```

## Hạn Chế

- Memory vẫn là JSON, chưa phù hợp cho nhiều request ghi đồng thời.
- Chroma DB và dataset lớn không được commit lên GitHub.
- Metadata trong index cũ còn một số keyword tiếng Anh/không dấu.
- Đây là baseline demo/học thuật, chưa phải production system.

## Hướng Phát Triển

- Chuyển memory sang SQLite.
- Chuẩn hóa canonical topic trong index.
- Tạo bộ evaluation lớn hơn.
- Deploy với storage bền vững cho Chroma và memory.

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/KhanhDao1512/VietCulture_Rag"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    QA_CHROMA_DIR=/app/vector_index \
    QA_CHROMA_COLLECTION=vietculture_topic_qa_v1 \
    QA_RETRIEVER_DEVICE=cpu \
    CONVERSATION_DB=/app/runtime/conversations.sqlite3 \
    MEMORY_FILE=/app/runtime/user_memories.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY .streamlit/ ./.streamlit/
COPY assets/ ./assets/
COPY src/ ./src/
COPY streamlit_app.py ./streamlit_app.py
COPY vector_index/ ./vector_index/

RUN mkdir -p /app/runtime /app/.cache/huggingface

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

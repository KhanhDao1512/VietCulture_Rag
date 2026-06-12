"""Streamlit demo for the Memory Agent baseline."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.agent.graph import create_agent_bundle, invoke_agent
from src.routing.routing import build_langgraph_config


st.set_page_config(
    page_title="Memory Agent Baseline",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading retriever and graph...")
def load_bundle():
    """Create the agent once per Streamlit process."""

    return create_agent_bundle()


def load_memory_db(memory_file: Path) -> dict:
    """Read the JSON memory file for display."""

    if not memory_file.exists():
        return {}
    try:
        return json.loads(memory_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_memory_db(memory_file: Path, memory_db: dict) -> None:
    """Write the JSON memory file after cleanup."""

    memory_file.write_text(
        json.dumps(memory_db, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reset_user_memory(memory_file: Path, user_id: str) -> None:
    """Remove one user's long-term memory."""

    memory_db = load_memory_db(memory_file)
    memory_db.pop(user_id, None)
    save_memory_db(memory_file, memory_db)


def build_chat_history_key(user_id: str, conversation_id: str) -> str:
    """Keep Streamlit chat history separate for each user/thread pair."""

    return f"chat_history::{user_id}::{conversation_id}"


def render_debug_state(state: dict) -> None:
    """Show routing and retrieval details for demo/debugging."""

    with st.expander("Debug state", expanded=False):
        st.write("Intent:", state.get("intent"))
        st.write("Route reason:", state.get("route_reason"))
        st.write("Transformed question:", state.get("transformed_question"))
        st.write("Retrieved documents:", len(state.get("documents", [])))
        st.code(state.get("user_preferences", "{}"), language="json")


bundle = load_bundle()
settings = bundle.settings

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1080px;
        padding-top: 1.5rem;
    }
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }
    .app-title {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        color: #64748b;
        margin-bottom: 1.25rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-pill-warning {
        background: #fef3c7;
        color: #92400e;
    }
    .sidebar-note {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Memory Agent Baseline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Intent router, RAG, recommendation, and user memory demo.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("User Profile")
    user_id = st.text_input("User ID", value="demo_user_a")
    conversation_id = st.text_input("Conversation", value="demo_thread")
    thread_id = build_langgraph_config(user_id, conversation_id)["configurable"]["thread_id"]
    st.caption(f"Thread: `{thread_id}`")

    chat_history_key = build_chat_history_key(user_id, conversation_id)

    col_a, col_b = st.columns(2)
    if col_a.button("Reset memory", use_container_width=True):
        reset_user_memory(settings.memory_file, user_id)
        st.session_state.pop(chat_history_key, None)
        st.session_state.pop("last_state", None)
        st.success(f"Reset memory for {user_id}")
    if col_b.button("Clear chat", use_container_width=True):
        st.session_state.pop(chat_history_key, None)
        st.success("Cleared this chat")

    st.markdown("---")
    if bundle.llm_generate:
        st.markdown('<span class="status-pill">Gemini ready</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="status-pill status-pill-warning">Gemini missing</span>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="sidebar-note">Long-term memory is keyed by User ID. '
        'Chat display is keyed by User ID + Conversation.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Saved Memory")
    memory_db = load_memory_db(settings.memory_file)
    current_memory = memory_db.get(user_id, {})
    if current_memory:
        st.json(current_memory)
    else:
        st.caption("No saved preference yet.")

if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []

for message in st.session_state[chat_history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Nhập câu hỏi hoặc sở thích của bạn...")

if prompt:
    st.session_state[chat_history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent đang xử lý..."):
            state = invoke_agent(
                bundle=bundle,
                user_id=user_id,
                conversation_id=conversation_id,
                message=prompt,
            )
        answer = state.get("answer", "")
        st.markdown(answer)
        render_debug_state(state)

    st.session_state[chat_history_key].append({"role": "assistant", "content": answer})
    st.session_state.last_state = state

st.divider()
with st.expander("Demo prompts", expanded=False):
    st.code(
        "\n".join(
            [
                "Tôi thích lễ hội và ẩm thực",
                "Tôi thích gì?",
                "Gợi ý cho tôi vài chủ đề phù hợp",
                "Xe máy là gì?",
            ]
        )
    )

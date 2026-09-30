"""Lưu danh sách hội thoại và tin nhắn local bằng SQLite.

Mục đích file:
Giúp sidebar liệt kê và khôi phục lịch sử theo từng user/conversation.

Luồng xử lý:
initialize_store() -> create_conversation() -> append_message()
-> list_conversations()/load_messages().

SQLite chỉ dùng cho demo local một máy; user_id ở đây là nhãn phân vùng dữ liệu,
không phải hệ thống xác thực người dùng.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any, Iterator


NEW_CONVERSATION_TITLE = "Cuộc trò chuyện mới"


def _utc_now() -> str:
    """Trả timestamp UTC dạng ISO để sắp xếp hội thoại nhất quán."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(database_path: str | Path) -> sqlite3.Connection:
    """Tạo connection ngắn hạn và bảo đảm thư mục database tồn tại."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection(database_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Đóng connection rõ ràng sau mỗi thao tác, kể cả khi có lỗi."""

    connection = _connect(database_path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_store(database_path: str | Path) -> None:
    """Tạo schema hội thoại/tin nhắn nếu database chưa có."""

    with _connection(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, conversation_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                recommendation_cards_html TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id, conversation_id)
                    REFERENCES conversations(user_id, conversation_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(user_id, conversation_id, id);
            """
        )


def create_conversation(
    database_path: str | Path,
    user_id: str,
    conversation_id: str,
    title: str = NEW_CONVERSATION_TITLE,
) -> None:
    """Tạo hội thoại rỗng; lệnh lặp lại cùng ID không làm mất dữ liệu."""

    initialize_store(database_path)
    now = _utc_now()
    with _connection(database_path) as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO conversations
                (user_id, conversation_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, conversation_id, title, now, now),
        )


def list_conversations(
    database_path: str | Path,
    user_id: str,
    limit: int = 100,
) -> list[dict[str, str]]:
    """Lấy danh sách hội thoại mới cập nhật nhất của một user."""

    initialize_store(database_path)
    with _connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT conversation_id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (user_id, max(1, limit)),
        ).fetchall()
    return [dict(row) for row in rows]


def load_messages(
    database_path: str | Path,
    user_id: str,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """Đọc transcript và recommendation cards của một hội thoại."""

    initialize_store(database_path)
    with _connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT role, content, recommendation_cards_html, created_at
            FROM messages
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY id
            """,
            (user_id, conversation_id),
        ).fetchall()
    return [dict(row) for row in rows]


def append_message(
    database_path: str | Path,
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    recommendation_cards_html: str = "",
) -> None:
    """Thêm message và đổi title từ câu user đầu tiên nếu đang dùng title mặc định."""

    if role not in {"user", "assistant"}:
        raise ValueError("role phải là 'user' hoặc 'assistant'")

    create_conversation(database_path, user_id, conversation_id)
    now = _utc_now()
    with _connection(database_path) as connection:
        connection.execute(
            """
            INSERT INTO messages
                (user_id, conversation_id, role, content, recommendation_cards_html, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, conversation_id, role, content, recommendation_cards_html, now),
        )
        if role == "user":
            title = " ".join(content.split())[:48].strip() or NEW_CONVERSATION_TITLE
            if len(" ".join(content.split())) > 48:
                title = title.rstrip(" .,!?;:") + "…"
            connection.execute(
                """
                UPDATE conversations
                SET title = CASE
                        WHEN title = ? THEN ?
                        ELSE title
                    END,
                    updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (NEW_CONVERSATION_TITLE, title, now, user_id, conversation_id),
            )
        else:
            connection.execute(
                """
                UPDATE conversations SET updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (now, user_id, conversation_id),
            )

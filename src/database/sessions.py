import time
import sqlite3
import uuid
from config import CHAT_DB_PATH


def init_chat_db(db_path: str = CHAT_DB_PATH):
    """Initialize SQLite tables for chat sessions and messages.

    Args:
        db_path (str): SQLite database path.

    Returns:
        None: This function does not return a value.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at REAL,
            updated_at REAL
        )
    """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            ts REAL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """
    )
    conn.commit()
    conn.close()


def db_conn():
    """Create a new SQLite connection to the chat database.

    Args:
        None.

    Returns:
        sqlite3.Connection: Open database connection.
    """

    return sqlite3.connect(CHAT_DB_PATH)


def list_sessions() -> list[tuple[str, str]]:
    """List sessions ordered by most recent update time.

    Args:
        None.

    Returns:
        list[tuple[str, str]]: List of (session_id, title) tuples.
    """

    res = list()

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, COALESCE(title,'(no title)') FROM sessions ORDER BY updated_at DESC")
    rows = cur.fetchall()
    conn.close()

    res = [(r[0], r[1]) for r in rows]

    return res


def create_session(title: str = "New Chat") -> str:
    """Create a new chat session row.

    Args:
        title (str): Initial session title.

    Returns:
        str: Newly created session UUID.
    """

    sid = str(uuid.uuid4())
    now = time.time()
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (sid, title, now, now),
    )
    conn.commit()
    conn.close()

    return sid


def delete_session(session_id: str):
    """Delete a session and its messages.

    Args:
        session_id (str): Session UUID to delete.

    Returns:
        None: This function does not return a value.
    """

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def touch_session(session_id: str):
    """Update a session's updated_at timestamp.

    Args:
        session_id (str): Session UUID to update.

    Returns:
        None: This function does not return a value.
    """

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time(), session_id))
    conn.commit()
    conn.close()


def load_chat(session_id: str) -> list[dict[str, str]]:
    """Load all messages in a session sorted by timestamp.

    Args:
        session_id (str): Session UUID.

    Returns:
        list[dict[str, str]]: Message list in {\"role\", \"content\"} format.
    """

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY ts ASC", (session_id,))
    rows = cur.fetchall()
    conn.close()

    res: list[dict[str, str]] = [{"role": role, "content": content} for role, content in rows]

    return res


def maybe_set_title(session_id: str, first_user_message: str):
    """Set a session title from the first user message when still default.

    Args:
        session_id (str): Session UUID.
        first_user_message (str): Candidate title source text.

    Returns:
        None: This function does not return a value.
    """

    # set title only if default
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return
    title = row[0] or ""
    if title.strip() in ("New Chat", "(no title)", ""):
        new_title = first_user_message.strip().replace("\n", " ")
        if len(new_title) > 40:
            new_title = new_title[:40] + "…"
        cur.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
    conn.close()


def add_message(session_id: str, role: str, content: str):
    """Insert a chat message and refresh session activity time.

    Args:
        session_id (str): Session UUID.
        role (str): Message role such as user or assistant.
        content (str): Message body.

    Returns:
        None: This function does not return a value.
    """

    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (id, session_id, role, content, ts) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), session_id, role, content, time.time()),
    )
    conn.commit()
    conn.close()
    touch_session(session_id)


def refresh_session_choices() -> list[tuple[str, str]]:
    """Build Gradio dropdown choices from stored sessions.

    Args:
        None.

    Returns:
        list[tuple[str, str]]: List of (title, session_id) tuples.
    """

    sessions = list_sessions()
    # label show title, value is id
    return [(title, sid) for sid, title in sessions]

import json
import time
from pathlib import Path
from config import INSUFFICIENT_MSG, QA_CHAIN
from database.sessions import create_session, list_sessions, load_chat, delete_session, add_message, maybe_set_title
from database.vector_db import answer_from_db, add_pdf_to_db, auto_fetch_and_ingest, VECTOR_DB
from retreiver.pdf_utils import ensure_pdf
import gradio as gr

NO_IMPROVEMENT_FETCH_CACHE: set[tuple[str, str]] = set()


def _normalize_question_key(text: str) -> str:
    """Normalize question text for cache key generation.

    Args:
        text (str): Raw question text.

    Returns:
        str: Lowercased and whitespace-normalized string.
    """

    return " ".join((text or "").strip().lower().split())


def _build_fetch_status_note(fetch_result: str) -> str:
    """Build a user-visible status message from fetch result markers.

    Args:
        fetch_result (str): Result marker returned by ``auto_fetch_and_ingest``.

    Returns:
        str: Status text for UI.
    """

    if fetch_result.startswith("pdf_added:"):
        path = fetch_result.split("pdf_added:", 1)[1]
        return f"Paper added: {Path(path).name}"
    if fetch_result.startswith("pdf_existing:"):
        path = fetch_result.split("pdf_existing:", 1)[1]
        return f"Paper already indexed: {Path(path).name}"
    if fetch_result.startswith("pubmed:"):
        added = fetch_result.split("pubmed:", 1)[1]
        return f"PubMed abstracts added: {added}"
    return f"Fetch result: {fetch_result}"


def print_help():
    """Print CLI help text for supported commands.

    Args:
        None.

    Returns:
        None: This function does not return a value.
    """

    print("\nHelp:")
    print("  /add <pdf_path>        : Add PDF on Vector DB")
    print("  /where <json_filter>   : Apply meta data filter on next question (Chroma filter grammar)")
    print("  /clearwhere            : Clear filter")
    print("  /exit                  : Exit")
    print("\nIn-line filter is also available:")
    print('  예) "Summerize the conclusion of this paper @file=paper.pdf"')
    print('  예) "Experiment setting on p3? @file=paper.pdf @page=3"')
    print('  예) "..." @filter={"filename":{"$eq":"paper"}}')


def refresh_session_choices() -> list[tuple[str, str]]:
    """Build Gradio dropdown choices from stored sessions.

    Args:
        None.

    Returns:
        list[tuple[str, str]]: List of ``(title, session_id)`` tuples.
    """

    sessions = list_sessions()
    # label show title, value is id
    return [(title, sid) for sid, title in sessions]


def ui_new_chat() -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], str, str]:
    """Create a new chat session and return initial UI state.

    Args:
        None.

    Returns:
        tuple[str, list[tuple[str, str]], list[tuple[str, str]], str, str]:
        Session id, updated choices, empty chat list, empty input text, and status.
    """

    sid = create_session("New Chat")
    choices = refresh_session_choices()
    return sid, choices, [], "", "New Chat just started"


def ui_select_chat(session_id: str) -> tuple[list[dict[str, str]], str]:
    """Load chat history for a selected session.

    Args:
        session_id (str): Selected session UUID.

    Returns:
        tuple[list[dict[str, str]], str]: Loaded chat messages and status message.
    """

    if not session_id:
        return [], "Please choose session"
    chat = load_chat(session_id)
    return chat, "Chat history is loaded"


def ui_delete_chat(session_id: str) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], str]:
    """Delete a session and create a fresh replacement session.

    Args:
        session_id (str): Session UUID to delete.

    Returns:
        tuple[str, list[tuple[str, str]], list[tuple[str, str]], str]:
        New session id, updated choices, empty chat list, and status message.
    """

    if not session_id:
        return "", refresh_session_choices(), [], "There is no session to delete"
    cache_keys = [key for key in NO_IMPROVEMENT_FETCH_CACHE if key[0] == session_id]
    for key in cache_keys:
        NO_IMPROVEMENT_FETCH_CACHE.discard(key)
    delete_session(session_id)
    # create a new one
    new_id = create_session("New Chat")
    return new_id, refresh_session_choices(), [], "Delete current chat and start a new chatting"


def ui_upload_pdfs(session_id: str, files: list[gr.File]) -> str:
    """Validate and ingest uploaded PDF files into the vector database.

    Args:
        session_id (str): Active session UUID.
        files (list[gr.File]): Uploaded Gradio file objects.

    Returns:
        str: Multi-line status message for processed files.
    """

    if not files:
        return "We don't have any uploaded file"
    if not session_id:
        session_id = create_session("New Chat")

    msgs = []
    for f in files:
        try:
            p = ensure_pdf(f.name)
            if p:
                n = add_pdf_to_db(VECTOR_DB, f.name)
                msgs.append(f"{f.name} Added (chunks={n})")
            else:
                print("pdf format is not right")
        except Exception as e:
            msgs.append(f"{getattr(f,'name','(unknown)')} filed to add: {repr(e)}")

    # store a system-like message to history
    add_message(session_id, "assistant", "\n".join(msgs))
    return "\n".join(msgs)


def stream_text(text: str, delay: float = 0.001):
    """Yield text progressively for streaming UI updates.

    Args:
        text (str): Target text to stream.
        delay (float): Sleep duration between yielded chunks.

    Returns:
        Iterator[str]: Generator that emits incrementally built text.

    Yields:
        str: Incrementally growing text buffer.
    """

    buffer = list()
    for ch in text:
        buffer.append(ch)
        yield "".join(buffer)
        time.sleep(delay)


def ui_send(
    session_id: str,
    chat: list[tuple[str, str]],
    user_text: str,
    session_filter_json: str,
):
    """Handle a user message and stream assistant responses to the UI.

    Args:
        session_id (str): Active session UUID.
        chat (list[tuple[str, str]]): Current chat state.
        user_text (str): Raw user message.
        session_filter_json (str): Optional JSON filter string.

    Yields:
        tuple[str, list[dict[str, str]], str, str]: Updated session id, chat state,
        status text, and textbox value.
    """

    user_text = (user_text or "").strip()
    if not user_text:
        yield session_id, chat, "Please ask a question."
        return

    if not session_id:
        session_id = create_session("New Chat")

    # parse session filter json if any
    session_filter = None
    sf = (session_filter_json or "").strip()
    if sf:
        try:
            session_filter = json.loads(sf)
        except Exception:
            yield session_id, chat, 'Failed to parse Session filter JSON . e.g. {"filename": {"$eq": "paper"}}'
            return

    # append user message
    chat = chat or []
    chat.append({"role": "user", "content": user_text})
    chat.append({"role": "assistant", "content": ""})
    yield session_id, chat, "Searching...", ""

    add_message(session_id, "user", user_text)
    maybe_set_title(session_id, user_text)

    # answer from DB
    ans, _, _ = answer_from_db(VECTOR_DB, QA_CHAIN, user_text, session_filter=session_filter)

    # if insufficient -> fetch paper -> retry once
    if ans == INSUFFICIENT_MSG:
        question_key = _normalize_question_key(user_text)
        fetch_cache_key = (session_id, question_key)

        if fetch_cache_key in NO_IMPROVEMENT_FETCH_CACHE:
            final = (
                f"{INSUFFICIENT_MSG}\n\n"
                "(Skipped repeated web search because the same question previously did not improve after retrieval.)"
            )
            for part in stream_text(final, delay=0.004):
                chat[-1]["content"] = part
                yield session_id, chat, "Skipped repeated web search", ""
            add_message(session_id, "assistant", final)
            yield session_id, chat, "Skipped repeated web search", ""
            return

        # show interim to user
        interim = f"{INSUFFICIENT_MSG}\n\n(Retry to generate the answer after adding web retrieved documents since we need more evidence in vector DB.)"
        for part in stream_text(interim, delay=0.005):
            chat[-1]["content"] = part
            yield session_id, chat, "Web Search/Downloading...", ""
        add_message(session_id, "assistant", interim)

        downloaded = auto_fetch_and_ingest(VECTOR_DB, user_text)
        if downloaded:
            note = _build_fetch_status_note(downloaded)
            yield session_id, chat, note, ""
            add_message(session_id, "assistant", note)

            ans2, _, _ = answer_from_db(VECTOR_DB, QA_CHAIN, user_text, session_filter=session_filter)
            for part in stream_text(ans2, delay=0.004):
                chat[-1]["content"] = part
                yield session_id, chat, "Generating the answer...", ""
            add_message(session_id, "assistant", ans2)

            if ans2 == INSUFFICIENT_MSG:
                NO_IMPROVEMENT_FETCH_CACHE.add(fetch_cache_key)
            else:
                NO_IMPROVEMENT_FETCH_CACHE.discard(fetch_cache_key)

            yield session_id, chat, "Done", ""
            return
        else:
            final = f"{INSUFFICIENT_MSG}\n\n(We could not find any appropriate document on Web.)"
            for part in stream_text(final, delay=0.004):
                chat[-1]["content"] = part
                yield session_id, chat, "Web retrieving failed", ""

            add_message(session_id, "assistant", final)
            NO_IMPROVEMENT_FETCH_CACHE.add(fetch_cache_key)
            yield session_id, chat, "Web retrieving failed", ""
            return

    # normal answer
    for part in stream_text(ans, delay=0.004):
        chat[-1]["content"] = part
        yield session_id, chat, "Generating the answer...", ""

    add_message(session_id, "assistant", ans)
    yield session_id, chat, "Done", ""

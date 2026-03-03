import json
import time
from pathlib import Path
from config import INSUFFICIENT_MSG, QA_CHAIN
from database.sessions import create_session, list_sessions, load_chat, delete_session, add_message, maybe_set_title
from database.vector_db import answer_from_db, add_pdf_to_db, auto_fetch_and_ingest, VECTOR_DB
from retreiver.pdf_utils import ensure_pdf
import gradio as gr


def print_help():
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
    """Gradio choices: list of (label, value) or (value, label) depending on component.
    We'll use Dropdown with choices as list of tuples (label, value).
    """
    sessions = list_sessions()
    # label show title, value is id
    return [(title, sid) for sid, title in sessions]


def ui_new_chat() -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], str, str]:
    sid = create_session("New Chat")
    choices = refresh_session_choices()
    return sid, choices, [], "", "New Chat just started"


def ui_select_chat(session_id: str) -> tuple[list[dict[str, str]], str]:
    if not session_id:
        return [], "Please choose session"
    chat = load_chat(session_id)
    return chat, "Chat history is loaded"


def ui_delete_chat(session_id: str) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], str]:
    if not session_id:
        return "", refresh_session_choices(), [], "There is no session to delete"
    delete_session(session_id)
    # create a new one
    new_id = create_session("New Chat")
    return new_id, refresh_session_choices(), [], "Delete current chat and start a new chatting"


def ui_upload_pdfs(session_id: str, files: list[gr.File]) -> str:
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
    """
    Returns: (session_id, updated_chat, status)
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
        # show interim to user
        interim = f"{INSUFFICIENT_MSG}\n\n(Retry to generate the answer after adding web retrieved documents since we need more evidence in vector DB.)"
        for part in stream_text(interim, delay=0.005):
            chat[-1]["content"] = part
            yield session_id, chat, "Web Search/Downloading...", ""
        add_message(session_id, "assistant", interim)

        downloaded = auto_fetch_and_ingest(VECTOR_DB, user_text)
        if downloaded:
            try:
                fname = Path(downloaded).name
            except Exception:
                fname = str(downloaded)
            note = f"Paper added: {fname}"
            # 노트는 상태로만 보여줘도 되고, 채팅에 남겨도 됨(여기선 상태로만)
            yield session_id, chat, note, ""
            add_message(session_id, "assistant", note)

            ans2, _, _ = answer_from_db(VECTOR_DB, QA_CHAIN, user_text, session_filter=session_filter)
            for part in stream_text(ans2, delay=0.004):
                chat[-1]["content"] = part
                yield session_id, chat, "Generating the answer...", ""
            add_message(session_id, "assistant", ans2)
            yield session_id, chat, "Done", ""
            return
        else:
            final = f"{INSUFFICIENT_MSG}\n\n(We could not find any appropriate document on Web.)"
            for part in stream_text(final, delay=0.004):
                chat[-1]["content"] = part
                yield session_id, chat, "Web retrieving failed", ""

            add_message(session_id, "assistant", final)
            yield session_id, chat, "Web retrieving failed", ""
            return

    # normal answer
    for part in stream_text(ans, delay=0.004):
        chat[-1]["content"] = part
        yield session_id, chat, "Generating the answer...", ""

    add_message(session_id, "assistant", ans)
    yield session_id, chat, "Done", ""

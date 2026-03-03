import json
import time
from pathlib import Path
from config import INSUFFICIENT_MSG, QA_CHAIN
from database.sessions import create_session, list_sessions, load_chat, delete_session, add_message, maybe_set_title
from database.vector_db import answer_from_db, add_pdf_to_db, auto_fetch_and_ingest, VECTOR_DB
from retreiver.pdf_utils import ensure_pdf
import gradio as gr


def print_help():
    print("\n명령 도움말:")
    print("  /add <pdf_path>        : PDF를 벡터DB에 추가")
    print("  /where <json_filter>   : 다음 질문에 메타데이터 필터 적용 (Chroma filter 문법)")
    print("  /clearwhere            : 필터 해제")
    print("  /exit                  : 종료")
    print("\n질문에 인라인 필터도 가능:")
    print('  예) "이 논문 결론 요약해줘 @file=paper.pdf"')
    print('  예) "p3에서 실험 세팅? @file=paper.pdf @page=3"')
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
    return sid, choices, [], "", "새 채팅을 시작했습니다."


def ui_select_chat(session_id: str) -> tuple[list[dict[str, str]], str]:
    if not session_id:
        return [], "세션이 선택되지 않았습니다."
    chat = load_chat(session_id)
    return chat, "채팅 히스토리를 불러왔습니다."


def ui_delete_chat(session_id: str) -> tuple[str, list[tuple[str, str]], list[tuple[str, str]], str]:
    if not session_id:
        return "", refresh_session_choices(), [], "삭제할 세션이 없습니다."
    delete_session(session_id)
    # create a new one
    new_id = create_session("New Chat")
    return new_id, refresh_session_choices(), [], "채팅을 삭제하고 새 채팅을 시작했습니다."


def ui_upload_pdfs(session_id: str, files: list[gr.File]) -> str:
    if not files:
        return "업로드된 파일이 없습니다."
    if not session_id:
        session_id = create_session("New Chat")

    msgs = []
    for f in files:
        try:
            p = ensure_pdf(f.name)
            if p:
                n = add_pdf_to_db(VECTOR_DB, f.name)
                msgs.append(f"{f.name} 추가 완료 (chunks={n})")
            else:
                print("pdf가 올바르지 않습니다.")
        except Exception as e:
            msgs.append(f"{getattr(f,'name','(unknown)')} 추가 실패: {repr(e)}")

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
        yield session_id, chat, "입력된 질문이 없습니다."
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
            yield session_id, chat, 'Session filter JSON 파싱 실패. 예: {"filename": {"$eq": "paper"}}'
            return

    # append user message
    chat = chat or []
    chat.append({"role": "user", "content": user_text})
    chat.append({"role": "assistant", "content": ""})
    yield session_id, chat, "검색 중...", ""

    add_message(session_id, "user", user_text)
    maybe_set_title(session_id, user_text)

    # answer from DB
    ans, _, _ = answer_from_db(VECTOR_DB, QA_CHAIN, user_text, session_filter=session_filter)

    # if insufficient -> fetch paper -> retry once
    if ans == INSUFFICIENT_MSG:
        # show interim to user
        interim = f"{INSUFFICIENT_MSG}\n\n(벡터DB에 근거가 부족해 Web에서 관련 논문을 자동 검색/추가 후 재시도합니다.)"
        for part in stream_text(interim, delay=0.005):
            chat[-1]["content"] = part
            yield session_id, chat, "Web 검색/다운로드 중...", ""
        add_message(session_id, "assistant", interim)

        downloaded = auto_fetch_and_ingest(VECTOR_DB, user_text)
        if downloaded:
            try:
                fname = Path(downloaded).name
            except Exception:
                fname = str(downloaded)
            note = f"논문 추가됨: {fname}"
            # 노트는 상태로만 보여줘도 되고, 채팅에 남겨도 됨(여기선 상태로만)
            yield session_id, chat, note, ""
            add_message(session_id, "assistant", note)

            ans2, _, _ = answer_from_db(VECTOR_DB, QA_CHAIN, user_text, session_filter=session_filter)
            for part in stream_text(ans2, delay=0.004):
                chat[-1]["content"] = part
                yield session_id, chat, "답변 생성 중...", ""
            add_message(session_id, "assistant", ans2)
            yield session_id, chat, "답변 완료", ""
            return
        else:
            final = f"{INSUFFICIENT_MSG}\n\n(Web에서 적절한 PDF를 가져오지 못했습니다.)"
            for part in stream_text(final, delay=0.004):
                chat[-1]["content"] = part
                yield session_id, chat, "Web 자동 수집 실패", ""

            add_message(session_id, "assistant", final)
            yield session_id, chat, "Web 자동 수집 실패", ""
            return

    # normal answer
    for part in stream_text(ans, delay=0.004):
        chat[-1]["content"] = part
        yield session_id, chat, "답변 생성 중...", ""

    add_message(session_id, "assistant", ans)
    yield session_id, chat, "답변 완료", ""

import gradio as gr
from database.sessions import refresh_session_choices, create_session
from app.ui_utils import ui_delete_chat, ui_select_chat, ui_send, ui_new_chat, ui_upload_pdfs


def build_app():
    """Build and wire the Gradio application.

    Args:
        None.

    Returns:
        gr.Blocks: Configured Gradio app instance.
    """

    with gr.Blocks(title="PDF RAG Chat (with History)") as demo:
        gr.Markdown(
            "## PDF RAG Chatbot (Chroma) — Based on uploaded PDF + automatically retrieved documents when it needs more evidence"
        )

        session_id = gr.State(value="")

        with gr.Row():
            # Left: chat history
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Chat History")

                session_dropdown = gr.Dropdown(
                    label="Select Session",
                    choices=refresh_session_choices(),
                    value=None,
                    interactive=True,
                )

                btn_new = gr.Button("New Chat", variant="primary")
                btn_delete = gr.Button("Delete Chat", variant="stop")

                gr.Markdown("### Session Metadata Filter (Optional)")
                gr.Markdown('- 예: `{"filename": {"$eq": "paper"}}`')
                session_filter_json = gr.Textbox(
                    label="Filter(JSON)",
                    placeholder='{"filename":{"$eq":"paper"}}',
                    lines=3,
                )

                gr.Markdown("### Upload PDF (Add on DB)")
                pdf_files = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    file_count="multiple",
                )
                upload_status = gr.Textbox(label="Upload status", lines=6)

            # Right: chat
            with gr.Column(scale=3):
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(height=520)
                user_text = gr.Textbox(
                    label="Insert question",
                    placeholder="Please ask me (필터: @file=xxx.pdf / @page=3 / @doc_id=... / @filter={...})",
                    lines=2,
                )
                with gr.Row():
                    btn_send = gr.Button("Send", variant="primary")
                    btn_clear = gr.Button("Clear Chat UI")

                status = gr.Markdown("")

        # New chat
        def _new_chat():
            """Create a fresh chat session and reset UI state.

            Args:
                None.

            Returns:
                tuple[str, gr.Dropdown, list, str]: New session id, updated dropdown,
                empty chat history, and status message.
            """

            sid, choices, empty_chat, _, msg = ui_new_chat()
            return sid, gr.Dropdown(choices=choices, value=sid), empty_chat, msg

        btn_new.click(
            _new_chat,
            inputs=[],
            outputs=[session_id, session_dropdown, chatbot, status],
        )

        # Delete chat
        def _delete_chat(sid: str):
            """Delete selected session and create a replacement session.

            Args:
                sid (str): Session UUID to delete.

            Returns:
                tuple[str, gr.Dropdown, list, str]: New session id, updated dropdown,
                empty chat history, and status message.
            """

            new_id, choices, empty_chat, msg = ui_delete_chat(sid)
            return new_id, gr.Dropdown(choices=choices, value=new_id), empty_chat, msg

        btn_delete.click(
            _delete_chat,
            inputs=[session_id],
            outputs=[session_id, session_dropdown, chatbot, status],
        )

        # Select chat
        def _select_chat(dd_value: str):
            """Load selected session history into chat UI.

            Args:
                dd_value (str): Selected session UUID from dropdown.

            Returns:
                tuple[str, list, str]: Active session id, chat messages, and status.
            """

            # dd_value is session_id
            if not dd_value:
                return "", [], "Please select a session."
            chat, msg = ui_select_chat(dd_value)
            return dd_value, chat, msg

        session_dropdown.change(
            _select_chat,
            inputs=[session_dropdown],
            outputs=[session_id, chatbot, status],
        )

        # Upload PDFs
        def _upload(sid: str, files):
            """Handle uploaded PDF files and update upload status text.

            Args:
                sid (str): Active session UUID.
                files (Any): Uploaded Gradio file objects.

            Returns:
                str: Upload result message.
            """

            # gr.File passes list of TemporaryFile objects with .name
            msg = ui_upload_pdfs(sid, files or [])
            return msg

        pdf_files.change(
            _upload,
            inputs=[session_id, pdf_files],
            outputs=[upload_status],
        )

        # Send message
        btn_send.click(
            ui_send,
            inputs=[session_id, chatbot, user_text, session_filter_json],
            outputs=[session_id, chatbot, status, user_text],
        )

        # Enter to send
        user_text.submit(
            ui_send,
            inputs=[session_id, chatbot, user_text, session_filter_json],
            outputs=[session_id, chatbot, status, user_text],
        )

        # Clear UI only (does not delete stored history)
        def _clear_ui():
            """Clear chat widgets without deleting persisted history.

            Args:
                None.

            Returns:
                tuple[list, str]: Empty chat list and cleared input text.
            """

            return [], ""

        btn_clear.click(_clear_ui, inputs=[], outputs=[chatbot, user_text])

        # Initial auto-create a session if none
        def _init():
            """Initialize default session state when the app loads.

            Args:
                None.

            Returns:
                tuple[str, gr.Dropdown, list, str]: Session id, dropdown state,
                initial chat history, and status.
            """

            sid = create_session("New Chat")
            return sid, gr.Dropdown(choices=refresh_session_choices(), value=sid), [], "Ready"

        demo.load(_init, inputs=[], outputs=[session_id, session_dropdown, chatbot, status])

    return demo

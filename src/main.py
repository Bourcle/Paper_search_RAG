from dotenv import load_dotenv

load_dotenv()

from app.app_process import build_app
from database import sessions

if __name__ == "__main__":
    sessions.init_chat_db()
    app = build_app()
    app.queue()
    # For local dev: share=False, server_name can be "0.0.0.0" if needed
    app.launch(share=False)

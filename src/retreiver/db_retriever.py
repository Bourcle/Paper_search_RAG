from typing import Optional, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import TOP_K


def retrieve_with_scores(
    db: Chroma, question: str, top_k: int = TOP_K, chroma_filter: Optional[dict[str, Any]] = None
) -> list[tuple[Document, float]]:
    res = list()

    if chroma_filter:
        res = db.similarity_search_with_relevance_scores(question, k=top_k, filter=chroma_filter)
    else:
        res = db.similarity_search_with_relevance_scores(question, k=top_k)

    if not res:
        print("Could not find any simmilar documents")

    return res


def format_context(documents_with_scores: list[tuple[Document, float]]) -> str:
    res = ""

    lines = list()
    for doc, score in documents_with_scores:
        meta_data = doc.metadata or dict()
        page = meta_data.get("page", "NA")
        filename = meta_data.get("filename", meta_data.get("source", "NA"))
        lines.append(f"[{filename} | p.{page} | score={score:3f}] {doc.page_content}")

    res = "\n\n".join(lines)

    return res

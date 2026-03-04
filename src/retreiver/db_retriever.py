from typing import Optional, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import TOP_K


def retrieve_with_scores(
    db: Chroma, question: str, top_k: int = TOP_K, chroma_filter: Optional[dict[str, Any]] = None
) -> list[tuple[Document, float]]:
    """Retrieve top-k similar documents with relevance scores from Chroma.

    Args:
        db (Chroma): Target vector database.
        question (str): Query text.
        top_k (int): Maximum number of documents to retrieve.
        chroma_filter (Optional[dict[str, Any]]): Optional Chroma metadata filter.

    Returns:
        list[tuple[Document, float]]: Retrieved documents with relevance scores.
    """

    res = list()

    if chroma_filter:
        res = db.similarity_search_with_relevance_scores(question, k=top_k, filter=chroma_filter)
    else:
        res = db.similarity_search_with_relevance_scores(question, k=top_k)

    if not res:
        print("Could not find any simmilar documents")

    return res


def format_context(documents_with_scores: list[tuple[Document, float]]) -> str:
    """Format retrieved documents into a context block for prompting.

    Args:
        documents_with_scores (list[tuple[Document, float]]): Retrieved documents
            and their relevance scores.

    Returns:
        str: Prompt-ready context string with source metadata per chunk.
    """

    res = ""

    lines = list()
    for doc, score in documents_with_scores:
        meta_data = doc.metadata or dict()
        page = meta_data.get("page", "NA")
        filename = meta_data.get("filename", meta_data.get("source", "NA"))
        lines.append(f"[{filename} | p.{page} | score={score:3f}] {doc.page_content}")

    res = "\n\n".join(lines)

    return res

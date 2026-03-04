import re
from collections import Counter
from typing import Optional, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import TOP_K


def retrieve_with_sparse(question: str, sparse_raw: dict) -> tuple[dict, float]:
    """Retrieve documents using a simple sparse (token-overlap) scoring method.

    Args:
        question (str): The input query string.
        sparse_raw (dict): Dictionary containing raw document data retrieved from a vector database (e.g., Chroma).

    Returns:
        tuple[dict, float]: A tuple containing:
            - sparse_res (list[tuple[Document, float]]):
              Top-k documents with their sparse relevance scores.
            - max_sparse_score (float):
              The maximum sparse score among the retrieved documents.
              This is typically used for score normalization during
              hybrid (dense + sparse) retrieval.
    """

    res = tuple()
    sparse_res = list()
    max_sparse_score = 0.0

    query_tokens = re.findall(r"[0-9a-zA-Z가-힣]+", (question or "").lower())
    query_counter = Counter(query_tokens)  # 빈도수 기반 sparse vector

    raw_documents = sparse_raw.get("documents", list())
    raw_metadatas = sparse_raw.get("metadatas", list())

    for idx, doc_text in enumerate(raw_documents):
        if not doc_text:
            continue
        doc_tokens = re.findall(r"[0-9a-zA-Z가-힣]+", (doc_text or "").lower())
        if not doc_tokens:
            continue
        doc_counter = Counter(doc_tokens)
        # find query counter in doc counter
        overlap = sum(min(doc_counter[token], count) for token, count in query_counter.items() if token in doc_counter)
        if overlap <= 0:
            continue
        sparse_score = overlap / (len(doc_tokens) + 1) ** 0.5
        if idx < len(raw_documents):
            metadata = raw_metadatas[idx]
        else:
            metadata = dict()
        sparse_res.append((Document(page_content=doc_text, metadata=metadata), sparse_score))

    if sparse_res:
        sparse_res = sorted(sparse_res, key=lambda item: item[1], reverse=True)[:TOP_K]
        max_sparse_score = max((score for _, score in sparse_res))

    res = (sparse_res, max_sparse_score)

    return res


def retrieve_with_scores(
    db: Chroma, question: str, top_k: int = TOP_K, chroma_filter: Optional[dict[str, Any]] = None
) -> list[tuple[Document, float]]:
    """Retrieve top-k documents using Dense + Sparse hybrid scoring.

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
        dense_res = db.similarity_search_with_relevance_scores(question, k=top_k, filter=chroma_filter)
        sparse_raw = db.get(where=chroma_filter, include=["documents", "metadatas"])
    else:
        dense_res = db.similarity_search_with_relevance_scores(question, k=top_k)
        sparse_raw = db.get(include=["documents", "metadatas"])

    sparse_res, max_sparse_score = retrieve_with_sparse(question, sparse_raw)

    merged: dict[tuple[str, str, str, str], dict[str, Any]] = dict()

    for doc, dense_score in dense_res:
        metadata = doc.metadata or dict()
        key = (
            str(metadata.get("doc_id", "")),
            str(metadata.get("filename_full", metadata.get("source", ""))),
            str(metadata.get("page", "")),
            doc.page_content[:300],
        )
        merged[key] = {"doc": doc, "dense": dense_score, "sparse": 0.0}

    for doc, sparse_score in sparse_res:
        metadata = doc.metadata or dict()
        key = (
            str(metadata.get("doc_id", "")),
            str(metadata.get("filename_full", metadata.get("source", ""))),
            str(metadata.get("page", "")),
            doc.page_content[:300],
        )
        normalized_sparse = (sparse_score / max_sparse_score) if max_sparse_score > 0 else 0.0
        if key not in merged:
            merged[key] = {"doc": doc, "dense": 0.0, "sparse": normalized_sparse}
        else:
            merged[key]["sparse"] = max(merged[key]["sparse"], normalized_sparse)

    for item in merged.values():
        hybrid_score = (0.7 * item["dense"]) + (0.3 * item["sparse"])
        res.append((item["doc"], hybrid_score))

    res = sorted(res, key=lambda item: item[1], reverse=True)[:top_k]

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

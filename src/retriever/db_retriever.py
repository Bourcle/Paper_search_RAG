from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import TOP_K, DENSE_K_MULTIPLIER, W_DENSE, W_SPARSE
from retriever.sparse_index import SPARSE_INDEX


def minmax_norm(scores: list[float]) -> list[float]:
    """Apply min-max normalization to score list.

    Args:
        scores (list[float]): Raw score values.

    Returns:
        list[float]: Scores normalized to [0, 1] range.
    """

    res = list()
    if not scores:
        return res

    min_score = min(scores)
    max_score = max(scores)
    if max_score <= min_score:
        res = [1.0 for _ in scores]
        return res

    res = [(score - min_score) / (max_score - min_score) for score in scores]

    return res


def make_key(doc: Document) -> str:
    """Create stable merge key for a document chunk.

    Args:
        doc (Document): LangChain document.

    Returns:
        str: Deterministic key based on metadata and content prefix.
    """

    meta = doc.metadata or dict()
    chunk_key = str(meta.get("chunk_key", ""))
    if chunk_key:
        return chunk_key

    doc_id = str(meta.get("doc_id", ""))
    chunk_id = str(meta.get("chunk_id", meta.get("page", "")))
    if doc_id and chunk_id:
        return f"{doc_id}:{chunk_id}"

    return (
        f'{meta.get("doc_id","")}|{meta.get("filename_full", meta.get("source",""))}|'
        f'{meta.get("page","")}|{doc.page_content[:200]}'
    )


def retrieve_dense_candidates(
    db: Chroma,
    question: str,
    dense_k: int,
    chroma_filter: Optional[dict[str, object]] = None,
) -> list[tuple[Document, float]]:
    """Retrieve dense candidates from Chroma.

    Args:
        db (Chroma): Target vector database.
        question (str): Query text.
        dense_k (int): Number of dense candidates.
        chroma_filter (Optional[dict[str, object]]): Optional Chroma filter.

    Returns:
        list[tuple[Document, float]]: Dense candidates with scores.
    """

    res = list()

    if chroma_filter:
        res = db.similarity_search_with_relevance_scores(question, k=dense_k, filter=chroma_filter)
    else:
        res = db.similarity_search_with_relevance_scores(question, k=dense_k)

    return res


def retrieve_with_scores(
    db: Chroma,
    question: str,
    top_k: int = TOP_K,
    chroma_filter: Optional[dict[str, object]] = None,
) -> list[tuple[Document, float]]:
    """Retrieve documents with parallel dense+sparse hybrid retrieval.

    Args:
        db (Chroma): Target vector database.
        question (str): Query text.
        top_k (int): Number of final results to return.
        chroma_filter (Optional[dict[str, object]]): Chroma metadata filter.

    Returns:
        list[tuple[Document, float]]: Top-k documents with hybrid scores.
    """

    res = list()
    dense_k = max(top_k * DENSE_K_MULTIPLIER, top_k)

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(retrieve_dense_candidates, db, question, dense_k, chroma_filter)
        sparse_future = executor.submit(SPARSE_INDEX.search, question, dense_k, chroma_filter)
        dense_candidates = dense_future.result()
        sparse_candidates = sparse_future.result()

    if not dense_candidates and not sparse_candidates:
        print("Could not find any similar documents")
        return list()

    dense_scores = [float(score) for _, score in dense_candidates]
    sparse_scores = [float(score) for _, score in sparse_candidates]
    dense_norm = minmax_norm(dense_scores)
    sparse_norm = minmax_norm(sparse_scores)

    merged_docs: dict[str, Document] = dict()
    dense_map: dict[str, float] = dict()
    sparse_map: dict[str, float] = dict()

    for idx, (doc, _) in enumerate(dense_candidates):
        key = make_key(doc)
        merged_docs[key] = doc
        dense_map[key] = max(dense_map.get(key, 0.0), dense_norm[idx])

    for idx, (doc, _) in enumerate(sparse_candidates):
        key = make_key(doc)
        if key not in merged_docs:
            merged_docs[key] = doc
        sparse_map[key] = max(sparse_map.get(key, 0.0), sparse_norm[idx])

    res = list()
    for key, doc in merged_docs.items():
        hybrid_score = (W_DENSE * dense_map.get(key, 0.0)) + (W_SPARSE * sparse_map.get(key, 0.0))
        res.append((doc, hybrid_score))

    res.sort(key=lambda item: item[1], reverse=True)

    res = res[:top_k]

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

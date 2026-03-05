import re
from collections import Counter
from math import log
from typing import Optional, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import TOP_K, DENSE_K_MULTIPLIER, W_DENSE, W_SPARSE

SPARSE_INDEX_CACHE: dict[str, dict[str, Any]] = dict()


def get_db_cache_key(db: Chroma) -> str:
    """Build a process-local cache key for a Chroma object.

    Args:
        db (Chroma): Target vector database.

    Returns:
        str: Cache key.
    """

    return str(id(db))


def invalidate_sparse_index_cache(db: Optional[Chroma] = None):
    """Invalidate local sparse index cache.

    Args:
        db (Optional[Chroma]): Target DB instance. If None, clear all cache.

    Returns:
        None: This function does not return a value.
    """

    if db is None:
        SPARSE_INDEX_CACHE.clear()
        return
    SPARSE_INDEX_CACHE.pop(get_db_cache_key(db), None)


def minmax_norm(scores: list[float]) -> list[float]:
    """Apply min-max normalization to score list.

    Args:
        scores (list[float]): Raw score values.

    Returns:
        list[float]: Scores normalized to [0, 1] range.
    """

    if not scores:
        return list()

    min_score = min(scores)
    max_score = max(scores)
    if max_score <= min_score:
        return [0.0 for _ in scores]

    return [(score - min_score) / (max_score - min_score) for score in scores]


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for sparse retrieval.

    Args:
        text (str): Raw text.

    Returns:
        list[str]: Lowercased tokens matched by regex [0-9a-zA-Z가-힣]+.
    """

    return re.findall(r"[0-9a-zA-Z가-힣]+", (text or "").lower())


def make_key(doc: Document) -> str:
    """Create stable merge key for a document chunk.

    Args:
        doc (Document): LangChain document.

    Returns:
        str: Deterministic key based on metadata and content prefix.
    """

    doc_meta = doc.metadata or dict()
    return (
        f'{doc_meta.get("doc_id","")}|{doc_meta.get("filename_full", doc_meta.get("source",""))}|'
        f'{doc_meta.get("page","")}|{doc.page_content[:200]}'
    )


def metadata_matches_filter(metadata: dict[str, Any], chroma_filter: Optional[dict[str, Any]]) -> bool:
    """Check whether metadata satisfies a subset of Chroma-style filters.

    Supported patterns:
    - {"field": {"$eq": value}}
    - {"$and": [ ... ]}

    Args:
        metadata (dict[str, Any]): Document metadata.
        chroma_filter (Optional[dict[str, Any]]): Chroma filter.

    Returns:
        bool: True if matched.
    """

    if not chroma_filter:
        return True

    if "$and" in chroma_filter:
        clauses = chroma_filter.get("$and", list())
        return all(metadata_matches_filter(metadata, clause) for clause in clauses if isinstance(clause, dict))

    for key, condition in chroma_filter.items():
        if isinstance(condition, dict) and "$eq" in condition:
            if metadata.get(key) != condition["$eq"]:
                return False
        else:
            if metadata.get(key) != condition:
                return False
    return True


def build_sparse_index(db: Chroma) -> dict[str, Any]:
    """Build or reuse local sparse BM25 index for all documents.

    Args:
        db (Chroma): Target vector database.

    Returns:
        dict[str, Any]: Cached sparse index payload.
    """

    cache_key = get_db_cache_key(db)
    cached = SPARSE_INDEX_CACHE.get(cache_key)
    if cached:
        return cached

    raw = db.get(include=["documents", "metadatas"])
    documents = raw.get("documents") or list()
    metadatas = raw.get("metadatas") or list()

    postings: dict[str, list[tuple[int, int]]] = dict()
    doc_lens: list[int] = list()

    for idx, text in enumerate(documents):
        tokens = tokenize_text(text or "")
        doc_lens.append(len(tokens))
        tf = Counter(tokens)
        for term, freq in tf.items():
            postings.setdefault(term, list()).append((idx, freq))

    doc_count = len(documents)
    avg_doc_len = (sum(doc_lens) / doc_count) if doc_count > 0 else 1.0
    index_data = {
        "documents": documents,
        "metadatas": metadatas,
        "postings": postings,
        "doc_lens": doc_lens,
        "doc_count": doc_count,
        "avg_doc_len": avg_doc_len,
    }
    SPARSE_INDEX_CACHE[cache_key] = index_data
    return index_data


def retrieve_sparse_candidates(
    db: Chroma,
    question: str,
    sparse_k: int,
    chroma_filter: Optional[dict[str, Any]] = None,
) -> list[tuple[Document, float]]:
    """Retrieve sparse candidates by BM25 scoring with local inverted index.

    Args:
        db (Chroma): Target vector database.
        question (str): Query text.
        sparse_k (int): Number of sparse candidates.
        chroma_filter (Optional[dict[str, Any]]): Chroma metadata filter.

    Returns:
        list[tuple[Document, float]]: Sparse candidates with BM25 scores.
    """

    index_data = build_sparse_index(db)
    doc_count = index_data["doc_count"]
    if doc_count == 0:
        return list()

    query_tokens = tokenize_text(question)
    if not query_tokens:
        return list()

    postings = index_data["postings"]
    documents = index_data["documents"]
    metadatas = index_data["metadatas"]
    doc_lens = index_data["doc_lens"]
    avg_doc_len = index_data["avg_doc_len"] or 1.0

    k1 = 1.5
    b = 0.75
    scores: dict[int, float] = dict()

    for term in set(query_tokens):
        term_postings = postings.get(term, list())
        df = len(term_postings)
        if df == 0:
            continue
        idf = log(1 + ((doc_count - df + 0.5) / (df + 0.5)))

        for doc_idx, tf in term_postings:
            metadata = metadatas[doc_idx] if doc_idx < len(metadatas) else dict()
            if not metadata_matches_filter(metadata or dict(), chroma_filter):
                continue

            dl = doc_lens[doc_idx] or 1
            denom = tf + k1 * (1 - b + b * (dl / avg_doc_len))
            bm25_score = idf * ((tf * (k1 + 1)) / denom)
            scores[doc_idx] = scores.get(doc_idx, 0.0) + bm25_score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:sparse_k]
    res: list[tuple[Document, float]] = list()
    for doc_idx, score in ranked:
        metadata = metadatas[doc_idx] if doc_idx < len(metadatas) else dict()
        doc_text = documents[doc_idx] if doc_idx < len(documents) else ""
        res.append((Document(page_content=doc_text or "", metadata=metadata or dict()), float(score)))

    return res


def retrieve_with_scores(
    db: Chroma,
    question: str,
    top_k: int = TOP_K,
    chroma_filter: Optional[dict[str, Any]] = None,
) -> list[tuple[Document, float]]:
    """Retrieve documents with dense+sparse hybrid retrieval.

    Dense candidates come from Chroma vector search, while sparse candidates come
    from local BM25 index. Final ranking is weighted normalized score fusion.

    Args:
        db (Chroma): Target vector database.
        question (str): Query text.
        top_k (int): Number of final results to return.
        chroma_filter (Optional[dict[str, Any]]): Chroma metadata filter.

    Returns:
        list[tuple[Document, float]]: Top-k documents with hybrid scores.
    """

    dense_k = max(top_k * DENSE_K_MULTIPLIER, top_k)

    if chroma_filter:
        dense_candidates = db.similarity_search_with_relevance_scores(question, k=dense_k, filter=chroma_filter)
    else:
        dense_candidates = db.similarity_search_with_relevance_scores(question, k=dense_k)

    sparse_candidates = retrieve_sparse_candidates(db, question, sparse_k=dense_k, chroma_filter=chroma_filter)

    if not dense_candidates and not sparse_candidates:
        print("Could not find any similar documents")
        return list()

    dense_norm_map: dict[str, float] = dict()
    sparse_norm_map: dict[str, float] = dict()
    docs_map: dict[str, Document] = dict()

    dense_keys = [make_key(doc) for doc, _ in dense_candidates]
    dense_norm = minmax_norm([float(score) for _, score in dense_candidates])
    for key, norm_score, (doc, _) in zip(dense_keys, dense_norm, dense_candidates):
        dense_norm_map[key] = norm_score
        docs_map[key] = doc

    sparse_keys = [make_key(doc) for doc, _ in sparse_candidates]
    sparse_norm = minmax_norm([float(score) for _, score in sparse_candidates])
    for key, norm_score, (doc, _) in zip(sparse_keys, sparse_norm, sparse_candidates):
        sparse_norm_map[key] = norm_score
        if key not in docs_map:
            docs_map[key] = doc

    res: list[tuple[Document, float]] = list()
    all_keys = set(docs_map.keys())
    for key in all_keys:
        hybrid_score = (W_DENSE * dense_norm_map.get(key, 0.0)) + (W_SPARSE * sparse_norm_map.get(key, 0.0))
        res.append((docs_map[key], hybrid_score))

    res.sort(key=lambda item: item[1], reverse=True)
    return res[:top_k]


def format_context(documents_with_scores: list[tuple[Document, float]]) -> str:
    """Format retrieved documents into a context block for prompting.

    Args:
        documents_with_scores (list[tuple[Document, float]]): Retrieved documents
            and their relevance scores.

    Returns:
        str: Prompt-ready context string with source metadata per chunk.
    """

    lines = list()
    for doc, score in documents_with_scores:
        meta_data = doc.metadata or dict()
        page = meta_data.get("page", "NA")
        filename = meta_data.get("filename", meta_data.get("source", "NA"))
        lines.append(f"[{filename} | p.{page} | score={score:3f}] {doc.page_content}")

    return "\n\n".join(lines)

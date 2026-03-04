import uuid
from pathlib import Path
from typing import Callable, Optional, Any
from config import (
    DEFAULT_EMB_MODEL,
    DEFAUL_DB_DIR,
    DEFAULT_COLLECTION,
    INSUFFICIENT_MSG,
    MIN_RELEVANCE,
    TOP_K,
    QUERY_REWRITE_CHAIN,
    ARXIV_MAX_RESULTS,
    AUTO_PAPERS_DIR,
    PMC_MAX_RESULTS,
    PUBMED_MAX_RESULTS,
)
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from retreiver.pdf_utils import load_pdf_docs, split_docs, download_pdf_checked
from retreiver.web_retriever import arxiv_search, pmc_search, pubmed_search_abstracts
from retreiver.db_retriever import retrieve_with_scores, format_context
from utils.utils import parse_filter_from_question, looks_korean


def open_vector_db(
    emb_model: str = DEFAULT_EMB_MODEL, persist_dir: str = DEFAUL_DB_DIR, collection: str = DEFAULT_COLLECTION
) -> Chroma:
    """Initialize and return a Chroma vector database instance.

    Args:
        emb_model (str): Embedding model name.
        persist_dir (str): Chroma persistence directory.
        collection (str): Collection name.

    Returns:
        Chroma: Opened Chroma client.
    """

    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    res = Chroma(collection_name=collection, persist_directory=persist_dir, embedding_function=embeddings)

    return res


VECTOR_DB = open_vector_db()


def add_pdf_to_db(db: Chroma, pdf_path: str) -> int:
    """Ingest a PDF file into the vector database as chunked documents.

    Args:
        db (Chroma): Target vector database.
        pdf_path (str): Path to a PDF file.

    Raises:
        ValueError: Raised when no valid text chunks are extracted.

    Returns:
        int: Number of chunks added.
    """

    res = 0

    raw_docs = load_pdf_docs(pdf_path)
    chunked_docs = split_docs(raw_docs)

    if not chunked_docs:
        raise ValueError(f"Could not extract text from PDF.: {pdf_path}")

    document_id = str(uuid.uuid4())
    filename = Path(pdf_path).stem
    filename_full = Path(pdf_path).name

    for chunk in chunked_docs:
        chunk.metadata = dict(chunk.metadata or dict())
        chunk.metadata.update({"doc_id": document_id, "filename": filename, "filename_full": filename_full})

    document_ids = [str(uuid.uuid4()) for _ in range(len(chunked_docs))]
    db.add_documents(chunked_docs, ids=document_ids)

    res = len(chunked_docs)

    return res


def _doc_dedupe_key(doc: Document) -> tuple[str, str, str, str]:
    """Build a deterministic key to deduplicate retrieved chunks.

    Args:
        doc (Document): Retrieved LangChain document.

    Returns:
        tuple[str, str, str, str]: Key components based on metadata and content.
    """

    metadata = doc.metadata or dict()
    doc_id = str(metadata.get("doc_id", ""))
    filename_full = str(metadata.get("filename_full", metadata.get("source", "")))
    page = str(metadata.get("page", ""))
    snippet = doc.page_content[:300]

    return doc_id, filename_full, page, snippet


def _merge_ranked_results(
    primary: list[tuple[Document, float]], secondary: list[tuple[Document, float]], top_k: int
) -> list[tuple[Document, float]]:
    """Merge two ranked retrieval results, deduplicate, and keep top-k by score.

    Args:
        primary (list[tuple[Document, float]]): First ranked result list.
        secondary (list[tuple[Document, float]]): Second ranked result list.
        top_k (int): Maximum number of merged items.

    Returns:
        list[tuple[Document, float]]: Merged and score-sorted retrieval results.
    """

    merged: dict[tuple[str, str, str, str], tuple[Document, float]] = dict()

    for doc, score in primary + secondary:
        key = _doc_dedupe_key(doc)
        prev = merged.get(key)
        if (not prev) or (score > prev[1]):
            merged[key] = (doc, score)

    res = sorted(merged.values(), key=lambda item: item[1], reverse=True)[:top_k]
    return res


def _rewrite_query_to_english(question: str) -> str:
    """Rewrite a query into English keywords for multilingual retrieval.

    Args:
        question (str): Original user query.

    Returns:
        str: Rewritten English query. Returns the original question on failure.
    """

    try:
        rewritten = QUERY_REWRITE_CHAIN.invoke({"question": question}).strip()
        if rewritten and len(rewritten) <= 200:
            return rewritten
    except Exception as e:
        print(f"[query_rewrite] failed: {repr(e)}")
    return question


def answer_from_db(
    db: Chroma, chain: Callable, raw_question: str, session_filter: Optional[dict[str, Any]] = None
) -> tuple[str, list[tuple[Document, float]], Optional[dict[str, Any]]]:
    """Generate an answer from retrieved evidence in the vector database.

    Args:
        db (Chroma): Target vector database.
        chain (Callable): QA chain used to produce final answers.
        raw_question (str): Raw user question, optionally with inline filters.
        session_filter (Optional[dict[str, Any]]): Session-level metadata filter.

    Returns:
        tuple[str, list[tuple[Document, float]], Optional[dict[str, Any]]]:
        Answer text, retrieved docs with scores, and the merged filter.
    """

    clean_question, inline_filter = parse_filter_from_question(raw_question)
    chroma_filter = merge_filters(inline_filter, session_filter)

    docs_scores_primary = retrieve_with_scores(db, clean_question, top_k=TOP_K, chroma_filter=chroma_filter)
    docs_scores = docs_scores_primary

    if looks_korean(clean_question):
        english_query = _rewrite_query_to_english(clean_question)
        if english_query != clean_question:
            docs_scores_secondary = retrieve_with_scores(db, english_query, top_k=TOP_K, chroma_filter=chroma_filter)
            docs_scores = _merge_ranked_results(docs_scores_primary, docs_scores_secondary, top_k=TOP_K)

    if not docs_scores:
        res = (INSUFFICIENT_MSG, list(), chroma_filter)
        return res
    best_score = max(score for _, score in docs_scores)
    if best_score < MIN_RELEVANCE:
        res = (INSUFFICIENT_MSG, docs_scores, chroma_filter)
        return res

    context = format_context(docs_scores)
    answer = chain.invoke({"context": context, "question": clean_question}).strip()

    if answer == INSUFFICIENT_MSG:
        res = (INSUFFICIENT_MSG, docs_scores, chroma_filter)
        return res

    res = (answer, docs_scores, chroma_filter)

    return res


def check_doc_exist(pdf_path: str, db: Chroma) -> bool:
    """Check whether a PDF (by stem filename) already exists in Chroma metadata.

    Args:
        pdf_path (str): Local PDF path.
        db (Chroma): Target vector database.

    Returns:
        bool: ``True`` when matching chunks already exist.
    """

    res = False

    if not pdf_path:
        return res

    filename = Path(pdf_path).stem
    existing = db.get(where={"filename": filename})

    if filename and existing.get("ids"):
        res = True

    return res


def merge_filters(
    inline_filter: Optional[dict[str, Any]], session_filter: Optional[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    """Merge inline filter and session filter into a single Chroma filter.

    Args:
        inline_filter (Optional[dict[str, Any]]): Filter parsed from the query.
        session_filter (Optional[dict[str, Any]]): Filter from session UI state.

    Returns:
        Optional[dict[str, Any]]: Merged filter. Returns ``None`` if both are empty.
    """

    if inline_filter and session_filter:
        return {"$and": [session_filter, inline_filter]}
    return session_filter or inline_filter


def add_pubmed_abstracts_to_db(db: Chroma, papers: list[Any]) -> int:
    """Add PubMed abstract documents to the vector database.

    Args:
        db (Chroma): Target vector database.
        papers (list[Any]): PubMed-like paper objects with ``pmid``, ``title``,
            ``abstract``, ``journal``, and ``pub_date`` attributes.

    Returns:
        int: Number of newly added abstract documents.
    """

    docs = list()
    ids = list()

    for paper in papers:
        existing = db.get(where={"pmid": paper.pmid})
        if existing.get("ids"):
            continue

        content = (
            f"Title: {paper.title}\n"
            f"Journal: {paper.journal}\n"
            f"PublicationDate: {paper.pub_date}\n"
            f"PMID: {paper.pmid}\n\n"
            f"Abstract:\n{paper.abstract}"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "pubmed_abstract",
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "filename": "pubmed",
                    "filename_full": f"pubmed_{paper.pmid}",
                    "doc_id": f"pubmed-{paper.pmid}",
                    "page": 0,
                },
            )
        )
        ids.append(str(uuid.uuid4()))

    if docs:
        db.add_documents(docs, ids=ids)
    return len(docs)


def auto_fetch_and_ingest(db: Chroma, question: str) -> Optional[str]:
    """Automatically retrieve external papers and ingest evidence into Chroma.

    Retrieval strategy is PMC PDF -> PubMed abstract -> arXiv PDF fallback.

    Args:
        db (Chroma): Target vector database.
        question (str): User question used as search query seed.

    Returns:
        Optional[str]: Downloaded file path or ``pubmed:{n}`` marker when success;
        empty string when all retrieval paths fail.
    """

    res = ""

    search_query = question
    if looks_korean(question):
        search_query = _rewrite_query_to_english(question)
        print(f"search query: {search_query}")

    # 1) PMC(PubMed Central, OA PDF)
    pmc_download_failures = 0
    try:
        pmc_papers = pmc_search(search_query, max_results=PMC_MAX_RESULTS)
        for paper in pmc_papers:
            try:
                path = download_pdf_checked(
                    paper.pdf_url,
                    out_dir=AUTO_PAPERS_DIR,
                    filename_hint=f"{paper.pmcid}.pdf",
                )
                if not check_doc_exist(path, db):
                    add_pdf_to_db(db, path)
                res = path
                break
            except Exception as e:
                print(f"[PMC] {paper.pmcid} Failed to download PDF: {repr(e)}")
                pmc_download_failures += 1
                continue

    except Exception as e:
        print(f"[auto_fetch_and_ingest] PMC failed: {repr(e)}")

    if not res:
        print(
            f"[auto_fetch_and_ingest] PMC OA ingest failed "
            f"(download_failures={pmc_download_failures}). Try PubMed abstract fallback."
        )
        try:
            pubmed_papers = pubmed_search_abstracts(search_query, max_results=PUBMED_MAX_RESULTS)
            added = add_pubmed_abstracts_to_db(db, pubmed_papers)
            if added > 0:
                print(f"[PubMed] Success to add abstract to DB: {added}")
                res = f"pubmed:{added}"
        except Exception as e:
            print(f"[auto_fetch_and_ingest] PubMed abstract fallback failed: {repr(e)}")

    # 3) arXiv
    if not res:
        print(
            "[auto_fetch_and_ingest] Try out arXiv fallback since the model could not find enough evidence in PMC/PubMed."
        )
        try:
            arxiv_papers = arxiv_search(search_query, max_results=ARXIV_MAX_RESULTS)
            for paper in arxiv_papers:
                try:
                    path = download_pdf_checked(
                        paper.pdf_url,
                        out_dir=AUTO_PAPERS_DIR,
                        filename_hint=f"{paper.arxiv_id}.pdf",
                    )
                    if not check_doc_exist(path, db):
                        add_pdf_to_db(db, path)
                    res = path
                    break
                except Exception as e:
                    print(f"[Arvix] {paper.arxiv_id} Failed to download PDF: {repr(e)}")
                    continue

        except Exception as e:
            print(f"[auto_fetch_and_ingest] arXiv failed: {repr(e)}")

    return res

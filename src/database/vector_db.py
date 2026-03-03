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
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    res = Chroma(collection_name=collection, persist_directory=persist_dir, embedding_function=embeddings)

    return res


VECTOR_DB = open_vector_db()


def add_pdf_to_db(db: Chroma, pdf_path: str) -> int:
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


def answer_from_db(
    db: Chroma, chain: Callable, raw_question: str, session_filter: Optional[dict[str, Any]] = None
) -> tuple[str, list[tuple[Document, float]], Optional[dict[str, Any]]]:
    clean_question, inline_filter = parse_filter_from_question(raw_question)
    chroma_filter = merge_filters(inline_filter, session_filter)
    docs_scores = retrieve_with_scores(db, clean_question, top_k=TOP_K, chroma_filter=chroma_filter)

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
    if inline_filter and session_filter:
        return {"$and": [session_filter, inline_filter]}
    return session_filter or inline_filter


def add_pubmed_abstracts_to_db(db: Chroma, papers: list[Any]) -> int:
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
    res = ""

    search_query = question
    try:
        if looks_korean(question):
            search_query = QUERY_REWRITE_CHAIN.invoke({"question": question}).strip()
            print(f"search quer: {search_query}")
            if not search_query or len(search_query) > 200:
                search_query = question
    except Exception as e:
        print(f"[query_rewrite] failed: {repr(e)}")
        search_query = question

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

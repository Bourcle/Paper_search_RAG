import os
import re
import urllib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import CHUNCK_SIZE, CHUNK_OVERLAP


def ensure_pdf(path_to_pdf: str) -> bool:

    res = True
    if not os.path.exists(path_to_pdf):
        res = False
        raise FileNotFoundError(f"No file found on {path_to_pdf}")
    if not path_to_pdf.lower().endswith(".pdf"):
        res = False
        raise ValueError(f"We only support PDF. Current file : {path_to_pdf.split('.')[-1].lower()}")

    return res


def load_pdf_docs(pdf_path: str) -> list[Document]:
    res = list()

    loader = PyPDFLoader(pdf_path)
    res = loader.load()

    return res


def is_reference_chunk(text: str) -> bool:

    res = False
    section_headers = {"REFERENCES", "REFERENCE", "BIBLIOGRAPHY", "ACKNOWLEDGMENTS", "ACKNOWLEDGEMENTS", "참고문헌"}
    lines = text.strip().splitlines()
    if not lines:
        res = True

    for line in lines:
        if line.strip().upper() in section_headers:
            res = True

    dot_ref_pattern = re.compile(r"^\d{1,3}\.\s+[A-Z]")
    bracket_ref_pattern = re.compile(r"^\[\d{1,3}\]\s*[A-Z]")

    ref_lines = sum(
        1 for line in lines if dot_ref_pattern.match(line.strip()) or bracket_ref_pattern.match(line.strip())
    )
    if len(lines) >= 2 and ref_lines / len(lines) >= 0.3:
        res = True

    return res


def is_metadata_chunk(text: str) -> bool:

    res = False

    lines = text.strip().splitlines()
    if not lines:
        res = True

    dept_pattern = re.compile(r"(^department of|^school of|university|langone|grossman)", re.IGNORECASE)
    dept_lines = sum(1 for line in lines if dept_pattern.search(line.strip()))
    if len(lines) >= 3 and dept_lines / len(lines) >= 0.4:
        res = True

    return res


def split_docs(documents: list[Document]) -> list[Document]:
    res = list()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNCK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n"]
    )
    splited_docs = splitter.split_documents(documents=documents)

    res = [
        doc
        for doc in splited_docs
        if not is_reference_chunk(doc.page_content) and not is_metadata_chunk(doc.page_content)
    ]

    return res


def download_pdf_checked(url: str, out_dir: str, filename_hint: str) -> str:
    """
    Safer downloader: checks content-type + size; handles 403/html.
    """
    os.makedirs(out_dir, exist_ok=True)

    res = ""
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", filename_hint)[:120]

    res = f"{out_dir}/{safe if safe.lower().endswith('.pdf') else safe + '.pdf'}"
    if os.path.exists(res) and Path(res).stat().st_size >= 10000:
        return res

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        body = resp.read()

    # Some endpoints may return HTML (e.g., consent page / 403 page)
    if "pdf" not in ctype and body[:200].lstrip().startswith(b"<"):
        raise RuntimeError(f"Response without PDF(Content-Type={ctype}). URL={url}")

    Path(res).write_bytes(body)

    if Path(res).stat().st_size < 10000:
        raise RuntimeError(f"The size of file is too small: {res} ({Path(res).stat().st_size} bytes)")

    return res

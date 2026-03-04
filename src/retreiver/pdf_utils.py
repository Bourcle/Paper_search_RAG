import os
import re
import urllib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config import CHUNCK_SIZE, CHUNK_OVERLAP


def ensure_pdf(path_to_pdf: str) -> bool:
    """Validate that the given path exists and points to a PDF file.

    Args:
        path_to_pdf (str): Path to a local file.

    Raises:
        FileNotFoundError: Raised when the file does not exist.
        ValueError: Raised when the file extension is not ``.pdf``.

    Returns:
        bool: ``True`` when the file is a valid PDF path.
    """

    res = True
    if not os.path.exists(path_to_pdf):
        res = False
        raise FileNotFoundError(f"No file found on {path_to_pdf}")
    if not path_to_pdf.lower().endswith(".pdf"):
        res = False
        raise ValueError(f"We only support PDF. Current file : {path_to_pdf.split('.')[-1].lower()}")

    return res


def load_pdf_docs(pdf_path: str) -> list[Document]:
    """Load PDF pages into LangChain documents.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list[Document]: Loaded page-level documents.
    """

    res = list()

    loader = PyPDFLoader(pdf_path)
    res = loader.load()

    return res


def is_reference_chunk(text: str) -> bool:
    """Heuristically detect whether a text chunk is from a references section.

    Args:
        text (str): Chunk text.

    Returns:
        bool: ``True`` when the chunk appears to be bibliography/reference content.
    """

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
    """Detect chunks that look like author affiliation or metadata blocks.

    Args:
        text (str): Chunk text.

    Returns:
        bool: ``True`` when the chunk appears to be non-content metadata.
    """

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
    """Split loaded PDF documents and filter out noisy chunks.

    Args:
        documents (list[Document]): Raw documents loaded from a PDF.

    Returns:
        list[Document]: Filtered content chunks suitable for vector indexing.
    """

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
    """Download a PDF file with content-type and size validation.

    Args:
        url (str): Source URL of the PDF.
        out_dir (str): Directory where the file should be saved.
        filename_hint (str): Suggested output filename.

    Raises:
        RuntimeError: Raised when the response is not a PDF-like payload.
        RuntimeError: Raised when the downloaded file size is too small.

    Returns:
        str: Absolute or relative path to the downloaded PDF.
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

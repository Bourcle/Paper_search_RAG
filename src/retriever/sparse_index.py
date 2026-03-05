import sqlite3
import re
from pathlib import Path
from typing import Any, Optional
from langchain_core.documents import Document
from config import SPARSE_DB_PATH


class SparseIndex:
    """SQLite FTS5-based sparse index with incremental upsert/delete."""

    def __init__(self, db_path: str = SPARSE_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def db_conn(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        conn = self.db_conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sparse_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_key TEXT UNIQUE,
                doc_id TEXT,
                chunk_id TEXT,
                filename TEXT,
                filename_full TEXT,
                source TEXT,
                page INTEGER,
                pmid TEXT,
                title TEXT,
                text TEXT
            )
        """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sparse_docs_doc_id ON sparse_docs(doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sparse_docs_filename ON sparse_docs(filename)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sparse_docs_pmid ON sparse_docs(pmid)")
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS sparse_fts
            USING fts5(text, content='sparse_docs', content_rowid='id')
        """
        )
        conn.commit()
        conn.close()

    def row_to_document(self, row: sqlite3.Row) -> Document:
        metadata = {
            "doc_id": row["doc_id"] or "",
            "chunk_id": row["chunk_id"] or "",
            "chunk_key": row["chunk_key"] or "",
            "filename": row["filename"] or "",
            "filename_full": row["filename_full"] or "",
            "source": row["source"] or "",
            "page": row["page"] if row["page"] is not None else "NA",
            "pmid": row["pmid"] or "",
            "title": row["title"] or "",
        }
        return Document(page_content=row["text"] or "", metadata=metadata)

    def upsert_document(self, doc: Document):
        metadata = doc.metadata or {}
        chunk_key = str(
            metadata.get("chunk_key")
            or (f'{metadata.get("doc_id","")}:' f'{metadata.get("chunk_id", metadata.get("page", "0"))}')
        )

        conn = self.db_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT id FROM sparse_docs WHERE chunk_key = ?", (chunk_key,))
        existing = cur.fetchone()
        if existing:
            old_id = int(existing["id"])
            cur.execute("DELETE FROM sparse_fts WHERE rowid = ?", (old_id,))
            cur.execute("DELETE FROM sparse_docs WHERE id = ?", (old_id,))

        cur.execute(
            """
            INSERT INTO sparse_docs (chunk_key, doc_id, chunk_id, filename, filename_full, source, page, pmid, title, text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                chunk_key,
                str(metadata.get("doc_id", "")),
                str(metadata.get("chunk_id", "")),
                str(metadata.get("filename", "")),
                str(metadata.get("filename_full", "")),
                str(metadata.get("source", "")),
                metadata.get("page", None),
                str(metadata.get("pmid", "")),
                str(metadata.get("title", "")),
                doc.page_content or "",
            ),
        )
        rowid = cur.lastrowid
        cur.execute("INSERT INTO sparse_fts(rowid, text) VALUES (?, ?)", (rowid, doc.page_content or ""))
        conn.commit()
        conn.close()

    def delete_by_chunk_key(self, chunk_key: str):
        conn = self.db_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id FROM sparse_docs WHERE chunk_key = ?", (chunk_key,))
        row = cur.fetchone()
        if row:
            rowid = int(row["id"])
            cur.execute("DELETE FROM sparse_fts WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM sparse_docs WHERE id = ?", (rowid,))
            conn.commit()
        conn.close()

    def delete_by_doc_id(self, doc_id: str):
        conn = self.db_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id FROM sparse_docs WHERE doc_id = ?", (doc_id,))
        rows = cur.fetchall()
        for row in rows:
            rowid = int(row["id"])
            cur.execute("DELETE FROM sparse_fts WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM sparse_docs WHERE id = ?", (rowid,))
        conn.commit()
        conn.close()

    def delete_by_filename(self, filename: str):
        conn = self.db_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id FROM sparse_docs WHERE filename = ?", (filename,))
        rows = cur.fetchall()
        for row in rows:
            rowid = int(row["id"])
            cur.execute("DELETE FROM sparse_fts WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM sparse_docs WHERE id = ?", (rowid,))
        conn.commit()
        conn.close()

    def build_filter_sql(self, chroma_filter: Optional[dict[str, Any]]) -> tuple[str, list[Any]]:
        if not chroma_filter:
            return "", list()

        if "$and" in chroma_filter:
            parts = list()
            params = list()
            for clause in chroma_filter.get("$and", list()):
                clause_sql, clause_params = self.build_filter_sql(clause)
                if clause_sql:
                    parts.append(f"({clause_sql})")
                    params.extend(clause_params)
            return " AND ".join(parts), params

        sql_parts = list()
        params = list()
        supported_cols = {
            "doc_id",
            "chunk_id",
            "chunk_key",
            "filename",
            "filename_full",
            "source",
            "page",
            "pmid",
            "title",
        }
        for key, condition in chroma_filter.items():
            if key not in supported_cols:
                continue
            if isinstance(condition, dict) and "$eq" in condition:
                sql_parts.append(f"{key} = ?")
                params.append(condition["$eq"])
            else:
                sql_parts.append(f"{key} = ?")
                params.append(condition)
        return " AND ".join(sql_parts), params

    def search(
        self, query: str, top_k: int, chroma_filter: Optional[dict[str, Any]] = None
    ) -> list[tuple[Document, float]]:
        if not (query or "").strip():
            return list()

        tokens = re.findall(r"[0-9a-zA-Z가-힣]+", query.lower())
        if not tokens:
            return list()
        # Use quoted tokens to avoid FTS parser errors with symbols like '-'.
        match_query = " ".join(f'"{token}"' for token in tokens)

        conn = self.db_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        filter_sql, params = self.build_filter_sql(chroma_filter)
        where_sql = "sparse_fts MATCH ?"
        sql_params: list[Any] = [match_query]
        if filter_sql:
            where_sql += f" AND {filter_sql}"
            sql_params.extend(params)

        cur.execute(
            f"""
            SELECT
                sparse_docs.*,
                bm25(sparse_fts) AS bm25_score
            FROM sparse_fts
            JOIN sparse_docs ON sparse_docs.id = sparse_fts.rowid
            WHERE {where_sql}
            ORDER BY bm25_score ASC
            LIMIT ?
        """,
            [*sql_params, top_k],
        )
        rows = cur.fetchall()
        conn.close()

        res = list()
        for row in rows:
            doc = self.row_to_document(row)
            # sqlite bm25: smaller is better -> convert to larger-is-better score
            sparse_score = -float(row["bm25_score"])
            res.append((doc, sparse_score))
        return res


SPARSE_INDEX = SparseIndex()

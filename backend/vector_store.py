"""Persistent SQLite + NumPy vector store supporting multi-provider embeddings and full-document reads."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import List, Optional, Sequence
import uuid

import numpy as np

from backend.parser import TextChunk


@dataclass(frozen=True)
class DocumentRecord:
    """Metadata describing an indexed document and its embedding source."""

    doc_id: str
    filename: str
    file_hash: str
    page_count: int
    chunk_count: int
    embed_provider: str
    embed_model: str
    created_at: str


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with similarity score and document metadata."""

    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float
    embed_provider: str


class VectorStore:
    """SQLite-backed document & chunk store supporting ordered full-PDF reads and vector search."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    embed_provider TEXT NOT NULL DEFAULT 'ollama',
                    embed_model TEXT NOT NULL DEFAULT 'nomic-embed-text',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(file_hash, embed_provider, embed_model)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dim INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                """
            )
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "embed_provider" not in cols:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN embed_provider TEXT NOT NULL DEFAULT 'ollama'"
                )
            if "embed_model" not in cols:
                conn.execute(
                    "ALTER TABLE documents ADD COLUMN embed_model TEXT NOT NULL DEFAULT 'nomic-embed-text'"
                )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            filename=row["filename"],
            file_hash=row["file_hash"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            embed_provider=row["embed_provider"],
            embed_model=row["embed_model"],
            created_at=row["created_at"],
        )

    def get_by_hash(
        self,
        file_hash: str,
        embed_provider: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> Optional[DocumentRecord]:
        """Return an existing document record matching file hash (and optional embedder)."""
        sql = """
            SELECT d.doc_id, d.filename, d.file_hash, d.page_count,
                   d.embed_provider, d.embed_model, d.created_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id
            WHERE d.file_hash = ?
        """
        params: List[str] = [file_hash]
        if embed_provider is not None:
            sql += " AND d.embed_provider = ?"
            params.append(embed_provider)
        if embed_model is not None:
            sql += " AND d.embed_model = ?"
            params.append(embed_model)
        sql += " GROUP BY d.doc_id LIMIT 1"

        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return self._row_to_document(row) if row is not None else None

    def add_document(
        self,
        filename: str,
        file_hash: str,
        page_count: int,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
        embed_provider: str = "ollama",
        embed_model: str = "nomic-embed-text",
    ) -> DocumentRecord:
        """Persist a document and its normalized chunk embeddings atomically."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings lengths must match")
        if not chunks:
            raise ValueError("Cannot index a document with zero chunks")

        doc_id = uuid.uuid4().hex[:12]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, filename, file_hash, page_count, embed_provider, embed_model
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, filename, file_hash, page_count, embed_provider, embed_model),
            )
            rows = []
            for chunk, vec in zip(chunks, embeddings):
                arr = np.asarray(vec, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0.0:
                    arr = arr / norm
                rows.append(
                    (
                        doc_id,
                        chunk.chunk_index,
                        chunk.page_number,
                        chunk.text,
                        arr.tobytes(),
                        int(arr.shape[0]),
                    )
                )
            conn.executemany(
                """
                INSERT INTO chunks (doc_id, chunk_index, page_number, text, embedding, dim)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

        record = self.get_by_hash(
            file_hash,
            embed_provider=embed_provider,
            embed_model=embed_model,
        )
        assert record is not None
        return record

    def list_documents(self, doc_ids: Optional[Sequence[str]] = None) -> List[DocumentRecord]:
        """Return indexed documents ordered newest first, optionally filtered by doc_ids."""
        sql = """
            SELECT d.doc_id, d.filename, d.file_hash, d.page_count,
                   d.embed_provider, d.embed_model, d.created_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.doc_id = d.doc_id
        """
        params: List[str] = []
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            sql += f" WHERE d.doc_id IN ({placeholders})"
            params.extend(doc_ids)

        sql += " GROUP BY d.doc_id ORDER BY d.created_at DESC, d.rowid DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_document(r) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks by doc_id. Returns True if deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cur.rowcount > 0

    def get_document_chunks(
        self,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> List[SearchResult]:
        """Return all chunks for the selected documents in exact page/chunk reading order."""
        sql = """
            SELECT c.doc_id, d.filename, d.embed_provider, c.page_number,
                   c.chunk_index, c.text
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
        """
        params: List[str] = []
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            sql += f" WHERE c.doc_id IN ({placeholders})"
            params.extend(doc_ids)

        sql += " ORDER BY d.created_at ASC, d.rowid ASC, c.chunk_index ASC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            SearchResult(
                doc_id=r["doc_id"],
                filename=r["filename"],
                page_number=r["page_number"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                score=1.0,
                embed_provider=r["embed_provider"],
            )
            for r in rows
        ]

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        doc_ids: Optional[Sequence[str]] = None,
        embed_provider: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> List[SearchResult]:
        """Find the top_k most similar chunks using cosine similarity."""
        if top_k <= 0:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q = q / q_norm

        sql = """
            SELECT c.doc_id, d.filename, d.embed_provider, c.page_number,
                   c.chunk_index, c.text, c.embedding, c.dim
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE 1=1
        """
        params: List[str] = []
        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            sql += f" AND c.doc_id IN ({placeholders})"
            params.extend(doc_ids)
        if embed_provider:
            sql += " AND d.embed_provider = ?"
            params.append(embed_provider)
        if embed_model:
            sql += " AND d.embed_model = ?"
            params.append(embed_model)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        q_dim = int(q.shape[0])
        compatible_rows = [r for r in rows if r["dim"] == q_dim]
        if not compatible_rows:
            return []

        matrix = np.vstack(
            [np.frombuffer(r["embedding"], dtype=np.float32) for r in compatible_rows]
        )
        scores = matrix @ q
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[SearchResult] = []
        for idx in top_indices:
            row = compatible_rows[int(idx)]
            results.append(
                SearchResult(
                    doc_id=row["doc_id"],
                    filename=row["filename"],
                    page_number=row["page_number"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    score=float(scores[int(idx)]),
                    embed_provider=row["embed_provider"],
                )
            )
        return results

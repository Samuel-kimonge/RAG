from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SPARSE_DB = (
    PROJECT_ROOT
    / "retrieval"
    / "indexes"
    / "knowledge_base_fts.sqlite3"
)

DEFAULT_BATCH_SIZE = 500
MAX_QUERY_TERMS = 64


class SparseRetrievalResult:
    """A document returned by SQLite FTS5/BM25."""

    def __init__(
        self,
        chunk_id: str,
        document: Document,
        bm25_score: float,
    ) -> None:
        self.chunk_id = chunk_id
        self.document = document
        self.bm25_score = bm25_score

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


class SQLiteFTS5Index:
    """
    Persistent lexical index synchronized from an existing Chroma collection.

    ChromaDB remains the source of truth. SQLite stores a searchable
    copy of document content, original chunk IDs and metadata.
    """

    def __init__(
        self,
        collection: Any,
        db_path: Path = DEFAULT_SPARSE_DB,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer.")

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        self.collection = collection
        self.db_path = Path(db_path)
        self.batch_size = batch_size

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

        # Synchronize automatically when the index is initialized.
        self.last_sync_stats = self.sync_from_collection()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute("PRAGMA busy_timeout = 30000")

        return connection

    def _initialize_database(self) -> None:
        """Create the SQLite tables and FTS5 virtual table."""

        connection = self._connect()

        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")

            connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    rowid INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source TEXT,
                    file_type TEXT,
                    filename TEXT
                )
            """)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS index_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_source
                ON chunks(source)
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_file_type
                ON chunks(file_type)
            """)

            connection.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    content,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
            """)

            connection.commit()

        except sqlite3.OperationalError as error:
            connection.rollback()

            raise RuntimeError(
                "Could not initialize SQLite FTS5. "
                "Verify that SQLite was compiled with FTS5 support."
            ) from error

        finally:
            connection.close()

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> str:
        try:
            return json.dumps(
                metadata,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Document metadata is not JSON serializable: {error}"
            ) from error

    @staticmethod
    def _document_hash(
        content: str,
        metadata_json: str,
    ) -> str:

        payload = (
            content.encode("utf-8")
            + b"\0"
            + metadata_json.encode("utf-8")
        )

        return hashlib.sha256(payload).hexdigest()

    def sync_from_collection(self) -> dict[str, int]:
        """
        Incrementally synchronize SQLite with the Chroma collection.

        Unchanged chunks are skipped. Changed chunks are updated.
        Chunks removed from Chroma are removed from SQLite.
        """

        connection = self._connect()

        stats = {
            "inserted": 0,
            "updated": 0,
            "unchanged": 0,
            "deleted": 0,
        }

        try:
            expected_count = int(self.collection.count())

            # Start the transaction before any write operation.
            connection.execute("BEGIN IMMEDIATE")

            connection.execute("""
                 CREATE TEMP TABLE IF NOT EXISTS current_ids (
                 chunk_id TEXT PRIMARY KEY
                 )
            """)

            connection.execute("DELETE FROM current_ids")
            for offset in range(
                0,
                expected_count,
                self.batch_size,
            ):
            
                batch = self.collection.get(
                    limit=self.batch_size,
                    offset=offset,
                    include=[
                        "documents",
                        "metadatas",
                    ],
                )

                ids = batch.get("ids") or []
                documents = batch.get("documents") or []
                metadatas = batch.get("metadatas") or []

                if not ids:
                    raise RuntimeError(
                        "Chroma returned an unexpected empty batch "
                        f"at offset {offset}."
                    )

                if len(ids) != len(documents) or len(ids) != len(metadatas):
                    raise RuntimeError(
                        "Chroma returned inconsistent IDs, documents "
                        "and metadata counts."
                    )

                connection.executemany(
                    """
                    INSERT OR IGNORE INTO current_ids(chunk_id)
                    VALUES (?)
                    """,
                    [(chunk_id,) for chunk_id in ids],
                )

                for chunk_id, raw_content, raw_metadata in zip(
                    ids,
                    documents,
                    metadatas,
                ):

                    if not isinstance(chunk_id, str) or not chunk_id:
                        raise ValueError(
                            "Chroma returned an invalid chunk ID."
                        )

                    content = raw_content or ""

                    if not isinstance(content, str):
                        raise TypeError(
                            f"Invalid document content for {chunk_id}."
                        )

                    metadata = raw_metadata or {}

                    if not isinstance(metadata, dict):
                        raise TypeError(
                            f"Invalid metadata for chunk {chunk_id}."
                        )

                    metadata_json = self._serialize_metadata(metadata)

                    content_hash = self._document_hash(
                        content,
                        metadata_json,
                    )

                    source = metadata.get("source")
                    file_type = metadata.get("file_type")
                    filename = metadata.get("filename")

                    source = str(source) if source is not None else None
                    file_type = (
                        str(file_type).strip().lower()
                        if file_type is not None
                        else None
                    )
                    filename = (
                        str(filename) if filename is not None else None
                    )

                    existing = connection.execute(
                        """
                        SELECT rowid, content_hash
                        FROM chunks
                        WHERE chunk_id = ?
                        """,
                        (chunk_id,),
                    ).fetchone()

                    if existing and existing["content_hash"] == content_hash:
                        stats["unchanged"] += 1
                        continue

                    if existing:
                        rowid = existing["rowid"]

                        connection.execute(
                            "DELETE FROM chunks_fts WHERE rowid = ?",
                            (rowid,),
                        )

                        connection.execute(
                            """
                            UPDATE chunks
                            SET content = ?,
                                metadata_json = ?,
                                content_hash = ?,
                                source = ?,
                                file_type = ?,
                                filename = ?
                            WHERE rowid = ?
                            """,
                            (
                                content,
                                metadata_json,
                                content_hash,
                                source,
                                file_type,
                                filename,
                                rowid,
                            ),
                        )

                        stats["updated"] += 1

                    else:
                        cursor = connection.execute(
                            """
                            INSERT INTO chunks (
                                chunk_id,
                                content,
                                metadata_json,
                                content_hash,
                                source,
                                file_type,
                                filename
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chunk_id,
                                content,
                                metadata_json,
                                content_hash,
                                source,
                                file_type,
                                filename,
                            ),
                        )

                        rowid = cursor.lastrowid
                        stats["inserted"] += 1

                    connection.execute(
                        """
                        INSERT INTO chunks_fts(rowid, content)
                        VALUES (?, ?)
                        """,
                        (rowid, content),
                    )

            indexed_ids = connection.execute(
                "SELECT COUNT(*) FROM current_ids"
            ).fetchone()[0]

            current_count = int(self.collection.count())

            if indexed_ids != expected_count or current_count != expected_count:
                raise RuntimeError(
                    "Chroma collection changed during synchronization. "
                    "Retry synchronization against a stable collection."
                )

            stale_rows = connection.execute("""
                SELECT rowid
                FROM chunks
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM current_ids
                    WHERE current_ids.chunk_id = chunks.chunk_id
                )
            """).fetchall()

            for row in stale_rows:
                connection.execute(
                    "DELETE FROM chunks_fts WHERE rowid = ?",
                    (row["rowid"],),
                )

            stats["deleted"] = len(stale_rows)

            connection.execute("""
                DELETE FROM chunks
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM current_ids
                    WHERE current_ids.chunk_id = chunks.chunk_id
                )
            """)

            indexed_chunks = connection.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]

            connection.execute(
                """
                INSERT INTO index_state(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (
                    "last_sync_utc",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            connection.commit()

            stats["indexed_chunks"] = indexed_chunks

            logger.info(
                "FTS5 synchronization complete: %s",
                stats,
            )

            return stats

        except Exception:
            connection.rollback()
            logger.exception("Sparse index synchronization failed.")
            raise

        finally:
            connection.close()

    @staticmethod
    def _build_match_query(query: str) -> str:
        """
        Convert user text into a safe FTS5 OR expression.

        User-provided FTS operators and syntax are never executed directly.
        """

        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)

        # Preserve order while removing duplicate terms.
        unique_tokens = list(dict.fromkeys(tokens))

        unique_tokens = unique_tokens[:MAX_QUERY_TERMS]

        return " OR ".join(
            f'"{token}"'
            for token in unique_tokens
        )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        file_type: str | None = None,
        source: str | None = None,
    ) -> list[SparseRetrievalResult]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

        if file_type is not None:
            file_type = file_type.strip().lower()

            if not file_type:
                raise ValueError("file_type cannot be empty.")

        if source is not None:
            source = source.strip()

            if not source:
                raise ValueError("source cannot be empty.")

        match_query = self._build_match_query(query.strip())

        if not match_query:
            return []

        sql = """
            SELECT
                c.chunk_id,
                c.content,
                c.metadata_json,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks AS c
                ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
        """

        parameters: list[Any] = [match_query]

        if file_type is not None:
            sql += " AND c.file_type = ?"
            parameters.append(file_type)

        if source is not None:
            sql += " AND c.source = ?"
            parameters.append(source)

        sql += """
            ORDER BY bm25(chunks_fts) ASC, c.chunk_id ASC
            LIMIT ?
        """

        parameters.append(k)

        connection = self._connect()

        try:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

            return [
                SparseRetrievalResult(
                    chunk_id=row["chunk_id"],
                    document=Document(
                        page_content=row["content"],
                        metadata=json.loads(row["metadata_json"]),
                    ),
                    bm25_score=float(row["bm25_score"]),
                )
                for row in rows
            ]

        except sqlite3.Error as error:
            logger.exception("BM25 sparse retrieval failed.")

            raise RuntimeError(
                "Could not retrieve documents from the SQLite FTS5 index."
            ) from error

        finally:
            connection.close()

    def count(self) -> int:
        """Return the number of indexed chunks."""

        connection = self._connect()

        try:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM chunks"
                ).fetchone()[0]
            )

        finally:
            connection.close()

from __future__ import annotations

import csv
import hashlib
import logging
import sqlite3

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "csv"

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "structured_data.sqlite3"
)

# Only explicitly registered datasets and columns can be accessed.
# This prevents callers from choosing arbitrary SQL tables or columns.
DATASETS = {
    "inventory": {
        "filename": "NexaTech_Inventory.csv",
        "table": "inventory_assets",
        "primary_key": "asset_tag",
        "columns": (
            "asset_tag",
            "asset_type",
            "location",
            "assigned_team",
            "status",
        ),
        "indexed_columns": (
            "asset_tag",
            "asset_type",
            "location",
            "assigned_team",
            "status",
        ),
    },

    "tickets": {
        "filename": "NexaTech_Support_Tickets.csv",
        "table": "support_tickets",
        "primary_key": "ticket_id",
        "columns": (
            "ticket_id",
            "created_at",
            "priority",
            "product",
            "issue",
            "resolution",
            "status",
        ),
        "indexed_columns": (
            "ticket_id",
            "created_at",
            "priority",
            "product",
            "status",
        ),
    },
}


@dataclass(frozen=True)
class IngestionResult:
    dataset: str
    rows: int
    status: str


class StructuredDataStore:
    """
    Validated CSV ingestion and read-only structured querying.

    Supported datasets are explicitly registered in DATASETS.
    CSV replacement is performed in a single SQLite transaction.
    """

    def __init__(
        self,
        database_path: Path = DATABASE_PATH,
        data_directory: Path = DATA_DIRECTORY,
    ) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.data_directory = Path(data_directory).expanduser().resolve()

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(
        self,
        readonly: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        """
        Open a database connection and always close it.

        Write connections commit on success and roll back on failure.
        Read-only connections additionally enable SQLite query_only mode.
        """

        if readonly:
            if not self.database_path.is_file():
                raise FileNotFoundError(self.database_path)

            database_uri = (
                f"{self.database_path.as_uri()}?mode=ro"
            )

            connection = sqlite3.connect(
                database_uri,
                uri=True,
                timeout=15,
            )
        else:
            connection = sqlite3.connect(
                self.database_path,
                timeout=15,
            )

        try:
            connection.row_factory = sqlite3.Row

            connection.execute(
                "PRAGMA busy_timeout = 15000"
            )

            if readonly:
                connection.execute(
                    "PRAGMA query_only = ON"
                )

            yield connection

            if not readonly:
                connection.commit()

        except BaseException:
            if not readonly:
                connection.rollback()

            raise

        finally:
            connection.close()

    def _initialize_database(self) -> None:
        """Create required tables and indexes without deleting existing data."""

        with self._connect() as connection:

            connection.execute(
                "PRAGMA journal_mode = WAL"
            )

            connection.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_state (
                    source_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                )
            """)

            for dataset in DATASETS.values():

                table = dataset["table"]
                columns = dataset["columns"]

                column_definitions = ", ".join(
                    f'"{column}" TEXT NOT NULL'
                    for column in columns
                )

                connection.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{table}" (
                        {column_definitions},
                        source_path TEXT NOT NULL,
                        record_key TEXT NOT NULL,
                        row_number INTEGER NOT NULL,
                        row_hash TEXT NOT NULL,

                        UNIQUE(source_path, record_key)
                    )
                """)

                connection.execute(f"""
                    CREATE INDEX IF NOT EXISTS
                    "idx_{table}_source"
                    ON "{table}" (source_path)
                """)

                for column in dataset["indexed_columns"]:

                    connection.execute(f"""
                        CREATE INDEX IF NOT EXISTS
                        "idx_{table}_{column}"
                        ON "{table}" ("{column}")
                    """)

    @staticmethod
    def _file_hash(path: Path) -> str:
        """Calculate a SHA-256 hash without loading the entire file into memory."""

        digest = hashlib.sha256()

        with path.open("rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(block)

        return digest.hexdigest()

    def _source_path(self, path: Path) -> str:
        """
        Produce stable relative source paths.

        Files under the project's data/raw directory retain paths such as
        csv/NexaTech_Inventory.csv. Custom data directories used in tests
        or other deployments use paths relative to that directory.
        """

        resolved_path = path.resolve()

        roots = (
            PROJECT_ROOT / "data" / "raw",
            self.data_directory,
        )

        for root in roots:
            try:
                return resolved_path.relative_to(
                    root.resolve()
                ).as_posix()
            except ValueError:
                continue

        raise ValueError(
            f"Source file is outside the configured data directory: {path}"
        )

    @staticmethod
    def _validate_row(
        dataset_name: str,
        row: dict[str, str],
    ) -> None:
        """Validate dataset-specific field formats."""

        if dataset_name == "tickets":
            value = row["created_at"]

            try:
                parsed_date = date.fromisoformat(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid ticket date: {value!r}. "
                    "Expected YYYY-MM-DD."
                ) from error

            if parsed_date.isoformat() != value:
                raise ValueError(
                    f"Invalid ticket date format: {value!r}. "
                    "Expected YYYY-MM-DD."
                )

    def _read_csv(
        self,
        dataset_name: str,
        path: Path,
    ) -> list[dict[str, Any]]:
        """Read and validate an entire CSV before database modification."""

        dataset = DATASETS[dataset_name]
        expected_columns = list(dataset["columns"])

        records = []
        seen_keys = set()

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file,
                strict=True,
            )

            if reader.fieldnames != expected_columns:
                raise ValueError(
                    f"Unexpected CSV schema in {path.name}. "
                    f"Expected {expected_columns}, "
                    f"received {reader.fieldnames}"
                )

            for row_number, raw_row in enumerate(reader, start=2):

                if None in raw_row:
                    raise ValueError(
                        f"Extra columns at CSV row {row_number}"
                    )

                row = {
                    column: (raw_row[column] or "").strip()
                    for column in expected_columns
                }

                if any(not value for value in row.values()):
                    raise ValueError(
                        f"Missing required value at CSV row {row_number}"
                    )

                self._validate_row(dataset_name, row)

                record_key = row[dataset["primary_key"]]

                if record_key in seen_keys:
                    raise ValueError(
                        f"Duplicate record ID at CSV row {row_number}: "
                        f"{record_key}"
                    )

                seen_keys.add(record_key)

                row_hash = hashlib.sha256(
                    "\x1f".join(
                        row[column]
                        for column in expected_columns
                    ).encode("utf-8")
                ).hexdigest()

                records.append({
                    "values": row,
                    "record_key": record_key,
                    "row_number": row_number,
                    "row_hash": row_hash,
                })

        return records

    @staticmethod
    def _existing_records_match(
        connection: sqlite3.Connection,
        table: str,
        source_path: str,
        records: list[dict[str, Any]],
    ) -> bool:
        """Check individual record keys and hashes, not only row counts."""

        existing_rows = connection.execute(
            f"""
            SELECT record_key, row_hash
            FROM "{table}"
            WHERE source_path = ?
            """,
            (source_path,),
        ).fetchall()

        if len(existing_rows) != len(records):
            return False

        existing = {
            row["record_key"]: row["row_hash"]
            for row in existing_rows
        }

        expected = {
            record["record_key"]: record["row_hash"]
            for record in records
        }

        return existing == expected

    def ingest_dataset(
        self,
        dataset_name: str,
    ) -> IngestionResult:
        """Validate and transactionally ingest one registered CSV dataset."""

        if (
            not isinstance(dataset_name, str)
            or dataset_name not in DATASETS
        ):
            raise ValueError(
                f"Unsupported dataset: {dataset_name!r}"
            )

        dataset = DATASETS[dataset_name]
        path = self.data_directory / dataset["filename"]

        if not path.is_file():
            raise FileNotFoundError(path)

        source_path = self._source_path(path)

        # Hash before and after parsing to detect changes during ingestion.
        current_hash = self._file_hash(path)

        # No database changes occur until the entire file is validated.
        records = self._read_csv(dataset_name, path)

        if self._file_hash(path) != current_hash:
            raise RuntimeError(
                f"Source file changed while being read: {path}"
            )

        table = dataset["table"]
        columns = dataset["columns"]

        with self._connect() as connection:

            # Serialize competing writers before checking or replacing data.
            connection.execute("BEGIN IMMEDIATE")

            previous = connection.execute(
                """
                SELECT file_hash, row_count
                FROM ingestion_state
                WHERE source_path = ?
                """,
                (source_path,),
            ).fetchone()

            if (
                previous
                and previous["file_hash"] == current_hash
                and previous["row_count"] == len(records)
                and self._existing_records_match(
                    connection,
                    table,
                    source_path,
                    records,
                )
            ):
                logger.info(
                    "Unchanged dataset: %s",
                    source_path,
                )

                return IngestionResult(
                    dataset=dataset_name,
                    rows=len(records),
                    status="skipped",
                )

            # Replace this source's records atomically.
            connection.execute(
                f'DELETE FROM "{table}" WHERE source_path = ?',
                (source_path,),
            )

            insert_columns = [
                *columns,
                "source_path",
                "record_key",
                "row_number",
                "row_hash",
            ]

            column_names = ", ".join(
                f'"{column}"'
                for column in insert_columns
            )

            placeholders = ", ".join(
                "?" for _ in insert_columns
            )

            insert_sql = f"""
                INSERT INTO "{table}" ({column_names})
                VALUES ({placeholders})
            """

            def values_to_insert():
                for record in records:
                    values = [
                        record["values"][column]
                        for column in columns
                    ]

                    values.extend([
                        source_path,
                        record["record_key"],
                        record["row_number"],
                        record["row_hash"],
                    ])

                    yield values

            connection.executemany(
                insert_sql,
                values_to_insert(),
            )

            connection.execute(
                """
                INSERT INTO ingestion_state (
                    source_path,
                    file_hash,
                    row_count,
                    indexed_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_path)
                DO UPDATE SET
                    file_hash = excluded.file_hash,
                    row_count = excluded.row_count,
                    indexed_at = excluded.indexed_at
                """,
                (
                    source_path,
                    current_hash,
                    len(records),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        logger.info(
            "Ingested %s: %d records",
            source_path,
            len(records),
        )

        return IngestionResult(
            dataset=dataset_name,
            rows=len(records),
            status="indexed",
        )

    def ingest_all(
        self,
        *,
        prune_missing: bool = False,
    ) -> list[IngestionResult]:
        """
        Ingest every registered dataset.

        When prune_missing=True, records for known CSV files that no longer
        exist are removed from SQLite. This is explicit to avoid accidental
        deletion if a data directory is unavailable.
        """

        if not isinstance(prune_missing, bool):
            raise TypeError("prune_missing must be a boolean")

        if not self.data_directory.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_directory}"
            )

        results = []

        for dataset_name, dataset in DATASETS.items():

            path = self.data_directory / dataset["filename"]

            if path.is_file():
                results.append(
                    self.ingest_dataset(dataset_name)
                )
                continue

            if not prune_missing:
                raise FileNotFoundError(path)

            source_path = self._source_path(path)
            table = dataset["table"]

            with self._connect() as connection:

                connection.execute("BEGIN IMMEDIATE")

                existing_count = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE source_path = ?',
                    (source_path,),
                ).fetchone()[0]

                previous = connection.execute(
                    """
                    SELECT 1
                    FROM ingestion_state
                    WHERE source_path = ?
                    """,
                    (source_path,),
                ).fetchone()

                connection.execute(
                    f'DELETE FROM "{table}" WHERE source_path = ?',
                    (source_path,),
                )

                connection.execute(
                    """
                    DELETE FROM ingestion_state
                    WHERE source_path = ?
                    """,
                    (source_path,),
                )

            was_present = existing_count > 0 or previous is not None

            status = "deleted" if was_present else "missing"

            results.append(
                IngestionResult(
                    dataset=dataset_name,
                    rows=0,
                    status=status,
                )
            )

            if was_present:
                logger.info(
                    "Removed missing source from SQLite: %s",
                    source_path,
                )

        return results

    def query(
        self,
        dataset_name: str,
        filters: dict[str, str] | None = None,
        group_by: str | None = None,
        count: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Execute a safe structured query against a registered dataset.

        Supported operations:
        - Exact equality filters.
        - Row counts.
        - Grouped counts.
        - Bounded record retrieval.

        All filter values are passed as SQL parameters.
        """

        if (
            not isinstance(dataset_name, str)
            or dataset_name not in DATASETS
        ):
            raise ValueError(
                f"Unsupported dataset: {dataset_name!r}"
            )

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 1000
        ):
            raise ValueError(
                "limit must be an integer between 1 and 1000"
            )

        if not isinstance(count, bool):
            raise TypeError("count must be a boolean")

        if group_by is not None and count:
            raise ValueError(
                "Use either group_by or count, not both. "
                "Grouped queries already return counts."
            )

        dataset = DATASETS[dataset_name]
        table = dataset["table"]
        allowed_columns = set(dataset["columns"])

        if filters is None:
            filters = {}

        if not isinstance(filters, dict):
            raise TypeError("filters must be a dictionary")

        if not set(filters).issubset(allowed_columns):
            raise ValueError(
                "Unsupported filter column. "
                f"Allowed columns: {sorted(allowed_columns)}"
            )

        for column, value in filters.items():

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Filter value for {column!r} "
                    "must be a non-empty string"
                )

        if group_by is not None:
            if (
                not isinstance(group_by, str)
                or group_by not in allowed_columns
            ):
                raise ValueError("Unsupported grouping column")

        where_parts = []
        parameters: list[Any] = []

        for column, value in filters.items():
            where_parts.append(f'"{column}" = ?')
            parameters.append(value.strip())

        where_clause = (
            " WHERE " + " AND ".join(where_parts)
            if where_parts
            else ""
        )

        if group_by is not None:

            sql = f"""
                SELECT "{group_by}" AS value, COUNT(*) AS total
                FROM "{table}"
                {where_clause}
                GROUP BY "{group_by}"
                ORDER BY total DESC, value ASC
                LIMIT ?
            """

            parameters.append(limit)

        elif count:

            sql = f"""
                SELECT COUNT(*) AS total
                FROM "{table}"
                {where_clause}
            """

        else:

            sql = f"""
                SELECT *
                FROM "{table}"
                {where_clause}
                ORDER BY row_number ASC
                LIMIT ?
            """

            parameters.append(limit)

        with self._connect(readonly=True) as connection:
            rows = connection.execute(
                sql,
                parameters,
            ).fetchall()

        return [dict(row) for row in rows]


def get_structured_store() -> StructuredDataStore:
    """Create a structured data store using the configured project paths."""

    return StructuredDataStore()
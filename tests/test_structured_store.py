import sqlite3

import pytest

from retrieval.structured_store import StructuredDataStore


INVENTORY_CSV = """asset_tag,asset_type,location,assigned_team,status
NX-001,Laptop,Nairobi Office,Engineering,In use
NX-002,Laptop,Remote,Support,In use
NX-003,Server,Cloud Region A,Operations,Active
NX-004,Network Switch,Nairobi Office,IT,Maintenance
"""

TICKETS_CSV = """ticket_id,created_at,priority,product,issue,resolution,status
NT-1001,2026-01-08,P1,API,Repeated 429 responses,Implemented exponential backoff,Closed
NT-1002,2026-01-09,P1,Atlas,Workspace unavailable,Restarted service,Open
NT-1003,2026-01-10,P2,API,Search delay,Rebuilt index,Closed
"""


@pytest.fixture
def store(tmp_path):
    """Create an isolated structured database and CSV directory."""

    data_directory = tmp_path / "csv"
    data_directory.mkdir()

    (data_directory / "NexaTech_Inventory.csv").write_text(
        INVENTORY_CSV,
        encoding="utf-8",
    )

    (data_directory / "NexaTech_Support_Tickets.csv").write_text(
        TICKETS_CSV,
        encoding="utf-8",
    )

    database_path = (
        tmp_path / "processed" / "structured_data.sqlite3"
    )

    return StructuredDataStore(
        database_path=database_path,
        data_directory=data_directory,
    )


def test_ingest_all_datasets(store):
    results = store.ingest_all()

    assert len(results) == 2

    assert results[0].dataset == "inventory"
    assert results[0].rows == 4
    assert results[0].status == "indexed"

    assert results[1].dataset == "tickets"
    assert results[1].rows == 3
    assert results[1].status == "indexed"


def test_unchanged_datasets_are_skipped(store):
    store.ingest_all()

    results = store.ingest_all()

    assert all(result.status == "skipped" for result in results)


def test_changed_dataset_is_replaced_without_duplicates(store):
    store.ingest_all()

    path = store.data_directory / "NexaTech_Inventory.csv"

    updated_csv = INVENTORY_CSV + (
        "NX-005,Monitor,Nairobi Office,Finance,Standby\n"
    )

    path.write_text(updated_csv, encoding="utf-8")

    result = store.ingest_dataset("inventory")

    assert result.status == "indexed"
    assert result.rows == 5

    records = store.query("inventory")

    assert len(records) == 5

    asset_tags = {
        record["asset_tag"]
        for record in records
    }

    assert len(asset_tags) == 5
    assert "NX-005" in asset_tags


def test_invalid_schema_does_not_destroy_existing_records(store):
    store.ingest_all()

    path = store.data_directory / "NexaTech_Inventory.csv"

    invalid_csv = INVENTORY_CSV.replace(
        "asset_tag,asset_type",
        "wrong_column,asset_type",
        1,
    )

    path.write_text(invalid_csv, encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected CSV schema"):
        store.ingest_dataset("inventory")

    records = store.query("inventory")

    assert len(records) == 4
    assert records[0]["asset_tag"] == "NX-001"


def test_duplicate_record_ids_are_rejected(store):
    store.ingest_all()

    path = store.data_directory / "NexaTech_Inventory.csv"

    duplicate_csv = INVENTORY_CSV + (
        "NX-001,Laptop,Nairobi Office,Engineering,In use\n"
    )

    path.write_text(duplicate_csv, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate record ID"):
        store.ingest_dataset("inventory")

    assert len(store.query("inventory")) == 4


def test_invalid_ticket_date_is_rejected(store):
    store.ingest_all()

    path = store.data_directory / "NexaTech_Support_Tickets.csv"

    invalid_csv = TICKETS_CSV.replace(
        "2026-01-08",
        "not-a-date",
        1,
    )

    path.write_text(invalid_csv, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid ticket date"):
        store.ingest_dataset("tickets")

    assert len(store.query("tickets")) == 3


def test_exact_filtering(store):
    store.ingest_all()

    records = store.query(
        "inventory",
        filters={"status": "Maintenance"},
    )

    assert len(records) == 1
    assert records[0]["asset_tag"] == "NX-004"


def test_count_query(store):
    store.ingest_all()

    result = store.query(
        "tickets",
        filters={"priority": "P1"},
        count=True,
    )

    assert result == [{"total": 2}]


def test_grouped_query(store):
    store.ingest_all()

    result = store.query(
        "inventory",
        group_by="asset_type",
    )

    assert result == [
        {"value": "Laptop", "total": 2},
        {"value": "Network Switch", "total": 1},
        {"value": "Server", "total": 1},
    ]


def test_grouped_query_can_be_filtered(store):
    store.ingest_all()

    result = store.query(
        "tickets",
        filters={"priority": "P1"},
        group_by="status",
    )

    assert result == [
        {"value": "Closed", "total": 1},
        {"value": "Open", "total": 1},
    ]


def test_filter_values_are_parameterized(store):
    store.ingest_all()

    result = store.query(
        "inventory",
        filters={"status": "' OR 1=1 --"},
    )

    assert result == []


def test_unsupported_filter_column_is_rejected(store):
    store.ingest_all()

    with pytest.raises(ValueError, match="Unsupported filter column"):
        store.query(
            "inventory",
            filters={"unknown_column": "value"},
        )


def test_invalid_limits_are_rejected(store):
    store.ingest_all()

    for invalid_limit in (True, False, 0, -1, 1001, "5"):
        with pytest.raises(ValueError, match="limit"):
            store.query(
                "inventory",
                limit=invalid_limit,
            )


def test_count_and_group_by_cannot_be_combined(store):
    store.ingest_all()

    with pytest.raises(ValueError, match="either group_by or count"):
        store.query(
            "inventory",
            group_by="asset_type",
            count=True,
        )


def test_connection_is_closed_after_context_exit(store):
    with store._connect() as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_readonly_connection_rejects_database_changes(store):
    with store._connect(readonly=True) as connection:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute(
                "CREATE TABLE should_not_be_created (id INTEGER)"
            )


def test_missing_source_can_be_pruned_explicitly(store):
    store.ingest_all()

    inventory_path = (
        store.data_directory / "NexaTech_Inventory.csv"
    )

    inventory_path.unlink()

    results = store.ingest_all(prune_missing=True)

    inventory_result = next(
        result
        for result in results
        if result.dataset == "inventory"
    )

    assert inventory_result.status == "deleted"
    assert store.query("inventory", count=True) == [{"total": 0}]

    # The other CSV still exists and should remain indexed.
    assert store.query("tickets", count=True) == [{"total": 3}]


def test_transaction_rolls_back_on_failure(store):
    with pytest.raises(RuntimeError, match="simulated failure"):
        with store._connect() as connection:
            connection.execute("BEGIN")

            connection.execute("""
                CREATE TABLE rollback_test (
                    id INTEGER
                )
            """)

            raise RuntimeError("simulated failure")

    with store._connect(readonly=True) as connection:
        tables = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'rollback_test'
            """
        ).fetchall()

    assert tables == []
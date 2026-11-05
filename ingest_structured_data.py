
import logging

from retrieval.structured_store import get_structured_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main():
    store = get_structured_store()

    results = store.ingest_all()

    print("\n--- STRUCTURED DATA INGESTION ---")

    for result in results:
        print(
            f"{result.dataset}: "
            f"{result.rows} rows | {result.status}"
        )


if __name__ == "__main__":
    main()

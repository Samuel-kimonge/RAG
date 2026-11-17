
from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from typing import Any

from evaluation.metrics import evaluate_ranking

logger = logging.getLogger(__name__)


def percentile(values: list[float], p: float) -> float:
    """Calculate a linearly interpolated percentile."""

    if not values:
        return 0.0

    if not 0 <= p <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100

    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)

    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def aggregate_metrics(
    records: list[dict[str, Any]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    """Calculate macro-average ranking metrics."""

    successful = [
        record for record in records
        if record["status"] == "success"
        and record["answerable"]
    ]

    output = {}

    for k in ks:
        values = [
            record["metrics"][str(k)]
            for record in successful
        ]

        names = (
            "precision",
            "recall",
            "mrr",
            "ndcg",
            "hit_rate",
        )

        output[str(k)] = {
            name: (
                statistics.fmean(item[name] for item in values)
                if values else None
            )
            for name in names
        }

    return output


class RetrievalEvaluator:
    """Evaluate the existing KnowledgeBase retrieval interface."""

    def __init__(
        self,
        knowledge_base: Any,
        ks: tuple[int, ...] = (1, 3, 5, 10),
    ) -> None:
        if not ks or any(
            isinstance(k, bool) or not isinstance(k, int) or k < 1
            for k in ks
        ):
            raise ValueError("K values must be positive integers.")

        if len(ks) != len(set(ks)):
            raise ValueError("K values must be unique.")

        self.knowledge_base = knowledge_base
        self.ks = tuple(sorted(ks))
        self.max_k = max(self.ks)

    def evaluate(
        self,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:

        cases = dataset["test_cases"]
        records = []
        latencies = []

        for index, case in enumerate(cases, start=1):
            case_id = case["id"]
            query = case["query"]
            answerable = case["answerable"]
            relevant_chunks = case["relevant_chunks"]

            print(
                f"[{index}/{len(cases)}] Evaluating {case_id}...",
                flush=True,
            )

            started = time.perf_counter()

            try:
                retrieved = self.knowledge_base.search(
                    query,
                    k=self.max_k,
                )

                latency_ms = (
                    time.perf_counter() - started
                ) * 1000

                latencies.append(latency_ms)

                retrieved_ids = [
                    result.chunk_id for result in retrieved
                ]

                record = {
                    "id": case_id,
                    "query": query,
                    "category": case["category"],
                    "answerable": answerable,
                    "status": "success",
                    "latency_ms": latency_ms,
                    "retrieved_count": len(retrieved),
                    "retrieved": [
                            {
                                 "rank": rank,
                                 "chunk_id": result.chunk_id,
                                 "distance": getattr(
                                    result, "distance", None
                                    ),
                                 "source": result.metadata.get("source"),

                                 # Hybrid retrieval diagnostics.
                                 "fused_score": getattr(
                                    result, "fused_score", None
                                    ),
                                 "dense_rank": getattr(
                                    result, "dense_rank", None
                                    ),
                                 "sparse_rank": getattr(
                                    result, "sparse_rank", None
                                    ),
                                 "bm25_score": getattr(
                                    result, "bm25_score", None
                                    ),
                            }
                            for rank, result in enumerate(
                                 retrieved,
                                 start=1,
                            )
                    ],
                }

                if answerable:
                    record["metrics"] = evaluate_ranking(
                        retrieved_ids=retrieved_ids,
                        relevant_chunks=relevant_chunks,
                        ks=self.ks,
                    )
                else:
                    record["unanswerable_evaluation"] = {
                        "returned_results": bool(retrieved),
                        "empty_result": not bool(retrieved),
                    }

            except Exception as error:
                latency_ms = (
                    time.perf_counter() - started
                ) * 1000

                latencies.append(latency_ms)

                logger.exception(
                    "Evaluation failed for query %s",
                    case_id,
                )

                record = {
                    "id": case_id,
                    "query": query,
                    "category": case["category"],
                    "answerable": answerable,
                    "status": "failed",
                    "latency_ms": latency_ms,
                    "retrieved_count": 0,
                    "retrieved": [],
                    "metrics": None,
                    "error": str(error),
                }

            records.append(record)

        successful = [
            record for record in records
            if record["status"] == "success"
        ]

        unanswerable_records = [
            record for record in successful
            if not record["answerable"]
        ]

        unanswerable_records = [
            record for record in successful
            if not record["answerable"]
        ]

        categories = defaultdict(list)

        for record in records:
            categories[record["category"]].append(record)

        category_summary = {}

        for category, category_records in sorted(categories.items()):
            category_summary[category] = {
                "total_queries": len(category_records),
                "successful_queries": sum(
                    r["status"] == "success"
                    for r in category_records
                ),
                "answerable_queries": sum(
                    r["answerable"]
                    for r in category_records
                ),
                "metrics": aggregate_metrics(
                    category_records,
                    self.ks,
                ),
            }

        unanswerable_with_results = sum(
            bool(record["unanswerable_evaluation"]["returned_results"])
            for record in unanswerable_records
        )

        summary = {
            "total_queries": len(records),
            "successful_queries": len(successful),
            "failed_queries": len(records) - len(successful),
            "answerable_queries": sum(
                case["answerable"] for case in cases
            ),
            "unanswerable_queries": sum(
                not case["answerable"] for case in cases
            ),
            "ranking_metrics": aggregate_metrics(
                records,
                self.ks,
            ),
            "unanswerable": {
                "successful_queries": len(unanswerable_records),
                "queries_returning_results": unanswerable_with_results,
                "queries_returning_no_results": (
                    len(unanswerable_records) - unanswerable_with_results
                ),
                "nonempty_result_rate": (
                    unanswerable_with_results / len(unanswerable_records)
                    if unanswerable_records else 0.0
                ),
            },
            "latency_ms": {
                "mean": statistics.fmean(latencies) if latencies else 0.0,
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "max": max(latencies, default=0.0),
            },
            "empty_result_rate": (
                sum(
                    r["retrieved_count"] == 0
                    for r in successful
                ) / len(successful)
                if successful else 0.0
            ),
            "category_summary": category_summary,
        }

        return {
            "dataset_name": dataset.get("name"),
            "dataset_version": dataset.get("version"),
            "collection": dataset.get("collection"),
            "embedding_model": dataset.get("embedding_model"),
            "k_values": list(self.ks),
            "summary": summary,
            "queries": records,
        }

from __future__ import annotations

import argparse
import json
import statistics

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DENSE = ROOT / "reports/retrieval/dense_recheck/results.json"
DEFAULT_HYBRID = ROOT / "reports/retrieval/hybrid_recheck/results.json"
DEFAULT_DATASET = ROOT / "evaluation/datasets/retrieval_test.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fmt(value: Any, width: int = 9) -> str:
    if value is None:
        return "N/A".rjust(width)

    return f"{value:.4f}".rjust(width)


def first_relevant_rank(
    record: dict[str, Any],
    relevant_ids: set[str],
) -> int | None:

    for result in record.get("retrieved", []):
        if result["chunk_id"] in relevant_ids:
            return result["rank"]

    return None


def ranking_metrics_comparison(dense, hybrid):
    print("\n========== OVERALL RANKING COMPARISON ==========")

    print(
        f"{'Metric':<14}{'K':>4}"
        f"{'Dense':>12}{'Hybrid':>12}{'Change':>12}"
    )

    names = (
        "precision",
        "recall",
        "mrr",
        "ndcg",
        "hit_rate",
    )

    dense_metrics = dense["summary"]["ranking_metrics"]
    hybrid_metrics = hybrid["summary"]["ranking_metrics"]

    for k in dense_metrics:
        for name in names:
            d = dense_metrics[k][name]
            h = hybrid_metrics[k][name]

            change = h - d

            print(
                f"{name:<14}{k:>4}"
                f"{d:>12.4f}{h:>12.4f}{change:>+12.4f}"
            )


def category_comparison(dense, hybrid):
    print("\n========== CATEGORY COMPARISON @3 ==========")

    categories = sorted(
        set(dense["summary"]["category_summary"])
        | set(hybrid["summary"]["category_summary"])
    )

    print(
        f"{'Category':<22}"
        f"{'Dense Recall':>14}"
        f"{'Hybrid Recall':>15}"
        f"{'Dense MRR':>12}"
        f"{'Hybrid MRR':>12}"
    )

    for category in categories:
        dense_data = dense["summary"]["category_summary"].get(category, {})
        hybrid_data = hybrid["summary"]["category_summary"].get(category, {})

        dm = dense_data.get("metrics", {}).get("3", {})
        hm = hybrid_data.get("metrics", {}).get("3", {})

        print(
            f"{category:<22}"
            f"{fmt(dm.get('recall'), 14)}"
            f"{fmt(hm.get('recall'), 15)}"
            f"{fmt(dm.get('mrr'), 12)}"
            f"{fmt(hm.get('mrr'), 12)}"
        )


def query_comparison(dense, hybrid, dataset):
    print("\n========== INDIVIDUAL QUERY COMPARISON ==========")

    dense_queries = {
        item["id"]: item
        for item in dense["queries"]
    }

    hybrid_queries = {
        item["id"]: item
        for item in hybrid["queries"]
    }

    cases = dataset["test_cases"]

    if set(dense_queries) != set(hybrid_queries):
        raise ValueError(
            "Dense and hybrid reports contain different query IDs."
        )

    print(
        f"{'ID':<19}"
        f"{'Dense rank':>12}"
        f"{'Hybrid rank':>13}"
        f"{'MRR change':>12}"
        f"{'Overlap@10':>13}"
    )

    improvements = []
    regressions = []
    latency_deltas = []

    for case in cases:
        case_id = case["id"]

        dense_record = dense_queries[case_id]
        hybrid_record = hybrid_queries[case_id]

        d_ids = {
            item["chunk_id"]
            for item in dense_record.get("retrieved", [])[:10]
        }

        h_ids = {
            item["chunk_id"]
            for item in hybrid_record.get("retrieved", [])[:10]
        }

        overlap = len(d_ids & h_ids)

        d_latency = dense_record.get("latency_ms", 0.0)
        h_latency = hybrid_record.get("latency_ms", 0.0)

        latency_deltas.append(h_latency - d_latency)

        if not case["answerable"]:
            print(
                f"{case_id:<19}"
                f"{'N/A':>12}"
                f"{'N/A':>13}"
                f"{'N/A':>12}"
                f"{overlap:>13}"
            )
            continue

        relevant_ids = {
            item["chunk_id"]
            for item in case["relevant_chunks"]
        }

        d_rank = first_relevant_rank(
            dense_record,
            relevant_ids,
        )

        h_rank = first_relevant_rank(
            hybrid_record,
            relevant_ids,
        )

        d_mrr = (
            dense_record.get("metrics", {})
            .get("3", {})
            .get("mrr", 0.0)
        )

        h_mrr = (
            hybrid_record.get("metrics", {})
            .get("3", {})
            .get("mrr", 0.0)
        )

        delta = h_mrr - d_mrr

        item = (case_id, delta, d_rank, h_rank)

        if delta > 0:
            improvements.append(item)
        elif delta < 0:
            regressions.append(item)

        print(
            f"{case_id:<19}"
            f"{str(d_rank or '-'):>12}"
            f"{str(h_rank or '-'):>13}"
            f"{delta:>+12.4f}"
            f"{overlap:>13}"
        )

    print("\nRanking changes:")
    print(f"  Improved queries:   {len(improvements)}")
    print(f"  Regressed queries:  {len(regressions)}")

    print("\nImproved queries:")

    for item in improvements:
        print(
            f"  {item[0]}: "
            f"dense rank={item[2]}, "
            f"hybrid rank={item[3]}, "
            f"MRR change={item[1]:+.4f}"
        )

    print("\nRegressed queries:")

    for item in regressions:
        print(
            f"  {item[0]}: "
            f"dense rank={item[2]}, "
            f"hybrid rank={item[3]}, "
            f"MRR change={item[1]:+.4f}"
        )

    print("\n========== PAIRED LATENCY COMPARISON ==========")

    if latency_deltas:
        print(
            "Mean hybrid-minus-dense latency: "
            f"{statistics.fmean(latency_deltas):.2f} ms"
        )
        print(
            "Median hybrid-minus-dense latency: "
            f"{statistics.median(latency_deltas):.2f} ms"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare dense and hybrid retrieval reports."
    )

    parser.add_argument(
        "--dense",
        type=Path,
        default=DEFAULT_DENSE,
    )

    parser.add_argument(
        "--hybrid",
        type=Path,
        default=DEFAULT_HYBRID,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    args = parser.parse_args()

    dense = load_json(args.dense)
    hybrid = load_json(args.hybrid)
    dataset = load_json(args.dataset)

    ranking_metrics_comparison(dense, hybrid)
    category_comparison(dense, hybrid)
    query_comparison(dense, hybrid, dataset)


if __name__ == "__main__":
    main()

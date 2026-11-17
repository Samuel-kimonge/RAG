
from __future__ import annotations

import argparse
import json
import logging
import sys

from datetime import datetime, timezone
from pathlib import Path

from evaluation.evaluator.retrieval_evaluator import RetrievalEvaluator
from evaluation.reporting import write_reports

from retrieval.vector_store import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    get_knowledge_base,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "retrieval_test.json"
)

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "retrieval"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run dense or hybrid RAG retrieval evaluation."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--method",
        choices=("dense", "hybrid", "reranked"),
        default="dense",
        help="Retrieval method to benchmark.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional report directory override.",
    )

    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
    )

    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=1,
        help="Hybrid candidate count multiplier.",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100,
        help="Maximum number of candidates per retriever.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant.",
    )

    parser.add_argument(
        "--reranker-model-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--reranker-max-query-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--reranker-intra-op-threads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--reranker-inter-op-threads",
        type=int,
        default=1,
    )
    return parser.parse_args()


def load_dataset(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise TypeError(
            "Dataset root must be a JSON object."
        )

    if not isinstance(dataset.get("test_cases"), list):
        raise TypeError(
            "Dataset test_cases must be a list."
        )

    if not dataset["test_cases"]:
        raise ValueError(
            "Dataset contains no test cases."
        )

    return dataset


def validate_arguments(args) -> tuple[int, ...]:
    ks = tuple(args.ks)

    if not ks or any(k < 1 for k in ks):
        raise ValueError(
            "K values must be positive integers."
        )

    if len(ks) != len(set(ks)):
        raise ValueError(
            "K values must be unique."
        )

    if max(ks) > 100:
        raise ValueError(
            "Maximum supported K is 100."
        )

    if args.candidate_multiplier < 1:
        raise ValueError(
            "candidate-multiplier must be positive."
        )

    if args.max_candidates < 1:
        raise ValueError(
            "max-candidates must be positive."
        )

    if args.rrf_k < 1:
        raise ValueError(
            "rrf-k must be positive."
        )

    return ks


def build_retriever(args):
    """Build the selected retrieval architecture."""

    dense_store = get_knowledge_base()

    if args.method == "dense":
        return dense_store

    from retrieval.hybrid_store import HybridKnowledgeBase

    hybrid_store = HybridKnowledgeBase(
        dense_store=dense_store,
        candidate_multiplier=args.candidate_multiplier,
        max_candidates=args.max_candidates,
        rrf_k=args.rrf_k,
    )

    if args.method == "hybrid":
        return hybrid_store

    from retrieval.reranker import (
        CrossEncoderReranker,
        RerankedHybridKnowledgeBase,
    )

    reranker_options = {
        "batch_size": args.reranker_batch_size,
        "max_length": args.reranker_max_length,
        "max_query_tokens": args.reranker_max_query_tokens,
        "intra_op_num_threads": args.reranker_intra_op_threads,
        "inter_op_num_threads": args.reranker_inter_op_threads,
    }

    if args.reranker_model_dir is not None:
        reranker_options["model_dir"] = args.reranker_model_dir

    reranker = CrossEncoderReranker(**reranker_options)

    return RerankedHybridKnowledgeBase(
        hybrid_store=hybrid_store,
        reranker=reranker,
    )


def print_summary(
    summary: dict,
    method: str,
) -> None:

    title = {
         "dense": "DENSE BASELINE",
         "hybrid": "HYBRID",
         "reranked": "HYBRID + CROSS-ENCODER",
     }[method]

    print(f"\n========== {title} RETRIEVAL RESULTS ==========")

    print(f"Total queries:      {summary['total_queries']}")
    print(f"Successful:         {summary['successful_queries']}")
    print(f"Failed:             {summary['failed_queries']}")

    print("\nRanking metrics (macro averages):")

    for k, metrics in summary["ranking_metrics"].items():
        print(f"\nK = {k}")

        for name, value in metrics.items():
            print(f"  {name:<12}: {value:.4f}")

    print("\nUnanswerable query diagnostics:")

    for name, value in summary["unanswerable"].items():
        print(f"  {name:<30}: {value}")

    print("\nRetrieval latency (milliseconds):")

    for name, value in summary["latency_ms"].items():
        print(f"  {name:<12}: {value:.2f}")

    rerank_latency = summary.get("reranking_latency_ms")

    if rerank_latency:
        print("\nCross-encoder inference latency (milliseconds):")

        for name, value in rerank_latency.items():
            if name == "successful_reranking_queries":
                print(f"  {name:<30}: {value}")
            else:
                print(f"  {name:<30}: {value:.2f}")

    print("\nCategory performance:")

    # Show category-level recall and MRR at K=3 when available.
    metric_k = (
        "3"
        if "3" in summary["ranking_metrics"]
        else next(iter(summary["ranking_metrics"]))
    )

    print(
        f"  {'Category':<22}"
        f"{'Queries':>9}"
        f"{'Success':>9}"
        f"{'Recall@' + metric_k:>12}"
        f"{'MRR@' + metric_k:>10}"
    )

    for category, data in summary["category_summary"].items():
        metrics = data["metrics"][metric_k]

        recall = metrics["recall"]
        mrr = metrics["mrr"]

        recall_text = (
             f"{recall:.4f}" if recall is not None else "N/A"
        )

        mrr_text = (
              f"{mrr:.4f}" if mrr is not None else "N/A"
        )

        print(
             f"  {category:<22}"
             f"{data['total_queries']:>9}"
             f"{data['successful_queries']:>9}"
             f"{recall_text:>12}"
             f"{mrr_text:>10}"
        )

    print("================================================\n")


def main() -> int:
    args = parse_args()

    started_at = datetime.now(timezone.utc)

    try:
        dataset = load_dataset(args.dataset)
        ks = validate_arguments(args)

        if dataset.get("collection") != COLLECTION_NAME:
            raise ValueError(
                f"Dataset collection '{dataset.get('collection')}' "
                f"does not match active collection '{COLLECTION_NAME}'."
            )

        if dataset.get("embedding_model") != EMBEDDING_MODEL:
            raise ValueError(
                f"Dataset embedding model '{dataset.get('embedding_model')}' "
                f"does not match configured model '{EMBEDDING_MODEL}'."
            )

        output_dir = args.output_dir
        if output_dir is None:
            folder = {
                "dense": "baseline",
                "hybrid": "hybrid",
                "reranked": "reranked",
            }[args.method]

        output_dir = DEFAULT_OUTPUT_ROOT / folder

        print("\nInitializing existing knowledge base...")

        retriever = build_retriever(args)

        evaluator = RetrievalEvaluator(
            knowledge_base=retriever,
            ks=ks,
        )

        print(
            f"Starting {args.method} evaluation "
            f"using {len(dataset['test_cases'])} queries...\n"
        )

        report = evaluator.evaluate(dataset)

        completed_at = datetime.now(timezone.utc)

        run_metadata = {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "retrieval_method": (
                "hybrid_dense_bm25_rrf"
                if args.method == "hybrid"
                else "dense_vector_search"
            ),
            "collection": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "k_values": list(evaluator.ks),
        }

        if args.method in ("hybrid", "reranked"):
            run_metadata.update({
                "candidate_multiplier": args.candidate_multiplier,
                "max_candidates": args.max_candidates,
                "rrf_k": args.rrf_k,
                "sparse_index_path": str(
                    retriever.sparse_store.db_path
                    if args.method == "hybrid"
                    else retriever.hybrid_store.sparse_store.db_path
                ),
            })

            sparse_store = (
                retriever.sparse_store
                if args.method == "hybrid"
                else retriever.hybrid_store.sparse_store
            )

            run_metadata["sparse_sync_stats"] = (
                sparse_store.last_sync_stats
            )

        if args.method == "reranked":
            reranker = retriever.reranker

            run_metadata["reranker"] = {
                "model": str(reranker.model_path),
                "batch_size": reranker.batch_size,
                "max_length": reranker.max_length,
                "max_query_tokens": reranker.max_query_tokens,
                "execution_provider": "CPUExecutionProvider",
            }

        report["run_metadata"] = run_metadata

        results_path, summary_path = write_reports(
            report,
            output_dir,
        )

        print_summary(
            summary=report["summary"],
            method=args.method
            )
        
        print("Reports generated:")
        print(f"  Detailed results: {results_path}")
        print(f"  Summary:          {summary_path}")

        if report["summary"]["failed_queries"]:
            logger.warning(
                "Evaluation completed with failed queries."
            )
            return 1

        return 0

    except Exception:
        logger.exception(
            "%s retrieval benchmark failed.",
            args.method.capitalize(),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

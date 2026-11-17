
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _validate_k_values(ks: Sequence[int]) -> tuple[int, ...]:
    values = tuple(ks)

    if not values:
        raise ValueError("At least one K value is required.")

    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in values):
        raise ValueError("Every K value must be a positive integer.")

    if len(values) != len(set(values)):
        raise ValueError("K values must be unique.")

    return tuple(sorted(values))


def _relevance_map(
    relevant_chunks: Sequence[dict[str, Any]],
) -> dict[str, int]:
    relevance = {}

    for chunk in relevant_chunks:
        chunk_id = chunk["chunk_id"]
        score = chunk["relevance"]

        if chunk_id in relevance:
            raise ValueError(f"Duplicate relevant chunk ID: {chunk_id}")

        if isinstance(score, bool) or not isinstance(score, int) or score < 1:
            raise ValueError(f"Invalid relevance score for {chunk_id}.")

        relevance[chunk_id] = score

    return relevance


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    relevant_chunks: Sequence[dict[str, Any]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, dict[str, float]]:
    """
    Calculate ranking metrics at multiple cutoffs.

    Precision@K uses K as its denominator, even when fewer than K
    documents are returned.

    nDCG uses exponential gain: 2**relevance - 1.
    """

    cutoffs = _validate_k_values(ks)
    relevance = _relevance_map(relevant_chunks)

    if not relevance:
        raise ValueError(
            "Ranking metrics require at least one relevant chunk."
        )

    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("Retrieved chunk IDs must not contain duplicates.")

    results = {}

    for k in cutoffs:
        ranked = list(retrieved_ids[:k])

        relevant_count = sum(
            1 for chunk_id in ranked if chunk_id in relevance
        )

        precision = relevant_count / k
        recall = relevant_count / len(relevance)
        hit_rate = float(relevant_count > 0)

        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(ranked, start=1):
            if chunk_id in relevance:
                reciprocal_rank = 1.0 / rank
                break

        dcg = sum(
            (2 ** relevance[chunk_id] - 1)
            / math.log2(rank + 1)
            for rank, chunk_id in enumerate(ranked, start=1)
            if chunk_id in relevance
        )

        ideal_scores = sorted(
            relevance.values(),
            reverse=True,
        )[:k]

        ideal_dcg = sum(
            (2 ** score - 1) / math.log2(rank + 1)
            for rank, score in enumerate(ideal_scores, start=1)
        )

        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

        results[str(k)] = {
            "precision": precision,
            "recall": recall,
            "mrr": reciprocal_rank,
            "ndcg": ndcg,
            "hit_rate": hit_rate,
        }

    return results

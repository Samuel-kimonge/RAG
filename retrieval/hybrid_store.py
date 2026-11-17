from __future__ import annotations

import logging

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from retrieval.sparse_store import (
    DEFAULT_SPARSE_DB,
    SQLiteFTS5Index,
)
from retrieval.vector_store import (
    KnowledgeBase,
    RetrievalResult,
    get_knowledge_base,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridRetrievalResult:
    """A document ranked by dense and sparse retrieval fusion."""

    chunk_id: str
    document: Document
    fused_score: float
    dense_rank: int | None
    sparse_rank: int | None
    distance: float | None = None
    bm25_score: float | None = None

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


def reciprocal_rank_fusion(
    dense_ids: Sequence[str],
    sparse_ids: Sequence[str],
    *,
    rrf_k: int = 60,
) -> dict[str, float]:
    """
    Combine two ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (rrf_k + rank))

    Ranks start at 1. Higher fused scores indicate better results.
    """

    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer.")

    scores: dict[str, float] = {}

    for method, ranked_ids in (
        ("dense", dense_ids),
        ("sparse", sparse_ids),
    ):
        seen: set[str] = set()

        for rank, chunk_id in enumerate(ranked_ids, start=1):

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"{method} retrieval returned an invalid chunk ID."
                )

            if chunk_id in seen:
                raise ValueError(
                    f"{method} ranking contains duplicate chunk ID: "
                    f"{chunk_id}"
                )

            seen.add(chunk_id)

            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    return scores


class HybridKnowledgeBase:
    """
    Hybrid retrieval combining existing Chroma dense search
    with persistent SQLite FTS5/BM25 sparse search.
    """

    def __init__(
        self,
        dense_store: KnowledgeBase | None = None,
        *,
        sparse_db_path: Path = DEFAULT_SPARSE_DB,
        candidate_multiplier: int = 4,
        max_candidates: int = 100,
        rrf_k: int = 60,
    ) -> None:

        if (
            isinstance(candidate_multiplier, bool)
            or not isinstance(candidate_multiplier, int)
            or candidate_multiplier < 1
        ):
            raise ValueError(
                "candidate_multiplier must be a positive integer."
            )

        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates < 1
        ):
            raise ValueError(
                "max_candidates must be a positive integer."
            )

        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, int)
            or rrf_k < 1
        ):
            raise ValueError("rrf_k must be a positive integer.")

        self.dense_store = dense_store or get_knowledge_base()

        self.sparse_store = SQLiteFTS5Index(
            collection=self.dense_store.collection,
            db_path=sparse_db_path,
        )

        self.candidate_multiplier = candidate_multiplier
        self.max_candidates = max_candidates
        self.rrf_k = rrf_k

        logger.info(
            "Hybrid retrieval initialized. Sparse index: %s",
            self.sparse_store.last_sync_stats,
        )

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        file_type: str | None = None,
        source: str | None = None,
    ) -> list[HybridRetrievalResult]:
        """
        Retrieve dense and sparse candidates, fuse their rankings,
        and return the top k unique documents.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

        candidate_k = max(
            k,
            min(
                k * self.candidate_multiplier,
                self.max_candidates,
            ),
        )

        # Both retrievers apply the same metadata filters.
        dense_results: list[RetrievalResult] = self.dense_store.search(
            query,
            k=candidate_k,
            file_type=file_type,
            source=source,
        )

        sparse_results = self.sparse_store.search(
            query,
            k=candidate_k,
            file_type=file_type,
            source=source,
        )

        dense_ids = [
            result.chunk_id
            for result in dense_results
        ]

        sparse_ids = [
            result.chunk_id
            for result in sparse_results
        ]

        fused_scores = reciprocal_rank_fusion(
            dense_ids,
            sparse_ids,
            rrf_k=self.rrf_k,
        )

        dense_by_id = {
            result.chunk_id: result
            for result in dense_results
        }

        sparse_by_id = {
            result.chunk_id: result
            for result in sparse_results
        }

        dense_ranks = {
            result.chunk_id: rank
            for rank, result in enumerate(dense_results, start=1)
        }

        sparse_ranks = {
            result.chunk_id: rank
            for rank, result in enumerate(sparse_results, start=1)
        }

        def tie_breaker(chunk_id: str) -> tuple[int, str]:
            best_rank = min(
                dense_ranks.get(chunk_id, float("inf")),
                sparse_ranks.get(chunk_id, float("inf")),
            )

            return int(best_rank), chunk_id

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                tie_breaker(chunk_id),
            ),
        )

        output: list[HybridRetrievalResult] = []

        for chunk_id in ranked_ids[:k]:

            dense_result = dense_by_id.get(chunk_id)
            sparse_result = sparse_by_id.get(chunk_id)

            # Prefer the dense document when available.
            # Sparse-only results remain eligible.
            document = (
                dense_result.document
                if dense_result is not None
                else sparse_result.document
            )

            output.append(
                HybridRetrievalResult(
                    chunk_id=chunk_id,
                    document=document,
                    fused_score=fused_scores[chunk_id],
                    dense_rank=dense_ranks.get(chunk_id),
                    sparse_rank=sparse_ranks.get(chunk_id),
                    distance=(
                        dense_result.distance
                        if dense_result is not None
                        else None
                    ),
                    bm25_score=(
                        sparse_result.bm25_score
                        if sparse_result is not None
                        else None
                    ),
                )
            )

        return output

from __future__ import annotations

import numpy as np
import pytest
import threading

from types import SimpleNamespace
from langchain_core.documents import Document

from retrieval.reranker import CrossEncoderReranker


class FakeEncoding:
    def __init__(self, ids, type_ids, attention_mask):
        self.ids = ids
        self.type_ids = type_ids
        self.attention_mask = attention_mask


class FakeTokenizer:
    def enable_truncation(self, max_length, strategy="longest_first", direction="right"):
        self.truncation_strategy = strategy
        self.max_length = max_length
        self.truncation_strategies.append(strategy)

    def disable_truncation(self):
        self.truncation_strategy = None

    def no_padding(self):
        pass

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=text.split())

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)

    def enable_truncation(
        self,
        max_length,
        strategy="longest_first",
        direction="right",
    ):
        self.max_length = max_length
        self.truncation_strategy = strategy

    def enable_padding(
        self,
        direction="right",
        pad_id=0,
        pad_type_id=0,
        pad_token="[PAD]",
    ):
        pass

    def encode_batch(self, pairs):
        encodings = []

        for query, document in pairs:
            self.encoded_queries.append(query)

            # Simulate a higher relevance logit for a matching document.
            code = (
                9
                if "target fact" in document.lower()
                else 1
            )

            encodings.append(
                FakeEncoding(
                    ids=[101, 5, code, 102],
                    type_ids=[0, 0, 1, 1],
                    attention_mask=[1, 1, 1, 1],
                )
            )

        return encodings


class FakeSession:
    def __init__(self, should_fail=False):
        self.calls = 0
        self.should_fail = should_fail

    def run(self, output_names, inputs):
        self.calls += 1

        if self.should_fail:
            raise ValueError("Simulated ONNX inference failure.")

        scores = inputs["input_ids"][:, 2].astype(np.float32)

        return [scores.reshape(-1, 1)]


def make_candidate(
    chunk_id,
    content,
    *,
    fused_score=0.01,
    dense_rank=1,
    sparse_rank=1,
):
    return SimpleNamespace(
        chunk_id=chunk_id,
        document=Document(
            page_content=content,
            metadata={"source": "test.pdf"},
        ),
        distance=0.2,
        fused_score=fused_score,
        dense_rank=dense_rank,
        sparse_rank=sparse_rank,
        bm25_score=-0.5,
    )


def make_reranker(batch_size=2, should_fail=False):
    """
    Construct a reranker with fake inference components.

    This tests ranking behavior independently of downloading/loading
    the actual ONNX model.
    """

    reranker = object.__new__(CrossEncoderReranker)

    reranker.batch_size = batch_size
    reranker.max_length = 384
    reranker.max_query_tokens = 128

    reranker._tokenizer = FakeTokenizer()
    reranker._pad_token_id = 0

    reranker._session = FakeSession(
        should_fail=should_fail
    )

    reranker._output_name = "logits"
    reranker._lock = threading.RLock()

    return reranker


def test_relevant_document_is_ranked_first():
    reranker = make_reranker()

    candidates = [
        make_candidate(
            "chunk-low",
            "An unrelated document about employee benefits.",
            dense_rank=1,
            sparse_rank=1,
        ),
        make_candidate(
            "chunk-high",
            "This document contains the target fact.",
            dense_rank=2,
            sparse_rank=3,
        ),
    ]

    results = reranker.rerank(
        "What is the relevant fact?",
        candidates,
        top_k=2,
    )

    assert results[0].chunk_id == "chunk-high"
    assert results[1].chunk_id == "chunk-low"

    assert results[0].relevance_score > results[1].relevance_score


def test_original_ranking_information_is_preserved():
    reranker = make_reranker()

    candidate = make_candidate(
        "chunk-001",
        "This document contains the target fact.",
        fused_score=0.031,
        dense_rank=2,
        sparse_rank=1,
    )

    result = reranker.rerank(
        "Find the target fact.",
        [candidate],
    )[0]

    assert result.chunk_id == "chunk-001"
    assert result.original_rank == 1
    assert result.fused_score == 0.031
    assert result.dense_rank == 2
    assert result.sparse_rank == 1
    assert result.bm25_score == -0.5
    assert result.metadata["source"] == "test.pdf"


def test_inference_uses_configured_batch_size():
    reranker = make_reranker(batch_size=2)

    candidates = [
        make_candidate(
            f"chunk-{index}",
            "This document contains the target fact.",
        )
        for index in range(5)
    ]

    results = reranker.rerank(
        "Find the target fact.",
        candidates,
    )

    assert len(results) == 5

    # Five documents with batch size two require three ONNX calls.
    assert reranker._session.calls == 3


def test_only_document_sequence_is_truncated():
    reranker = make_reranker()

    reranker.max_query_tokens = 2

    reranker.rerank(
        "one two three four",
        [
            make_candidate(
                "chunk-001",
                "This document contains the target fact.",
            )
        ],
    )

    assert reranker._tokenizer.truncation_strategies == ["only_second"]
    assert reranker._tokenizer.encoded_queries == ["one two"]


def test_empty_candidates_return_empty_list():
    reranker = make_reranker()

    assert reranker.rerank(
        "Find something.",
        [],
    ) == []

    assert reranker._session.calls == 0


def test_duplicate_chunk_ids_are_rejected():
    reranker = make_reranker()

    candidates = [
        make_candidate("duplicate", "First document."),
        make_candidate("duplicate", "Second document."),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        reranker.rerank(
            "Find something.",
            candidates,
        )


def test_invalid_inputs_are_rejected():
    reranker = make_reranker()

    with pytest.raises(ValueError):
        reranker.rerank("", [])

    with pytest.raises(ValueError):
        reranker.rerank(
            "Find something.",
            [make_candidate("chunk-001", "Document.")],
            top_k=0,
        )


def test_inference_failure_is_not_silently_ignored():
    reranker = make_reranker(should_fail=True)

    candidates = [
        make_candidate(
            "chunk-001",
            "This document contains the target fact.",
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="Could not rerank documents",
    ):
        reranker.rerank(
            "Find the target fact.",
            candidates,
        )

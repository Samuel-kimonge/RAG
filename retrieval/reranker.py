from __future__ import annotations

import json
import logging
import threading
import time

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from langchain_core.documents import Document
from tokenizers import Tokenizer


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIRECTORY = (
    PROJECT_ROOT / "models" / "reranker" / "tinybert_l2_v2"
)

DEFAULT_MODEL_FILE = (
    "onnx/model_quint8_avx2.onnx"
)

REQUIRED_INPUTS = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
}


@dataclass(frozen=True)
class RerankedResult:
    """A retrieved document with its cross-encoder relevance score."""

    chunk_id: str
    document: Document
    relevance_score: float
    original_rank: int

    distance: float | None = None
    fused_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    bm25_score: float | None = None

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


class CrossEncoderReranker:
    """
    CPU-based cross-encoder reranker using ONNX Runtime.

    The model scores query-document pairs jointly.
    Higher logits indicate stronger predicted relevance.

    PyTorch and Transformers are not required.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIRECTORY,
        *,
        batch_size: int = 2,
        max_length: int = 384,
        max_query_tokens: int = 128,
        intra_op_num_threads: int = 2,
        inter_op_num_threads: int = 1,
    ) -> None:

        self.model_dir = Path(model_dir).resolve()
        self.model_path = self.model_dir / DEFAULT_MODEL_FILE

        self._validate_configuration(
            batch_size=batch_size,
            max_length=max_length,
            max_query_tokens=max_query_tokens,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
        )

        self.batch_size = batch_size
        self.max_length = max_length
        self.max_query_tokens = max_query_tokens

        self._lock = threading.RLock()

        self._validate_model_files()

        # Load the model's maximum supported sequence length.
        config_path = self.model_dir / "config.json"

        with config_path.open("r", encoding="utf-8") as file:
            model_config = json.load(file)

        model_max_length = int(
            model_config.get("max_position_embeddings", 512)
        )

        if self.max_length > model_max_length:
            raise ValueError(
                f"max_length={self.max_length} exceeds the model "
                f"limit of {model_max_length}."
            )

        # Load the fast tokenizer directly from tokenizer.json.
        tokenizer_path = self.model_dir / "tokenizer.json"

        self._tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self._pad_token_id = self._tokenizer.token_to_id("[PAD]")

        if self._pad_token_id is None:
            raise RuntimeError(
                "The tokenizer does not define the [PAD] token."
            )

        self._validate_runtime()

        options = ort.SessionOptions()

        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = inter_op_num_threads

        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        options.log_severity_level = 3

        # Use CPU explicitly. Do not attempt GPU initialization.
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        input_names = {
            item.name
            for item in self._session.get_inputs()
        }

        missing_inputs = REQUIRED_INPUTS - input_names
        unexpected_inputs = input_names - REQUIRED_INPUTS

        if missing_inputs or unexpected_inputs:
            raise RuntimeError(
                "Unexpected ONNX model input interface. "
                f"Missing={missing_inputs}, "
                f"unexpected={unexpected_inputs}."
            )

        outputs = self._session.get_outputs()

        if not outputs:
            raise RuntimeError(
                "The ONNX model exposes no output tensors."
            )

        self._output_name = outputs[0].name

        logger.info(
            "Cross-encoder initialized: model=%s, batch_size=%d, "
            "max_length=%d, CPU threads=%d",
            self.model_path,
            self.batch_size,
            self.max_length,
            intra_op_num_threads,
        )

    @staticmethod
    def _validate_configuration(
        *,
        batch_size: int,
        max_length: int,
        max_query_tokens: int,
        intra_op_num_threads: int,
        inter_op_num_threads: int,
    ) -> None:

        values = {
            "batch_size": batch_size,
            "max_length": max_length,
            "max_query_tokens": max_query_tokens,
            "intra_op_num_threads": intra_op_num_threads,
            "inter_op_num_threads": inter_op_num_threads,
        }

        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if max_length < 8:
            raise ValueError(
                "max_length must be at least 8."
            )

        if max_query_tokens >= max_length - 3:
            raise ValueError(
                "max_query_tokens must leave room for document tokens "
                "and the model's special tokens."
            )

    def _validate_model_files(self) -> None:
        required = [
            self.model_dir / "config.json",
            self.model_dir / "tokenizer.json",
            self.model_dir / "tokenizer_config.json",
            self.model_path,
        ]

        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Required reranker file missing or empty: {path}"
                )

    @staticmethod
    def _validate_runtime() -> None:
        providers = ort.get_available_providers()

        if "CPUExecutionProvider" not in providers:
            raise RuntimeError(
                "ONNX Runtime CPUExecutionProvider is unavailable. "
                f"Available providers: {providers}"
            )

    def _encode_batch(
        self,
        query: str,
        documents: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Tokenize query-document pairs.

        The query is preserved where possible. Long document text
        is truncated using the tokenizer's only_second strategy.
        """

        tokenizer = self._tokenizer

        tokenizer.disable_truncation()
        tokenizer.no_padding()

        try:
            query_ids = tokenizer.encode(
                query,
                add_special_tokens=False,
            ).ids

            # Prevent an unusually long query from consuming the
            # document's entire sequence budget.
            if len(query_ids) > self.max_query_tokens:
                query = tokenizer.decode(
                    query_ids[:self.max_query_tokens],
                    skip_special_tokens=True,
                ).strip()

            if not query:
                raise ValueError(
                    "Query became empty after token truncation."
                )

            tokenizer.enable_truncation(
                max_length=self.max_length,
                strategy="only_second",
                direction="right",
            )

            tokenizer.enable_padding(
                direction="right",
                pad_id=self._pad_token_id,
                pad_type_id=0,
                pad_token="[PAD]",
            )

            pairs = [
                (query, document)
                for document in documents
            ]

            encodings = tokenizer.encode_batch(pairs)

            input_ids = np.asarray(
                [encoding.ids for encoding in encodings],
                dtype=np.int64,
            )

            attention_mask = np.asarray(
                [
                    encoding.attention_mask
                    for encoding in encodings
                ],
                dtype=np.int64,
            )

            token_type_ids = np.asarray(
                [
                    encoding.type_ids
                    for encoding in encodings
                ],
                dtype=np.int64,
            )

            if (
                input_ids.ndim != 2
                or attention_mask.shape != input_ids.shape
                or token_type_ids.shape != input_ids.shape
            ):
                raise RuntimeError(
                    "Tokenizer produced inconsistent model input shapes."
                )

            if input_ids.shape[1] > self.max_length:
                raise RuntimeError(
                    "Tokenizer exceeded the configured sequence length."
                )

            return {
                "input_ids": np.ascontiguousarray(input_ids),
                "attention_mask": np.ascontiguousarray(attention_mask),
                "token_type_ids": np.ascontiguousarray(token_type_ids),
            }

        finally:
            # Reset mutable tokenizer settings after every batch.
            tokenizer.disable_truncation()
            tokenizer.no_padding()

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        *,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """
        Score and reorder retrieved candidates.

        Returns candidates ordered by descending cross-encoder logit.
        Original ranking is used as a deterministic tie-breaker.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string."
            )

        if isinstance(candidates, (str, bytes)):
            raise TypeError(
                "candidates must be a sequence of retrieval results."
            )

        candidates = list(candidates)

        if top_k is not None:
            if (
                isinstance(top_k, bool)
                or not isinstance(top_k, int)
                or top_k < 1
            ):
                raise ValueError(
                    "top_k must be a positive integer or None."
                )

        if not candidates:
            return []

        seen_ids: set[str] = set()
        documents: list[str] = []

        # Validate candidates before starting inference.
        for candidate in candidates:
            chunk_id = getattr(candidate, "chunk_id", None)

            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    "Every candidate must have a valid chunk_id."
                )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate candidate chunk ID: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            document = getattr(candidate, "document", None)

            if document is None or not isinstance(
                getattr(document, "page_content", None),
                str,
            ):
                raise ValueError(
                    f"Candidate {chunk_id} has no valid Document."
                )

            metadata = getattr(document, "metadata", None)

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Candidate {chunk_id} has invalid metadata."
                )

            documents.append(document.page_content)

        all_scores: list[float] = []

        started = time.perf_counter()

        try:
            # Tokenizer settings are mutable, so protect the complete
            # tokenization and inference operation from concurrent calls.
            with self._lock:

                for offset in range(
                    0,
                    len(candidates),
                    self.batch_size,
                ):

                    batch_documents = documents[
                        offset:offset + self.batch_size
                    ]

                    inputs = self._encode_batch(
                        query.strip(),
                        batch_documents,
                    )

                    logits = self._session.run(
                        [self._output_name],
                        inputs,
                    )[0]

                    scores = np.asarray(
                        logits,
                        dtype=np.float32,
                    ).reshape(-1)

                    if len(scores) != len(batch_documents):
                        raise RuntimeError(
                            "ONNX output count does not match input batch size."
                        )

                    if not np.all(np.isfinite(scores)):
                        raise RuntimeError(
                            "ONNX model returned non-finite relevance scores."
                        )

                    all_scores.extend(
                        float(score)
                        for score in scores
                    )

        except Exception as error:
            logger.exception(
                "Cross-encoder inference failed."
            )

            raise RuntimeError(
                "Could not rerank documents using the ONNX cross-encoder. "
                "The caller should decide whether to fail the request "
                "or explicitly use a fallback retrieval strategy."
            ) from error

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        logger.debug(
            "Reranked %d candidates in %.2f ms.",
            len(candidates),
            elapsed_ms,
        )

        ranked = list(
            zip(
                candidates,
                all_scores,
                range(1, len(candidates) + 1),
            )
        )

        # Higher logits are better. Original RRF rank breaks ties.
        ranked.sort(
            key=lambda item: (
                -item[1],
                item[2],
                item[0].chunk_id,
            )
        )

        if top_k is not None:
            ranked = ranked[:top_k]

        return [
            RerankedResult(
                chunk_id=candidate.chunk_id,
                document=candidate.document,
                relevance_score=score,
                original_rank=original_rank,
                distance=getattr(candidate, "distance", None),
                fused_score=getattr(candidate, "fused_score", None),
                dense_rank=getattr(candidate, "dense_rank", None),
                sparse_rank=getattr(candidate, "sparse_rank", None),
                bm25_score=getattr(candidate, "bm25_score", None),
            )
            for candidate, score, original_rank in ranked
        ]

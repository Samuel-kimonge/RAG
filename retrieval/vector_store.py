from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


logger = logging.getLogger(__name__)

# Resolve paths relative to the project, not the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIRECTORY = PROJECT_ROOT / "chromadb"

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
EXPECTED_EMBEDDING_DIMENSIONS = 768


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved document, its ChromaDB ID and similarity distance."""

    chunk_id: str
    document: Document
    distance: float

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def metadata(self) -> dict[str, Any]:
        return self.document.metadata


class KnowledgeBase:
    """Production-oriented LangChain interface to the existing Chroma index."""

    def __init__(self) -> None:
        self._validate_database_path()

        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIRECTORY)
        )

        self._validate_collection()

        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Reuse the existing Chroma client and collection.
        # Do not silently create a new, empty collection.
        self.vector_store = Chroma(
            client=self._client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            create_collection_if_not_exists=False,
        )

        logger.info(
            "Connected to collection '%s' containing %d chunks.",
            COLLECTION_NAME,
            self._collection.count(),
        )

    @property
    def collection(self):
        """Return the existing ChromaDB collection."""
        return self._client.get_collection(COLLECTION_NAME)

    @property
    def _collection(self):
        """Backward-compatible internal alias."""
        return self.collection

    @staticmethod
    def _validate_database_path() -> None:
        if not CHROMA_DIRECTORY.is_dir():
            raise FileNotFoundError(
                f"Chroma database directory not found: {CHROMA_DIRECTORY}"
            )

    def _validate_collection(self) -> None:
        try:
            collection = self._client.get_collection(COLLECTION_NAME)
        except Exception as error:
            raise RuntimeError(
                f"Existing Chroma collection '{COLLECTION_NAME}' "
                "could not be opened. Check the database path and collection name."
            ) from error

        count = collection.count()

        if count == 0:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' is empty. "
                "Refusing to continue with an empty knowledge base."
            )

        metadata = collection.metadata or {}

        if metadata.get("hnsw:space") != "cosine":
            logger.warning(
                "Expected cosine distance, but collection metadata is %s.",
                metadata,
            )

    @staticmethod
    def _build_metadata_filter(
        file_type: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a valid Chroma metadata filter."""

        conditions = []

        if file_type is not None:
            file_type = file_type.strip().lower()

            if not file_type:
                raise ValueError("file_type cannot be empty.")

            conditions.append({"file_type": file_type})

        if source is not None:
            source = source.strip()

            if not source:
                raise ValueError("source cannot be empty.")

            conditions.append({"source": source})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def search(
          self,
          query: str,
          *,
          k: int = 5,
          file_type: str | None = None,
          source: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve ranked chunks while preserving their original ChromaDB IDs.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

        metadata_filter = self._build_metadata_filter(
            file_type=file_type,
            source=source,
         )

        try:
            available_chunks = self._collection.count()
            result_count = min(k, available_chunks)

            if result_count == 0:
                return []

            # Generate embeddings through LangChain.
            query_embedding = self.embeddings.embed_query(query.strip())

            # Query the existing Chroma collection directly.
            response = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=result_count,
                where=metadata_filter,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            ids = response["ids"][0]
            documents = response["documents"][0]
            metadatas = response["metadatas"][0]
            distances = response["distances"][0]

            return [
                RetrievalResult(
                    chunk_id=chunk_id,
                    document=Document(
                        page_content=document,
                        metadata=metadata or {},
                    ),
                    distance=float(distance),
                )
                for chunk_id, document, metadata, distance in zip(
                    ids,
                    documents,
                    metadatas,
                    distances,
                )
            ]

        except Exception as error:
            logger.exception("Document retrieval failed.")

            raise RuntimeError(
                "Could not retrieve documents from the knowledge base. "
                "Check Ollama availability, embedding configuration "
                "and ChromaDB connectivity."
            ) from error


def get_knowledge_base() -> KnowledgeBase:
    """Create a knowledge-base interface."""
    return KnowledgeBase()
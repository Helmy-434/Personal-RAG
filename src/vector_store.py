
from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

class VectorStoreError(RuntimeError):
    """Raised when the Chroma vector store cannot be used."""


class ChromaVectorStore:
    def __init__(
        self,
        *,
        embedding_function: Embeddings,
        persist_directory: str | Path = "chroma_db",
        collection_name: str = "personal_rag",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name cannot be empty")

        self.persist_directory = Path(persist_directory).expanduser()
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._store = self._create_store(embedding_function)

    def add_chunks(self, chunks: Iterable[Document]) -> list[str]:
        documents = [self._validate_document(chunk) for chunk in chunks]
        if not documents:
            return []

        ids = [self._document_id(document) for document in documents]
        try:
            self._store.add_documents(documents=documents, ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"Could not add documents to Chroma: {exc}") from exc
        return ids


    def search_with_scores(
        self,
        query: str,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
       
        if not query.strip():
            raise ValueError("query cannot be empty")
        if k <= 0:
            raise ValueError("k must be greater than zero")

        search_kwargs: dict[str, Any] = {"k": k}
        if filter is not None:
            search_kwargs["filter"] = dict(filter)
        try:
            return self._store.similarity_search_with_score(query, **search_kwargs)
        except Exception as exc:
            raise VectorStoreError(f"Could not search Chroma: {exc}") from exc

    def delete_source(self, source: str) -> None:
        ids = self._store.get(where={"source": source})["ids"]
        if ids:
            self._store.delete(ids=ids)

    def recreate_collection(self) -> None:
        """Delete and recreate this collection, removing stale indexed data."""
        try:
            self._store.delete_collection()
            self._store = self._create_store(self.embedding_function)
        except Exception as exc:
            raise VectorStoreError(
                f"Could not recreate Chroma collection: {exc}"
            ) from exc

    def _create_store(self, embedding_function: Embeddings) -> Any:
        try:
            chroma_module = importlib.import_module("langchain_chroma") #imports chroma but has a fallback if not found
            chroma_class = chroma_module.Chroma
        except ImportError:
            try:
                community_module = importlib.import_module(
                    "langchain_community.vectorstores"
                )
                chroma_class = community_module.Chroma
            except ImportError as exc:
                raise VectorStoreError(
                    "Chroma support requires 'langchain-chroma'. Install it with: "
                    "pip install -U langchain-chroma chromadb"
                ) from exc

        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            return chroma_class(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                persist_directory=str(self.persist_directory),
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not open Chroma: {exc}") from exc

    @staticmethod
    def _validate_document(document: Document) -> Document:
        if not isinstance(document, Document):
            raise TypeError("add_chunks expects LangChain Document objects")
        if not document.page_content.strip():
            raise ValueError("Cannot add a document with empty page_content")
        return document

    @staticmethod
    def _document_id(document: Document) -> str:
        payload = {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
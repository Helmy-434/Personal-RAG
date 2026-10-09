
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:

    def __init__(
        self,
        *,
        chunk_size: int = 1_000, #drop to 800 if too many tokens
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n",".", " ", ""],
        )

    def chunk(self, document: Document) -> list[Document]: #splits document into chunks and add metadata
        text = document.page_content.strip()
        if not text:
            return []

        pieces = self._splitter.split_text(text)
        total = len(pieces)
        chunks: list[Document] = []
        for index, piece in enumerate(pieces):
            metadata: dict[str, Any] = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "chunk_count": total,
                    "char_count": len(piece),
                }
            )
            chunks.append(Document(page_content=piece, metadata=metadata))
        return chunks

    def chunk_many(self, documents: Iterable[Document]) -> list[Document]:#keeps order
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self.chunk(document))
        return chunks

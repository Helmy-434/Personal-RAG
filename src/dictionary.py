from __future__ import annotations

from pathlib import Path

from chunker import DocumentChunker
from embeddings import get_embeddings
from loaders import DocumentLoader
from vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "personal_rag_cosine"



def build_dictionary() -> int:
    if not DATA_DIRECTORY.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIRECTORY}")

    loader = DocumentLoader()
    documents = loader.load_directory(DATA_DIRECTORY)
    if not documents:
        raise ValueError(f"No supported documents found in {DATA_DIRECTORY}")

    chunks = DocumentChunker().chunk_many(documents)
    if not chunks:
        raise ValueError("The loaded documents produced no non-empty chunks")

    store = ChromaVectorStore(
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    store.recreate_collection()
    ids = store.add_chunks(chunks)

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Stored chunks: {len(ids)}")
    print(f"Chroma directory: {CHROMA_DIRECTORY}")
    return len(ids)


if __name__ == "__main__":
    build_dictionary()
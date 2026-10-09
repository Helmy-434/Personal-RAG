"""Streamlit question-answering interface for the Personal-RAG index.

Start from the project root with:

    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.embeddings import get_embeddings
from src.rag import ask
from src.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "personal_rag_cosine"

st.set_page_config(page_title="Personal RAG", page_icon="📚")
st.title("Personal RAG")
st.caption("Ask questions about the documents in your local knowledge base.")


@st.cache_resource
def get_store() -> ChromaVectorStore:
    """Create one cached vector-store connection for the Streamlit process."""
    return ChromaVectorStore(
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )


def format_source(document: object, score: float) -> str:
    metadata = getattr(document, "metadata", {})
    source = Path(str(metadata.get("source", "unknown"))).name
    page = metadata.get("page")
    location = f"{source}, page {page}" if page else source
    return f"{location} (distance: {score:.4f})"


with st.sidebar:
    st.header("Settings")
    result_count = st.slider("Retrieved chunks", min_value=1, max_value=10, value=4)

    if st.button("Clear conversation"):
        st.session_state.pop("messages", None)
        st.rerun()

if not os.getenv("OPENROUTER_API_KEY"):
    st.warning(
        "Set OPENROUTER_API_KEY in your .env file before asking questions."
    )

if not CHROMA_DIRECTORY.exists():
    st.info(
        "The vector index does not exist yet. Run `python data.py` first, "
        "then refresh this page."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            store = get_store()
            with st.spinner("Searching your documents..."):
                answer, results = ask(store, question, k=result_count)
            sources = [
                format_source(document, score) for document, score in results
            ]
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(source)
        except Exception as exc:
            answer = (
                "I could not answer that question. Check the API key, dependencies, "
                "and whether the index was built with `python data.py`."
            )
            sources = []
            st.error(f"{answer}\n\nDetails: {exc}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
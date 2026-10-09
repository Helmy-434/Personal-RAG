from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.embeddings import get_embeddings
from src.loaders import clean_source_name
from src.rag import ask
from src.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "personal_rag_cosine"

st.set_page_config(
    page_title="Meet Youssef Helmy",
    page_icon="📚",
)
st.title("Meet Youssef Helmy")
st.caption("Get to know me more")


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
    source = clean_source_name(metadata.get("source", "unknown"))
    page = metadata.get("page")
    location = f"{source}, page {page}" if page else source
    return f"{location} (distance: {score:.4f})"


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

question = st.chat_input("Ask a question about Youssef")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            store = get_store()
            with st.spinner("Searching your documents..."):
                answer, results = ask(store, question, k=4)
            sources = [
                format_source(document, score) for document, score in results
            ]
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(source)
        except Exception as exc:
            error_text = str(exc)
            if "429" in error_text or "rate-limited" in error_text.lower():
                answer = (
                    "The chat model is temporarily rate-limited by OpenRouter. "
                    "Try again shortly or choose another model in OPENROUTER_CHAT_MODEL."
                )
            else:
                answer = (
                    "I could not answer that question. Check the API key, dependencies, "
                    "and whether the index was built with `python src/dictionary.py`."
                )
            sources = []
            st.error(f"{answer}\n\nDetails: {exc}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
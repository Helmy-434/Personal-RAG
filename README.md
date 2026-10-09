# Personal RAG

A personal question-answering app that loads local documents, chunks them,
embeds them with OpenRouter, stores them in Chroma, and answers questions
through a Streamlit interface.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```


## Add documents

Place supported local files in [data/](./data/). The loader currently supports:

- PDF
- Plain text
- Markdown
- HTML files
- JSON
- CSV

## Build the vector database

Run the indexing script after adding or changing documents:

```powershell
python src/dictionary.py
```

This performs:

1. Load documents
2. Split them into chunks
3. Create OpenRouter embeddings
4. Store the chunks in the persistent `chroma_db/` directory

Rebuild the database after changing the embedding model or loader metadata.

## Run the app

```powershell
python -m streamlit run app.py
```

Open the local URL shown by Streamlit and ask questions about the indexed
documents.

## Project flow

```text
data/ documents
      ↓
src/loaders.py
      ↓
src/chunker.py
      ↓
src/embeddings.py
      ↓
src/vector_store.py
      ↓
src/rag.py
      ↓
app.py
```


```env
RAG_MAX_DISTANCE=0.9
```

Lower values are stricter; higher values retrieve more results.
